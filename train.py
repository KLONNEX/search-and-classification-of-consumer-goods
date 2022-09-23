import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import RAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from cfg.config import config
from dataset import ClassificationDataset
from transformers import BertForSequenceClassification

os.environ["TOKENIZERS_PARALLELISM"] = "false"
np.random.seed(42)
torch.manual_seed(1)


def worker_init_fn(worker_id):
    """
    Initialize worker seed.

    Args:
        worker_id: Id of the current cpu worker.
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def train_loop():
    """
    Model training pipeline.
    """
    tb_logger_save_path = Path(config.logs_dir, 'tb_logger', config.exp_name)
    checkpoint_save_path = Path(config.logs_dir, 'checkpoints', config.exp_name)
    tb_logger_save_path.mkdir(parents=True, exist_ok=True)
    checkpoint_save_path.mkdir(parents=True, exist_ok=True)

    tb_logger = SummaryWriter(log_dir=str(tb_logger_save_path), comment='Training writer.')
    train_data = pd.read_csv(config.train_data_path)

    model = BertForSequenceClassification.from_pretrained(
            config.model_config,
            num_labels=config.num_classes,
            output_attentions=False,
            output_hidden_states=False,
        )

    train_dataset = ClassificationDataset(train_data, config.model_config, config.max_pad_len)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        worker_init_fn=worker_init_fn
    )

    learning_rate = config.learning_rate
    num_batches_total = config.epochs * len(train_dataloader)

    param_groups = [
        {'params': model.bert.parameters(), 'lr': learning_rate / config.backbone_lr_div, 'name': 'backbone'},
        {'params': model.classifier.parameters(), 'lr': learning_rate, 'name': 'head'}
    ]

    optimizer = RAdam(param_groups, lr=learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_batches_total)

    _, classes_counts = np.unique(np.array(train_dataset.labels), return_counts=True)
    classes_weight = ((classes_counts.sum() / classes_counts) / (classes_counts.sum() / classes_counts).sum())

    print(f'Classes balance {classes_counts}')
    print(f'Classes weights {classes_weight}')

    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(classes_weight.astype(np.float32))).cuda()

    model = nn.DataParallel(model)
    model = model.cuda()
    model.train()

    for epoch in range(config.epochs):
        for step, (train_input, train_label) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            global_step = epoch * len(train_dataloader) + step

            if global_step < config.warmup_steps:
                lr = learning_rate * (step / config.warmup_steps) ** 4
                for param_lr in optimizer.param_groups:
                    if param_lr['name'] == 'backbone':
                        param_lr['lr'] = lr / config.backbone_lr_div
                    elif param_lr['name'] == 'head':
                        param_lr['lr'] = lr

            train_label = train_label.cuda()
            mask = train_input['attention_mask'].cuda()
            input_id = train_input['input_ids'].squeeze(1).cuda()

            output = model(
                input_id,
                token_type_ids=None,
                attention_mask=mask,
                labels=None,
            )

            batch_loss = criterion(output.logits, train_label.long())

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
            scheduler.step()

            curr_lrs = {}
            for param_lr in optimizer.param_groups:
                curr_lrs[param_lr['name']] = param_lr['lr']

            tb_logger.add_scalar(
                tag='Loss',
                scalar_value=batch_loss,
                global_step=global_step,
            )

            tb_logger.add_scalars(
                main_tag='LR',
                tag_scalar_dict=curr_lrs,
                global_step=global_step,
            )

        torch.save(model.state_dict(), f'{checkpoint_save_path}/model_epoch_{epoch}.pt')


if __name__ == "__main__":
    train_loop()
