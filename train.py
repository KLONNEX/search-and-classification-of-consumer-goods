import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import RAdam
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import ClassificationDataset
from transformers import BertForSequenceClassification

os.environ["TOKENIZERS_PARALLELISM"] = "false"
np.random.seed(42)


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
    tb_logger = SummaryWriter(log_dir='logs/tb_logger/radam_all_data', comment='Training writer.')
    data = pd.read_csv('../data/goods_classifier/train.csv')
    EPOCH_NUM = 20
    BATCH_SIZE = 96

    # train_data, val_data = np.split(data.sample(frac=1, random_state=42), [int(.9 * len(data))])
    train_data = data

    model = BertForSequenceClassification.from_pretrained(
            "DeepPavlov/rubert-base-cased",
            num_labels=9,
            output_attentions=False,
            output_hidden_states=False,
        )

    train_dataset = ClassificationDataset(train_data)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        worker_init_fn=worker_init_fn
    )

    # val_dataset = ClassificationDataset(val_data)
    # val_dataloader = DataLoader(
    #     dataset=val_dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    #     drop_last=False,
    #     num_workers=8,
    #     worker_init_fn=worker_init_fn
    # )

    num_batches_total = EPOCH_NUM * len(train_dataloader)
    print(len(train_dataloader))
    # print(len(val_dataset))

    param_groups = [
        {'params': model.bert.parameters(), 'lr': 5e-6, 'name': 'backbone'},
        {'params': model.classifier.parameters(), 'lr': 5e-4, 'name': 'head'}
    ]

    optimizer = RAdam(param_groups, lr=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_batches_total)

    _, classes_counts = np.unique(np.array(train_dataset.labels), return_counts=True)
    classes_weight = ((classes_counts.sum() / classes_counts) / (classes_counts.sum() / classes_counts).sum())

    print(f'Classes balance {classes_counts}')
    print(f'Classes weights {classes_weight}')

    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(classes_weight.astype(np.float32))).cuda()

    model = nn.DataParallel(model)
    model = model.cuda()

    for epoch in range(EPOCH_NUM):
        for step, (train_input, train_label) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            global_step = epoch * len(train_dataloader) + step

            train_label = train_label.cuda()
            mask = train_input['attention_mask'].cuda()
            input_id = train_input['input_ids'].squeeze(1).cuda()

            output = model(
                input_id,
                token_type_ids=None,
                attention_mask=mask,
                labels=None,
            )

            logits = output.logits

            batch_loss = criterion(logits, train_label.long())

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
            # scheduler.step()

            for param_lr in optimizer.param_groups:
                curr_lr = param_lr['lr']

            tb_logger.add_scalar(
                tag='Loss',
                scalar_value=batch_loss,
                global_step=global_step,
            )

            tb_logger.add_scalar(
                tag='LR',
                scalar_value=curr_lr,
                global_step=global_step,
            )

            # if global_step % 4000 == 0 and epoch != 0:
            #     model.eval()
            #     val_true = 0
            #     for val_input, val_label in tqdm(val_dataloader, total=len(val_dataloader)):
            #         val_label = val_label.cuda()
            #         mask = val_input['attention_mask'].cuda()
            #         input_id = val_input['input_ids'].squeeze(1).cuda()
            #
            #         with torch.no_grad():
            #             val_output = model(
            #                 input_id,
            #                 token_type_ids=None,
            #                 attention_mask=mask,
            #                 labels=val_label.long(),
            #             )
            #
            #         logits = val_output.logits.detach().cpu().numpy()
            #         labels = val_label.cpu().numpy()
            #
            #         preds = np.argmax(logits, axis=1).flatten()
            #         labels = labels.flatten()
            #
            #         val_true += (preds == labels).sum()
            #
            #     accuracy = val_true / len(val_dataset)
            #     tb_logger.add_scalar(
            #         tag='Accuracy/Val',
            #         scalar_value=accuracy,
            #         global_step=global_step,
            #     )
            #
            #     model.train()

        torch.save(model.state_dict(), f'logs/models/radam_all_data/model_epoch_{epoch}.pt')
        # torch.nn.DataParallel()


if __name__ == "__main__":
    train_loop()
