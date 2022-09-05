import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ClassificationDatasetInfer
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
    data = pd.read_csv('../data/goods_classifier/test.csv')
    BATCH_SIZE = 128


    model = BertForSequenceClassification.from_pretrained(
            "DeepPavlov/rubert-base-cased",
            num_labels=9,
            output_attentions=False,
            output_hidden_states=False,
        )

    weights = torch.load('logs/models/radam/model_epoch_0.pt')

    weights = {key.replace('module.', ''): value for key, value in weights.items()}

    model.load_state_dict(weights)
    model.cuda()
    model.eval()

    val_dataset = ClassificationDatasetInfer(data)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        worker_init_fn=worker_init_fn
    )

    all_preds = np.array([])
    for val_input in tqdm(val_dataloader, total=len(val_dataloader)):
        mask = val_input['attention_mask'].cuda()
        input_id = val_input['input_ids'].squeeze(1).cuda()

        with torch.no_grad():
            val_output = model(
                input_id,
                token_type_ids=None,
                attention_mask=mask,
                labels=None,
            )

        logits = val_output.logits.detach().cpu().numpy()

        preds = np.argmax(logits, axis=1).flatten()
        all_preds = np.concatenate([all_preds, preds])

    print(len(val_dataset))
    print(len(all_preds))

    with Path('../data/goods_classifier/test_output.pickle').open('wb') as file:
        pickle.dump(all_preds, file)



if __name__ == "__main__":
    train_loop()
