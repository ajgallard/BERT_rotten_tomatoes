import torch
import pytorch_lightning as pl
import logging
from typing import List
import multiprocessing
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd

"""
Dataset Class for BERT
"""
class BERTDataset(Dataset):
    def __init__(
            self,
            model_name: str,
            contents: List[str],
            labels: List[str] = None,
            max_seq_length: int = 512):
        self.contents = contents
        self.labels = labels
        self.model_name = model_name
        self.max_seq_length = max_seq_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logging.getLogger('transformers.tokenization_utils').setLevel(logging.FATAL)

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, index):
        content = self.contents[index]

        inputs = self.tokenizer.encode_plus(
            content,
            add_special_tokens=True,
            padding='max_length',
            max_length=self.max_seq_length,
            truncation=True,
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        BERT_dict = {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'labels': torch.tensor(self.labels[index], dtype=torch.long)
        }

        return BERT_dict


"""
DataModule Class for Lightning Trainer
"""
class BERTDataModule(pl.LightningDataModule):
    def __init__(self, params, colab: bool = True):
        super().__init__()
        """Setting Up Hyperparameters from YAML file"""
        if colab is True:
            print('Colab Parameters Enabled')
            self.params = params[0]['colab']
            self.num_workers = int(multiprocessing.cpu_count())
            print(f'Number of Workers Set to: {self.num_workers}')
        else:
            self.params = params[0]['data']
            print('Local Parameters Enabled')
            self.num_workers = 1
            print(f'Number of Workers Set to: {self.num_workers}')

        self.model_name = self.params['model_name']
        self.batch_size = self.params['batch_size']
        self.content_name = self.params['content_name']
        self.label_name = self.params['label_name']
        self.num_labels = self.params['num_labels']
        self.max_seq_length = self.params['max_seq_length']
        self.filepath = self.params['filepath']
        self.csv_names = self.params['csv_names']

    def setup(self, stage=None):

        print("Reading .csv files...")
        train_df = pd.read_csv(f"{self.filepath}{self.csv_names['train']}")
        val_df = pd.read_csv(f"{self.filepath}{self.csv_names['val']}")
        test_df = pd.read_csv(f"{self.filepath}{self.csv_names['test']}")

        print("Building Datasets...")
        self.train_dataset = BERTDataset(model_name=self.model_name,
                                         contents=train_df[self.content_name].values,
                                         labels=train_df[self.label_name].values,
                                         max_seq_length=self.max_seq_length)

        self.val_dataset = BERTDataset(model_name=self.model_name,
                                       contents=val_df[self.content_name].values,
                                       labels=val_df[self.label_name].values,
                                       max_seq_length=self.max_seq_length)

        self.test_dataset = BERTDataset(model_name=self.model_name,
                                        contents=test_df[self.content_name].values,
                                        labels=test_df[self.label_name].values,
                                        max_seq_length=self.max_seq_length)

    def train_dataloader(self):
        print("Building Train Dataloader...")
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        print("Building Val Dataloader...")
        return DataLoader(dataset=self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        print("Building Test Dataloader...")
        return DataLoader(dataset=self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)
