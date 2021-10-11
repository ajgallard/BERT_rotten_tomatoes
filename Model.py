import torch
import torch.nn as nn
from transformers import DistilBertForSequenceClassification as DBSC
import pytorch_lightning as pl
from pytorch_lightning import callbacks
import torchmetrics
from torch.optim import AdamW


"""
Callbacks
"""
early_stopping = callbacks.EarlyStopping(monitor="Val Accuracy", patience=5, verbose=False, mode="max")

"""
DistilBERT Model + Lightning
"""
class DistilBERT(pl.LightningModule):
    def __init__(self, params):
        super(DistilBERT, self).__init__()
        """Setting Up Hyperparameters from YAML file"""
        self.params1 = params[1]['model']
        self.model_name = self.params1["model_name"]
        self.num_labels = self.params1["num_labels"]

        self.params2 = params[2]['classifier']
        self.learning_rate = self.params2['learning_rate']

        """Instantiating Model"""
        model = DBSC.from_pretrained(self.model_name,
                                     num_labels=self.num_labels)
        self.model = model

    """
    LIGHTNING CLASSIFIER SECTION
    """
    def configure_optimizers(self):
        print("Configuring Optimizer...")
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):

        input_ids = batch['ids']
        attention_mask = batch['mask']
        labels = batch['labels']

        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask)

        # loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs.logits, labels)

        # accuracy
        preds = []
        for i in range(len(outputs.logits)):
            label_result = outputs.logits[i]
            if label_result[0] > label_result[1]:
                preds.append(0)
            else:
                preds.append(1)

        preds = torch.tensor(preds)
        metric = torchmetrics.Accuracy()
        acc = metric(preds.cpu(), labels.cpu())
        self.log("Train Loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("Train Accuracy", acc, prog_bar=True, on_step=False, on_epoch=True)
        return dict(loss=loss, train_acc=acc)

    def validation_step(self, batch, batch_idx):

        input_ids = batch['ids']
        attention_mask = batch['mask']
        labels = batch['labels']

        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask)

        # loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs.logits, labels)

        # accuracy
        preds = []
        for i in range(len(outputs.logits)):
            label_result = outputs.logits[i]
            if label_result[0] > label_result[1]:
                preds.append(0)
            else:
                preds.append(1)

        preds = torch.tensor(preds)
        metric = torchmetrics.Accuracy()
        acc = metric(preds.cpu(), labels.cpu())
        self.log("Val Loss", loss, prog_bar=True, on_epoch=True)
        self.log("Val Accuracy", acc, prog_bar=True, on_epoch=True)
        return dict(loss=loss, train_acc=acc)

    def test_step(self, batch, batch_idx):

        input_ids = batch['ids']
        attention_mask = batch['mask']
        labels = batch['labels']

        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask)

        preds = []
        for i in range(len(outputs.logits)):
            label_result = outputs.logits[i]
            if label_result[0] > label_result[1]:
                preds.append(0)
            else:
                preds.append(1)

        preds = torch.tensor(preds)
        return [preds.cpu(), labels.cpu()]
    
    def predict_step(self, batch, batch_idx):

        input_ids = batch['ids']
        attention_mask = batch['mask']
        labels = batch['labels']
        
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask)

        preds = []
        for i in range(len(outputs.logits)):
            label_result = outputs.logits[i]
            if label_result[0] > label_result[1]:
                preds.append(0)
            else:
                preds.append(1)

        preds = torch.tensor(preds)
        return [preds.cpu(), labels.cpu()]  # return predictions & labels for metrics utils
