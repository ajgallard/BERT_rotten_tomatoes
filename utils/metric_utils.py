from typing import List

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import sklearn.metrics as metrics
import torch


class MetricUtils:

    @staticmethod
    def prediction_metrics(trainer: pl.trainer,
                           model: pl.LightningModule,
                           datamodule: pl.LightningDataModule,
                           class_names: List = None,
                           ):

        print("Generating Predictions & Labels...")
        pred_labels = trainer.predict(model=model, datamodule=datamodule, return_predictions=True)

        pred_list = []
        label_list = []
        for (preds, labels) in pred_labels:
            pred_list.append(preds)
            label_list.append(labels)


        predictions = torch.cat(pred_list)
        labels = torch.cat(label_list)

        print("Generating Confusion Matrix & Classification Report...")
        cm = metrics.confusion_matrix(labels, predictions)
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        report = metrics.classification_report(labels, predictions, target_names=class_names)

        disp.plot()
        plt.show()
        print(report)
        return (disp, report)

    @staticmethod
    def testing_metrics(trainer: pl.trainer,
                        model: pl.LightningModule,
                        datamodule: pl.LightningDataModule,
                        class_names: List = None,
                        ):

        print("Generating Predictions & Labels...")
        pred_labels = trainer.test(model=model, datamodule=datamodule, return_predictions=True)

        pred_list = []
        label_list = []
        for (preds, labels) in pred_labels:
            pred_list.append(preds)
            label_list.append(labels)

        predictions = torch.cat(pred_list)
        labels = torch.cat(label_list)

        print("Generating Confusion Matrix & Classification Report...")
        cm = metrics.confusion_matrix(labels, predictions)
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        report = metrics.classification_report(labels, predictions, target_names=class_names)

        disp.plot()
        plt.show()
        print(report)
        return (disp, report)
