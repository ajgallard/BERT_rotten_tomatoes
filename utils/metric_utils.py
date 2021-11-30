from typing import List

import os
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import sklearn.metrics as metrics
import torch
import tempfile
import mlflow


class MetricUtils:

    @staticmethod
    def prediction_metrics(trainer: pl.trainer,
                           model: pl.LightningModule,
                           datamodule: pl.LightningDataModule,
                           class_names: List = None,
                           log: bool = True
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

        print("Generating Metrics...")
        cm = metrics.confusion_matrix(labels, predictions)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

        fpr, tpr, _ = metrics.roc_curve(labels, predictions)
        roc_auc = metrics.auc(fpr, tpr)
        roc_auc_display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)

        precision, recall, _ = metrics.precision_recall_curve(labels, predictions)
        precision_recall_display = metrics.PrecisionRecallDisplay(precision=precision, recall=recall)

        precision_micro = metrics.precision_score(labels, predictions, average="micro")
        precision_macro = metrics.precision_score(labels, predictions, average="macro")
        recall_micro = metrics.recall_score(labels, predictions, average="micro")
        recall_macro = metrics.recall_score(labels, predictions, average="macro")

        accuracy = metrics.accuracy_score(labels, predictions)
        f1_score = metrics.f1_score(labels, predictions, labels=class_names)

        report = metrics.classification_report(labels, predictions, target_names=class_names)
        statistics = dict(cm=cm,
                          fpr=fpr,
                          tpr=tpr,
                          roc_auc=roc_auc,
                          precision=precision,
                          recall=recall,
                          accuracy=accuracy,
                          f1_score=f1_score)

        if log is True:
            prefixes = ["cm_display", "roc_auc_display", "precision_recall_display"]
            suffix = ".png"
            plots = [cm_display, roc_auc_display, precision_recall_display]
            for idx, prefix in enumerate(prefixes):
                try:
                    temp = tempfile.NamedTemporaryFile(mode='w',
                                                       delete=False,
                                                       prefix=prefix,
                                                       suffix=suffix)
                    temp_name = temp.name
                    fig, ax = plt.subplots()
                    plots[idx].plot(ax=ax)
                    fig.savefig(temp_name)
                    mlflow.log_artifact(temp_name, prefix)
                finally:
                    temp.close()
                    os.unlink(temp_name)

            # logging for single value metrics
            metric_dict = dict(roc_auc=roc_auc,
                               precision_micro=precision_micro,
                               precision_macro=precision_macro,
                               recall_micro=recall_micro,
                               recall_macro=recall_macro,
                               accuracy=accuracy,
                               f1_score=f1_score
            )
            for k in metric_dict:
                try:
                    mlflow.log_metric(k, metric_dict[k])
                except:
                    print(f"ERROR: Metric {k} was not logged.")
                    pass

        return (cm_display, roc_auc_display, precision_recall_display, report, statistics)

    # TODO: revise testing metrics after prediction metrics
    @staticmethod
    def test_metrics(trainer: pl.trainer,
                     model: pl.LightningModule,
                     datamodule: pl.LightningDataModule,
                     class_names: List = None,
                     log: bool = True
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

        print("Generating Metrics...")
        cm = metrics.confusion_matrix(labels, predictions)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

        fpr, tpr, _ = metrics.roc_curve(labels, predictions)
        roc_auc = metrics.auc(fpr, tpr)
        roc_auc_display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)

        precision, recall, _ = metrics.precision_recall_curve(labels, predictions)
        precision_recall_display = metrics.PrecisionRecallDisplay(precision=precision, recall=recall)

        precision_micro = metrics.precision_score(labels, predictions, average="micro")
        precision_macro = metrics.precision_score(labels, predictions, average="macro")
        recall_micro = metrics.recall_score(labels, predictions, average="micro")
        recall_macro = metrics.recall_score(labels, predictions, average="macro")

        accuracy = metrics.accuracy_score(labels, predictions)
        f1_score = metrics.f1_score(labels, predictions, labels=class_names)

        report = metrics.classification_report(labels, predictions, target_names=class_names)
        statistics = dict(cm=cm,
                          fpr=fpr,
                          tpr=tpr,
                          roc_auc=roc_auc,
                          precision=precision,
                          recall=recall,
                          accuracy=accuracy,
                          f1_score=f1_score)

        if log is True:
            prefixes = ["cm_display", "roc_auc_display", "precision_recall_display"]
            suffix = ".png"
            plots = [cm_display, roc_auc_display, precision_recall_display]
            for idx, prefix in enumerate(prefixes):
                try:
                    temp = tempfile.NamedTemporaryFile(mode='w',
                                                       delete=False,
                                                       prefix=prefix,
                                                       suffix=suffix)
                    temp_name = temp.name
                    fig, ax = plt.subplots()
                    plots[idx].plot(ax=ax)
                    fig.savefig(temp_name)
                    mlflow.log_artifact(temp_name, prefix)
                finally:
                    temp.close()
                    os.unlink(temp_name)

            # logging for single value metrics
            metric_dict = dict(roc_auc=roc_auc,
                               precision_micro=precision_micro,
                               precision_macro=precision_macro,
                               recall_micro=recall_micro,
                               recall_macro=recall_macro,
                               accuracy=accuracy,
                               f1_score=f1_score
                               )
            for k in metric_dict:
                try:
                    mlflow.log_metric(k, metric_dict[k])
                except:
                    print(f"ERROR: Metric {k} was not logged.")
                    pass

        return (cm_display, roc_auc_display, precision_recall_display, report, statistics)
