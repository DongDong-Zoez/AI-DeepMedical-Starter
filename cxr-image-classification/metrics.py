from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
    accuracy_score,
)
import pandas as pd
import wandb

# Third-party packages

def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes] true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes] can either be probability estimates of the positive class,
          confidence values, or binary decisions.
    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    
    gt_np = gt
    pred_np = pred
    for i in range(gt_np.shape[1]):
        try:
            AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
        except:
            AUROCs.append(0)
    return AUROCs


def calculate_metrics(pred, target, cls_name, threshold=0.5, average=None):

    preds = (pred > threshold).to("cpu").float().numpy()
    targets = target.to("cpu").float().numpy()

    recall = recall_score(y_true=targets, y_pred=preds, average=average)
    precision = precision_score(y_true=targets, y_pred=preds, average=average)
    f1 = f1_score(y_true=targets, y_pred=preds, average=average)
    acc = accuracy_score(y_true=targets, y_pred=preds)
    auc = roc_auc_score(targets, preds)

    history = {}
    history["recall"] = recall
    history["precision"] = precision
    history["f1"] = f1
    history["auc"] = auc
    history["accuracy_score"] = acc
    history = pd.DataFrame(history)
    overall = history.mean(axis=0)
    history.loc[-1] = overall
    history.index = cls_name +  ["Summary"]

    return history

class MetricLogger:

    def __init__(self):
        self.recall = pd.DataFrame()
        self.precision = pd.DataFrame()
        self.f1 = pd.DataFrame()
        self.auc = pd.DataFrame()
        self.acc = pd.DataFrame()

    def cat(self, df_list):
        return pd.concat(df_list, axis=1)

    def log_every(self, history, fold):
        self.recall = self.cat([self.recall, history.recall])
        self.precision = self.cat([self.precision, history.recall])
        self.f1 = self.cat([self.f1, history.recall])
        self.auc = self.cat([self.auc, history.recall])
        self.acc = self.cat([self.acc, history.recall])

    def to_list(self):
        self.recall = self.recall.values.tolist()
        self.precision = self.precision.values.tolist()
        self.f1 = self.f1.values.tolist()
        self.auc = self.auc.values.tolist()
        self.acc = self.acc.values.tolist()

    def wandb_log(self):

        cls_name = self.recall.index.values.tolist()

        self.recall.columns = [i for i in range(self.recall.shape[1])]
        self.precision.columns = [i for i in range(self.precision.shape[1])]
        self.f1.columns = [i for i in range(self.f1.shape[1])]
        self.auc.columns = [i for i in range(self.auc.shape[1])]
        self.acc.columns = [i for i in range(self.acc.shape[1])]

        recall = wandb.Table(dataframe=self.recall, columns=[i for i in range(self.recall.shape[0])], rows=cls_name ,allow_mixed_types=True)
        precision = wandb.Table(dataframe=self.precision, columns=[i for i in range(self.precision.shape[0])], rows=cls_name, allow_mixed_types=True)
        f1 = wandb.Table(dataframe=self.f1, columns=[i for i in range(self.f1.shape[0])], rows=cls_name, allow_mixed_types=True)
        auc = wandb.Table(dataframe=self.auc, columns=[i for i in range(self.auc.shape[0])], rows=cls_name, allow_mixed_types=True)
        acc = wandb.Table(dataframe=self.acc, columns=[i for i in range(self.acc.shape[0])], rows=cls_name, allow_mixed_types=True)

        wandb.log({"Recall": recall})
        wandb.log({"Precision": precision})
        wandb.log({"F1 Score": f1})
        wandb.log({"AUROC": auc})
        wandb.log({"Accuracy Score": acc})