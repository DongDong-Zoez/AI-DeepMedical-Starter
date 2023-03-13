import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve

def calc_multilabel_roc(preds, targets, classes):
    fpr = dict()
    tpr = dict()
    
    # Compute ROC curve and ROC area for each class
    for i in range(targets.shape[1]):

        class_label = classes[i]

        fpr[class_label], tpr[class_label], _ = roc_curve(
            targets[:, i], preds[..., i], pos_label=i
        )

    df = pd.DataFrame(
        {
            "class": np.hstack([[k] * len(v) for k, v in fpr.items()]),
            "fpr": np.hstack(list(fpr.values())),
            "tpr": np.hstack(list(tpr.values())),
        }
    ).sort_values(["fpr", "tpr", "class"])

    df.fillna(0, inplace=True)
    df = df.round(3)
    
    return df