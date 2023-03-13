
import os
import datetime
import time
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import pandas as pd
import numpy as np
from tabulate import tabulate
import wandb

import torch
from torch.optim.swa_utils import AveragedModel, SWALR

import warnings

# Third-party packages
from configs import config, sweep_config
from dataset import preprocessor
from dataloader import get_dataloaders
from model import CXRNet
from scheduler import create_lr_scheduler
from trainer import trainer
from evaluate import evaluate, test
from loss_func import criterion
from metrics import MetricLogger
from augmentations import get_transform
from summarywritter import save_multiple_line_plot, wandb_settings
from utils import calc_multilabel_roc
from logger import show_config

warnings.filterwarnings("ignore")


def main():

    from configs import config
    wandb_settings(config["key"], config, config["project"],
                   config["entity"], config["name"])

    config = wandb.config

    train_df = preprocessor(config.base_root, "train.csv")
    val_df = preprocessor(config.base_root, "val.csv")
    test_df = preprocessor(config.base_root, "test.csv")

    if config.debug:
        train_df = train_df.iloc[:16, :]

    labels = train_df.iloc[:, 2:]
    config.update({"num_pos": torch.tensor(
        labels.sum().values).to(config.device)})
    config.update(
        {"num_neg": train_df.shape[0] - torch.tensor(config.num_pos)})
    # if config.debug:
    #     df = train_df.iloc[:16, :]
    #     config.update({"epochs": 2, "n_splits": None}, allow_val_change=True)
    #     labels = train_df.iloc[:, 2:]

    train_transform = get_transform(
        train=True, img_size=config.img_size, rotate_degree=config.rotate_degree)
    val_transform = get_transform(
        train=False, img_size=config.img_size, rotate_degree=config.rotate_degree)

    res = pd.DataFrame()
    metrics = []
    assert config.monitor in ['acc', 'f1', 'auc']

    train_loader, val_loader, test_loader = get_dataloaders(
        config.batch_size, train_transform, val_transform, train_df, val_df, test_df, config.normalize
    )
    best_metrics = fit(config, train_loader, test_loader, test_loader, None)
    metrics.append(best_metrics)
    print(res)


def fit(config, train_loader, val_loader, test_loader, fold):

    num_train_images = train_loader.__len__() * config.batch_size
    num_val_images = val_loader.__len__()

    if test_loader is None:
        test_loader = val_loader

    num_workers = min(
        [os.cpu_count(), config.batch_size if config.batch_size > 1 else 0, 8])

    num_steps_per_epoch = len(train_loader)

    model = CXRNet(config.model_name, config.classes)
    model.to(config.device)
    wandb.watch(model, log="all")

    params_to_optimize = [
        {"params": [p for p in model.parameters() if p.requires_grad]},
        #model.parameters(),
    ]


    named_parameter = [
        name for name, param in model.named_parameters() if param.requires_grad
    ]

    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=config.lr, weight_decay=config.weight_decay
        )
    elif config.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr, weight_decay=config.weight_decay
        )
    elif config.optimizer == "adadelta":
        optimizer = torch.optim.Adadelta(
            params_to_optimize
        )
    scaler = torch.cuda.amp.GradScaler() if config.amp else None

    assert config.lr_scheduler in [
        'warmup', 'poly', 'reduce', "consineAnneling"]

    lr_scheduler = create_lr_scheduler(config.lr_scheduler, optimizer, len(
        train_loader), config.epochs, config.T_max, config.patience, config.factor)

    if config.resume:
        checkpoint = torch.load(config.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if config.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    titles = {
        'device': config.device, 'batch size': config.batch_size, 'num workers': num_workers, 'learning rate': config.lr, 'loss function': config.loss_func,
        'params to optimize': named_parameter, 'optimizer': optimizer, 'scaler': scaler, 'epochs': config.epochs, 'num_classes': config.classes,
        'weight decay': config.weight_decay, 'lr_scheduler': config.lr_scheduler, 'num train images': num_train_images,
        'num val images': num_val_images, 'num steps per epoch': num_steps_per_epoch,
        # 'train transfrom': train_transforms.trans, 'val transforms': val_transforms.trans,
    }
    titles.pop('params to optimize', None)

    titles = show_config(titles)
    titles.log_every()

    print(titles)

    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=config.lr)

    best_metrics = -1
    logger = MetricLogger()
    loss_func = criterion(config.loss_func, torch.tensor(config.num_pos).to(
        config.device), torch.tensor(config.num_neg).to(config.device))

    start_time = time.time()
    for epoch in range(config.epochs):

        train_loss, lr = trainer(model, optimizer, train_loader, config.device,
                                 lr_scheduler, loss_func, swa_model, swa_scheduler)

        print(
            f"FOLD [{fold}] epochs: {epoch+1}/{config.epochs} | training   loss: {train_loss} lr: {lr}")

        history, val_loss = evaluate(model, val_loader, loss_func, config.device,
                                     config.threshold, config.tta, config.classes_name, config.average)

        logger.log_every(history, fold)
        wandb.log({f"Fold {fold} | Training   Loss": train_loss})
        wandb.log({f"Fold {fold} | Epoch": epoch})
        wandb.log({f"Fold {fold} | Validation Loss": val_loss})
        for cls, row in history.iterrows():
            for col, val in zip(history.columns, row):
                cls_name = f"Fold:{fold} | {cls} {col}"
                wandb.log({cls_name: val})

        print(
            f"FOLD [{fold}] epochs: {epoch+1}/{config.epochs} | validation loss: {val_loss}")

        print(tabulate(history, headers='keys', tablefmt='psql'))

        save_file = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch
        }

        if config.amp:
            save_file["scaler"] = scaler.state_dict()

        if config.monitor == 'acc':
            global_acc = history.accuracy_score.iloc[-1]
            if global_acc > best_metrics:
                best_metrics = global_acc
                torch.save(save_file, f"model{fold}.pth")
                wandb.save(f"model{fold}.pth")
        elif config.monitor == 'f1':
            f1 = history.f1.iloc[-1]
            if f1 > best_metrics:
                best_metrics = f1
                torch.save(save_file, f"model{fold}.pth")
                wandb.save(f"model{fold}.pth")
        elif config.monitor == 'auc':
            auc = history.auc.iloc[-1]
            if auc > best_metrics:
                best_metrics = auc
                torch.save(save_file, f"model{fold}.pth")
                wandb.save(f"model{fold}.pth")

    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

    for epoch in range(1):

        train_loss, lr = trainer(model, optimizer, train_loader, config.device,
                                 lr_scheduler, loss_func, swa_model, swa_scheduler)

        print(
            f"FOLD [{fold}] epochs: {epoch+1}/{config.epochs} | training   loss: {train_loss} lr: {lr}")

        history, val_loss = evaluate(model, val_loader, loss_func, config.device,
                                     config.threshold, config.tta, config.classes_name, config.average)

        logger.log_every(history, fold)
        wandb.log({f"Fold {fold} | Training   Loss": train_loss})
        wandb.log({f"Fold {fold} | Epoch": epoch})
        wandb.log({f"Fold {fold} | Validation Loss": val_loss})
        for cls, row in history.iterrows():
            for col, val in zip(history.columns, row):
                cls_name = f"Fold:{fold} | {cls} {col}"
                wandb.log({cls_name: val})

        print(
            f"FOLD [{fold}] epochs: {epoch+1}/{config.epochs} | validation loss: {val_loss}")

        print(tabulate(history, headers='keys', tablefmt='psql'))

        save_file = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch
        }

        if config.amp:
            save_file["scaler"] = scaler.state_dict()

        if config.monitor == 'acc':
            global_acc = history.accuracy_score.iloc[-1]
            if global_acc > best_metrics:
                best_metrics = global_acc
                torch.save(save_file, f"model{fold}.pth")
                wandb.save(f"model{fold}.pth")
        elif config.monitor == 'f1':
            f1 = history.f1.iloc[-1]
            if f1 > best_metrics:
                best_metrics = f1
                torch.save(save_file, f"model{fold}.pth")
                wandb.save(f"model{fold}.pth")
        elif config.monitor == 'auc':
            auc = history.auc.iloc[-1]
            if auc > best_metrics:
                best_metrics = auc
                torch.save(save_file, f"model{fold}.pth")
                wandb.save(f"model{fold}.pth")

    logger.wandb_log()
    logger.to_list()
    save_cls_name = config.classes_name + ["Summary"]
    xs = [e for e in range(config.epochs)]
    save_multiple_line_plot(
        f"FOLD [{fold}] Accuracy", xs, logger.acc, save_cls_name, "Accuracy")
    save_multiple_line_plot(
        f"FOLD [{fold}] AUROC", xs, logger.auc, save_cls_name, "Area Under Curve")
    save_multiple_line_plot(
        f"FOLD [{fold}] F1 Score", xs, logger.f1, save_cls_name, "F1 Score")
    save_multiple_line_plot(
        f"FOLD [{fold}] Precision", xs, logger.precision, save_cls_name, "Precision")
    save_multiple_line_plot(
        f"FOLD [{fold}] Recall", xs, logger.recall, save_cls_name, "Recall")

    print(f"Fold {fold} | best {config.monitor}: {best_metrics}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))

    start_time = time.time()
    print("*" * 5 + " Start testing model performance " + "*" * 5)

    checkpoint = torch.load(f"model{fold}.pth")
    history, preds, targets = test(model, test_loader, config.device,
                                   config.threshold, config.tta, config.classes_name, config.average)
    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    cm = wandb.plot.confusion_matrix(
        y_true=targets,
        preds=preds,
        class_names=config.classes_name,
        title=f"Fold {fold} | Test CM"
    )
    wandb.log({f"Fold[{fold}] Test CM": cm})

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("testing time {}".format(total_time_str))


if os.environ.get("WANDB_RUN_ID"):
    del os.environ['WANDB_RUN_ID']

if __name__ == '__main__':
    if config["sweep"]:
        sweep_id = wandb.sweep(sweep_config, project=config["project"])
        wandb.agent(sweep_id, function=main, count=config["count"])
    else:
        main()
