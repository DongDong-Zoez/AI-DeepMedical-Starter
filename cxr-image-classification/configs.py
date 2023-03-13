config = {
    'base_root': r"C:\Users\zoez5\cxr\chest_xray",
    'csv_name': "chest_dongdong.csv",
    'debug': False,
    'resume': "",
    'classes': 2,
    'img_size': 224,
    'normalize': False,
    'classes_name': ['NORMAL', 'PNEUMONIA'],
    'device': "cuda",
    'epochs': 5,
    'n_splits': 5,
    "batch_size": 8,
    'optimizer': "adam",
    'model_name': "tf_efficientnet_b4_ns",
    'lr': 1e-4,
    'weight_decay': 0,
    'amp': False,
    'lr_scheduler': "poly",
    'T_max': 10,
    'patience': 2,
    'factor': 0.5,
    'random_state': 42,
    'threshold': 0.5,
    'tta': False,
    'monitor': "f1",
    'loss_func': "WeightedBCE",
    'alpha': 0.5,
    'beta': 0.5,
    'gamma': 2,
    'k': 3,
    'average': None,
    'key': "Your API key here",
    'project': "CXR-Multi-label-Binary-Classification",
    'entity': "DDCVLAB",
    'name': "Testing",
    "sweep": False,
    "count": 5,
    "rotate_degree": 10,
}

sweep_config = {
    "program": "main.py",
    "method": "grid",
    "metrics": {'goal': 'minimize', 'name': 'val_loss'},
    "parameters": {
        "lr": {
            "values": [1e-3, 5e-4, 1e-4, 5e-5]
        },
        "weight_decay": {
            "values": [1e-6, 1e-7, 1e-8]
        },
        "threshold": {
            "values": [0.4, 0.5, 0.6]
        }
    }
}