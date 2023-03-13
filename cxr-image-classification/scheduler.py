import torch

def warmup_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=5,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):

        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)

            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

def poly_lr_scheduler(optimizer, num_step, epochs):
    def poly(x):
        return (1 - (x) / (num_step * epochs)) ** 0.9
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly)

def cosine_annealing_lr_scheduler(optimizer, T_max):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=T_max)

def create_lr_scheduler(scheduler_name, optimizer, steps, epochs, T_max=None, patience=2, factor=0.5):
    if scheduler_name == 'warmup':
        lr_scheduler = warmup_lr_scheduler(optimizer, steps, epochs, warmup=True)
    elif scheduler_name == 'poly':
        lr_scheduler = poly_lr_scheduler(optimizer, steps, epochs)
    elif scheduler_name == "consineAnneling":
        lr_scheduler = cosine_annealing_lr_scheduler(optimizer, T_max)
    else:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor, min_lr=1e-6)

    return lr_scheduler