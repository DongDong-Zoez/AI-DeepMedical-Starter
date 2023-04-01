import wandb

def wandb_settings(key, config, project, entity, name):

    # Wandb Login
    wandb.login(key=key)

    # Initialize W&B
    run = wandb.init(
        config=config,
        project=project,
        entity=entity,
        name=name,
    )