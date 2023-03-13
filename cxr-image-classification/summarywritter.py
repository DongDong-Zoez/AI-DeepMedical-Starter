import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import wandb


def wandb_settings(key, config, project, entity, name):

    # Wandb Login
    wandb.login(key=key, relogin=True)

    # Initialize W&B
    run = wandb.init(
        config=config,
        project=project,
        entity=entity,
        name=name,
        reinit=True
    )

def save_table(df, base_path, table_name):
    wandb.log({"CXR dataframe": df})
    table = wandb.Table(
        columns=['Id', 'Category', 'Image'], allow_mixed_types=True)

    for _, row in tqdm(df.iterrows()):
        id = row.ID
        filename = row.Filename
        cat = row.category_eng
        path = os.path.join(base_path, cat, filename)
        img = plt.imread(path)

        table.add_data(
            id,
            cat,
            wandb.Image(img),
        )

    wandb.log({table_name: table})


def save_multiple_line_plot(tname, xs, ys, keys, title, xname="epochs"):
    wandb.log({f"{tname}": wandb.plot.line_series(
        xs=xs,
        ys=ys,
        keys=keys,
        title=title,
        xname=xname)})
