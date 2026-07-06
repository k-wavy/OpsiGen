import argparse
import json
import os
import pickle
import random
import time

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, WeightedRandomSampler

import models
from pdb_dataset import PDBDataset, PDBDatasetConfig

try:
    import wandb
except ImportError:  # W&B is optional for local reproducible runs.
    wandb = None


def calculate_l1_reg(model, alpha=0.0001):
    return alpha * sum(torch.norm(p, 1) for p in model.parameters())


def get_pickle_name(base_name, test_error, start_time, epoch):
    root, ext = os.path.splitext(base_name)
    if not ext:
        ext = ".pkl"
    return f"{root}_best_epoch-{epoch:04d}_mae-{test_error:.3f}_{start_time}{ext}"


def calculate_energy_loss(pred, gt):
    ev = 1239.8

    if pred < 3 or gt < 3:
        return torch.tensor(7.0, device=pred.device)

    return (ev / pred) - (ev / gt)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train an OpsiGen/RhoMax graph model.")
    parser.add_argument("config_path", help="Path to a JSON training configuration.")
    return parser.parse_args()


def set_config(config_file):
    with open(config_file, "r") as f:
        config = json.load(f)

    print(json.dumps(config, indent=2))
    return config


def generate_dataset_config(config):
    dataset_config = PDBDatasetConfig()
    dataset_config.excel_path = config["excel_path"]
    dataset_config.graph_dists_path = config["graph_dists_path"]
    dataset_config.graph_features_path = config["graph_features_path"]
    dataset_config.indexes = config["indexes_to_keep"]
    dataset_config.means_path = config.get("means_path", "means.npy")
    dataset_config.stds_path = config.get("stds_path", "stds.npy")

    return dataset_config


def set_reproducible_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def maybe_init_wandb(config):
    if not config.get("wandb", False):
        return False

    if wandb is None:
        raise ImportError("wandb is enabled in the config, but the wandb package is not installed.")

    wandb.init(
        project=config.get("wandb_project", "opsigen"),
        name=config.get("wandb_run_name"),
        config=config,
    )
    return True


def log_metrics(metrics, use_wandb):
    print(
        "epoch={epoch} train_mae={train_loss:.4f} test_mae={test_loss:.4f} "
        "energy_loss={energy_loss:.4f}".format(**metrics)
    )
    if use_wandb:
        wandb.log(metrics)


def build_model(config, device):
    if config.get("load_checkpoint"):
        with open(config["load_checkpoint"], "rb") as f:
            return pickle.load(f).to(device)

    model_factory = getattr(models, config["model_name"])
    return model_factory(
        config["number_features"],
        config["hidden_layer_size"],
        config["out_layer_size"],
        device,
        dp_rate=config.get("dropout", 0.1),
    ).to(device)


def save_checkpoint(model, config, test_loss, start_time, epoch):
    checkpoint_dir = config.get("checkpoint_dir", ".")
    os.makedirs(checkpoint_dir, exist_ok=True)
    base_name = os.path.basename(config.get("pickle_file", "opsigen_model.pkl"))
    checkpoint_path = os.path.join(checkpoint_dir, get_pickle_name(base_name, test_loss, start_time, epoch))

    with open(checkpoint_path, "wb") as f:
        pickle.dump(model, f)

    return checkpoint_path


def run_epoch(model, dataloader, optimizer, config, device, train):
    if train:
        model.train()
    else:
        model.eval()

    loss_sum = 0.0
    energy_loss_sum = 0.0
    usable_graphs = 0

    for graph in dataloader:
        if graph[2] == 0:
            continue

        usable_graphs += 1

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            result = model.double().forward(graph[0], graph[1], config["graph_th"])
            target = graph[2].to(device)
            loss = torch.norm(result - target)
            if train:
                loss = loss + calculate_l1_reg(model, config.get("l1_alpha", 0.0001))
                loss.backward()
                optimizer.step()

        loss_sum += float(torch.norm(result - target).item())
        if not train:
            energy_loss_sum += float(calculate_energy_loss(result, target).item())

    if usable_graphs == 0:
        raise RuntimeError("No usable graphs were loaded for this epoch. Check graph paths and split files.")

    return loss_sum / usable_graphs, energy_loss_sum / usable_graphs


def main():
    args = parse_arguments()
    config = set_config(args.config_path)

    seed = int(config.get("seed", 123))
    set_reproducible_seed(seed)

    use_wandb = maybe_init_wandb(config)
    start_time = time.strftime("%Y%m%d-%H%M%S")

    print(torch.version.cuda)
    print(torch.__version__)
    print("cuda_available", torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0.00001),
    )

    dataset_config = generate_dataset_config(config)
    train_dataset = PDBDataset(
        dataset_config,
        config["train_wildtypes_list"],
        normalize_last=config["dataset_normalize_last"],
    )
    test_dataset = PDBDataset(
        dataset_config,
        config["test_wildtypes_list"],
        means=train_dataset.means,
        stds=train_dataset.stds,
        normalize_last=config["dataset_normalize_last"],
    )

    sampler_num_samples = int(config.get("num_samples", len(train_dataset)))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        sampler=WeightedRandomSampler(train_dataset.calculate_weights(), num_samples=sampler_num_samples),
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    epochs = int(config.get("epochs", 1000))
    best_test_loss = float("inf")
    best_checkpoint = None

    print("starting training loop")
    for epoch in range(1, epochs + 1):
        train_loss, _ = run_epoch(model, train_dataloader, optimizer, config, device, train=True)
        test_loss, energy_loss = run_epoch(model, test_dataloader, optimizer, config, device, train=False)

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "energy_loss": energy_loss,
        }
        log_metrics(metrics, use_wandb)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_checkpoint = save_checkpoint(model, config, test_loss, start_time, epoch)
            print(f"saved best checkpoint: {best_checkpoint}")

        if test_loss < config.get("test_goal", -float("inf")):
            print(f"test goal reached at epoch {epoch}: {test_loss:.4f}")

    print(f"training complete; best_test_loss={best_test_loss:.4f}; best_checkpoint={best_checkpoint}")


if __name__ == "__main__":
    main()
