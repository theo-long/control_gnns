from torch import nn
from torch import optim
import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Callable
from utils import BasicLogger, Logger, get_device

CHECKPOINT_PATH = "model.pt"


@dataclass
class TrainConfig:
    lr: float
    batch_size: int
    epochs: int
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999


def train(
    dataloader,
    model,
    device,
    optimiser,
    epoch,
    loss_fct,
    metric_fct,
    log_interval,
    logger=None,
):
    """Train model for one epoch."""
    model.train()
    for i, batch in enumerate(dataloader):
        batch.to(device)
        optimiser.zero_grad()
        y_hat = model(batch)
        loss = loss_fct(y_hat, batch.y, reduction="sum")
        metric = metric_fct(y_hat, batch.y)
        loss.backward()
        optimiser.step()
        if (i + 1) % log_interval == 0 and logger is not None:
            logger.log(
                {
                    "epoch": epoch,
                    "batch": i,
                    "train_loss": loss.data,
                    "train_metric": metric.data,
                }
            )
    return loss.data, metric.data


def evaluate(dataloader, model, device, loss_fct, metrics_fct):
    """Evaluate model on dataset."""
    model.eval()
    metrics_eval = 0
    loss_eval = 0
    for batch in dataloader:
        batch.to(device)
        y_hat = model(batch)
        metrics = metrics_fct(y_hat, batch.y)
        loss = loss_fct(y_hat, batch.y)

        metrics_eval += metrics.data * batch.y.shape[0]
        loss_eval += loss.data
    metrics_eval /= len(dataloader.dataset)
    return loss_eval, metrics_eval


def train_eval(
    model: nn.Module,
    training_config: TrainConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    loss_function: Callable,
    metric_function: Callable,
    log_interval: int = 1,
    logger: Logger = BasicLogger(),
    restore_best=True,
):
    """Train the model and return final evaluation stats on test data."""
    device = get_device()

    model.to(device)

    # Instantiate our optimiser
    optimiser = optim.AdamW(
        model.parameters(),
        lr=training_config.lr,
        betas=(training_config.beta1, training_config.beta2),
        weight_decay=training_config.weight_decay,
    )

    # initial evaluation (before training)
    val_loss, val_metric = evaluate(
        val_loader, model, device, loss_function, metric_function
    )
    train_loss, train_metric = evaluate(
        train_loader, model, device, loss_function, metric_function
    )
    epoch_stats = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_metric": train_metric,
        "val_metric": val_metric,
        "epoch": -1,
    }

    logger.log(epoch_stats)

    best_val_loss = torch.inf
    for epoch in range(training_config.epochs):
        train_loss, train_metric = train(
            train_loader,
            model,
            device,
            optimiser,
            epoch,
            loss_function,
            metric_function,
            log_interval,
            logger,
        )
        val_loss, val_metric = evaluate(
            val_loader, model, device, loss_function, metric_function
        )
        epoch_stats = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_metric": train_metric,
            "val_metric": val_metric,
            "epoch": epoch,
        }

        if epoch_stats["val_loss"] <= best_val_loss:
            best_val_loss = epoch_stats["val_loss"]
            if restore_best:
                torch.save(model.state_dict(), CHECKPOINT_PATH)

        logger.log(epoch_stats)

    if restore_best:
        model.load_state_dict(torch.load(CHECKPOINT_PATH))

    test_loss, test_metric = evaluate(
        test_loader, model, device, loss_function, metric_function
    )
    final_stats = {
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_metric": test_metric,
        "epoch": epoch,
    }
    logger.log(final_stats)
    return final_stats
