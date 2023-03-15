import copy

from torch import nn
from torch import optim
import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Callable, Optional
from utils import BasicLogger, Logger, get_device


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
    mask=None,
):
    """Train model for one epoch."""
    model.train()
    metrics_train = 0
    loss_train = 0
    num_examples = 0
    for i, batch in enumerate(dataloader):
        batch.to(device)
        optimiser.zero_grad()
        y_hat = model(batch)
        if mask is not None:
            loss = loss_fct(y_hat[mask], batch.y[mask], reduction="mean")
            metrics = metric_fct(y_hat[mask], batch.y[mask])
        else:
            loss = loss_fct(y_hat, batch.y, reduction="mean")
            metrics = metric_fct(y_hat, batch.y)
        loss.backward()
        optimiser.step()

        # Reweight metrics by number of examples in the batch
        metrics_train += metrics.data * batch.y.shape[0]
        loss_train += loss.data * batch.y.shape[0]
        num_examples += batch.y.shape[0]

    # Divide by total number of examples in whole dataset
    loss_train /= num_examples
    metrics_train /= num_examples

    return loss_train, metrics_train


def evaluate(dataloader, model, device, loss_fct, metrics_fct, mask=None):
    """Evaluate model on dataset."""
    model.eval()
    metrics_eval = 0
    loss_eval = 0
    num_examples = 0
    for batch in dataloader:
        batch.to(device)
        y_hat = model(batch)
        if mask is not None:
            metrics = metrics_fct(y_hat[mask], batch.y[mask])
            loss = loss_fct(y_hat[mask], batch.y[mask], reduction="sum")
        else:
            metrics = metrics_fct(y_hat, batch.y)
            loss = loss_fct(y_hat, batch.y, reduction="sum")

        # Reweight metrics by number of examples in the batch
        metrics_eval += metrics.data * batch.y.shape[0]
        loss_eval += loss.data
        num_examples += batch.y.shape[0]

    # Divide by total number of examples in whole dataset
    loss_eval /= num_examples
    metrics_eval /= num_examples

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
    train_mask: Optional[torch.Tensor] = None,
    val_mask: Optional[torch.Tensor] = None,
    test_mask: Optional[torch.Tensor] = None,
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
        val_loader, model, device, loss_function, metric_function, val_mask
    )
    train_loss, train_metric = evaluate(
        train_loader, model, device, loss_function, metric_function, test_mask
    )
    epoch_stats = {
        "train_loss": train_loss.item(),
        "val_loss": val_loss.item(),
        "train_metric": train_metric.item(),
        "val_metric": val_metric.item(),
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
            train_mask,
        )
        val_loss, val_metric = evaluate(
            val_loader,
            model,
            device,
            loss_function,
            metric_function,
            val_mask,
        )
        epoch_stats = {
            "train_loss": train_loss.item(),
            "val_loss": val_loss.item(),
            "train_metric": train_metric.item(),
            "val_metric": val_metric.item(),
            "epoch": epoch,
        }

        if epoch_stats["val_loss"] <= best_val_loss:
            best_val_loss = epoch_stats["val_loss"]
            if restore_best:
                best_model = copy.deepcopy(model)

        logger.log(epoch_stats)

    if restore_best:
        model = best_model

    test_loss, test_metric = evaluate(
        test_loader, model, device, loss_function, metric_function, test_mask
    )
    final_stats = {
        "best_val_loss": best_val_loss,
        "test_loss": test_loss.item(),
        "test_metric": test_metric.item(),
    }
    return final_stats
