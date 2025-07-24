import logging
from typing import Optional

from torch.utils.tensorboard import SummaryWriter


def log_epoch_metrics(
    epoch: int,
    avg_loss: float,
    correct: int,
    total: int,
    writer: Optional[SummaryWriter] = None,
) -> None:
    """Log average loss and global accuracy for an epoch.

    Parameters
    ----------
    epoch: int
        Current epoch number.
    avg_loss: float
        Mean training loss for the epoch.
    correct: int
        Number of correctly predicted samples.
    total: int
        Total number of samples.
    writer: Optional[SummaryWriter]
        TensorBoard writer to log metrics, if any.
    """
    acc = correct / total if total else 0.0
    logging.info("Epoch %d Average Loss: %.4f", epoch, avg_loss)
    logging.info("Epoch %d Global Accuracy: %.3f", epoch, acc)
    if writer is not None:
        writer.add_scalar("AverageLoss", avg_loss, epoch)
        writer.add_scalar("GlobalAccuracy", acc, epoch)


