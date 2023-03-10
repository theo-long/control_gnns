from abc import ABC, abstractmethod
import logging

import torch
import numpy as np

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


class Logger(ABC):
    """Abstract Logger Class"""

    @abstractmethod
    def log(message):
        pass


class BasicLogger(Logger):
    """Basic Command Line Logger"""

    def __init__(self) -> None:
        self.logger = logger

    def log(self, message):
        if isinstance(message, dict):
            message = ", ".join([f"{k}:{v}" for k, v in message.items()])
        self.logger.info(message)


class PrintLogger(Logger):
    """Logger that just uses print method"""

    def log(self, message):
        if isinstance(message, dict):
            message = ", ".join([f"{k}:{v}" for k, v in message.items()])
        print(message)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


CALLABLE_DICT = {
    "log": np.log,
    "sqrt": np.sqrt,
}


def parse_callable_string(callable_str: str):
    if callable_str.isdigit():
        return lambda n: int(callable_str)
    else:
        try:
            val = float(callable_str)
            return lambda n: val * n
        except ValueError:
            pass

        try:
            callable = CALLABLE_DICT[callable_str]
            return callable
        except KeyError:
            raise ValueError(
                f"Invalid callable string {callable_str} passed to parser."
            )
