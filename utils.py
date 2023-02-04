from abc import ABC, abstractmethod
import logging

logger = logging.getLogger()


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
