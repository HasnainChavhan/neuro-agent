"""
NeuroAgent — Structured Logging Utility
JSON-formatted logs compatible with CloudWatch, Datadog, and Grafana Loki.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

from pythonjsonlogger import jsonlogger


class NeuroAgentLogger:
    """Factory that creates named JSON loggers with context binding."""

    _instances: dict[str, logging.Logger] = {}

    @classmethod
    def get(cls, name: str, level: str = "INFO") -> logging.Logger:
        if name in cls._instances:
            return cls._instances[name]

        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))

        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = jsonlogger.JsonFormatter(
                fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = False

        cls._instances[name] = logger
        return logger


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    return NeuroAgentLogger.get(name, level)


class BoundLogger:
    """Logger with pre-bound context fields (e.g., session_id, query_id)."""

    def __init__(self, base: logging.Logger, **context: Any):
        self._logger = base
        self._context = context

    def _merge(self, kwargs: dict) -> dict:
        return {**self._context, **kwargs}

    def info(self, msg: str, **kwargs: Any) -> None:
        self._logger.info(msg, extra=self._merge(kwargs))

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._logger.warning(msg, extra=self._merge(kwargs))

    def error(self, msg: str, **kwargs: Any) -> None:
        self._logger.error(msg, extra=self._merge(kwargs))

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._logger.debug(msg, extra=self._merge(kwargs))

    def bind(self, **extra: Any) -> "BoundLogger":
        return BoundLogger(self._logger, **{**self._context, **extra})
