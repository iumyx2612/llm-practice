from dotenv import load_dotenv

from .settings import (
    Settings as AppSettings,
    load_settings
)
from .tracing import setup_tracing
from .modules import setup_modules


__all__ = [
    "AppSettings",
    "load_settings",
    "setup_tracing",
    "setup_modules",
    "initialize"
]


def initialize(dotenv_path: str):
    load_dotenv(dotenv_path)
    settings = load_settings()

    setup_tracing(settings)
    setup_modules(settings)
