import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        f'%(asctime)s - {__name__} - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

from .agent import Agent
from .toolkit import ActionContext, Toolkit
from .tools import Tools

__all__ = ['Agent', 'ActionContext', 'Toolkit', 'Tools']
