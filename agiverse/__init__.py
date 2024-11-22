import logging

logger = logging.getLogger('agiverse')
logger.setLevel(logging.ERROR)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

from .smart_building import ActionContext, SmartBuilding

__all__ = ['ActionContext', 'SmartBuilding']
