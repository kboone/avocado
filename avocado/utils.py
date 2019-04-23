import logging

from . import settings

# Logging
logger = logging.getLogger('avocado')

# Exceptions
class AvocadoException(Exception):
    pass
