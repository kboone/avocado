from .settings import settings
from .utils import *

from .augment import *
from .astronomical_object import *
from .classifier import *
from .dataset import *
from .features import *
from .instruments import *

from . import plasticc

# Expose the load method of Dataset
load = Dataset.load

# Expose the load method of Classifier
load_classifier = Classifier.load


__all__ = ["Dataset", "AstronomicalObject"]
