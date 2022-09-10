"""Various utility symbols."""

import random
import numpy as np
import logging
import re
import os
import torch
import torch.backends.cudnn as cudnn
from pathlib import PurePosixPath
from requests.exceptions import ConnectionError
from huggingface_hub.hf_api import HfApi, ModelInfo
from enum import Enum
from typing import Optional, List, Tuple, Union, NamedTuple
from typing_extensions import Literal


# Debugging.
try:
    if os.environ['DEBUG_MODE'] in ('True', 'true', 'T', 't'):
        DEBUG_MODE = True
    elif os.environ['DEBUG_MODE'] in ('False', 'false', 'F', 'f'):
        DEBUG_MODE = False
    else:
        raise ValueError('Environment variable `DEBUG_MODE`, if set, should ' +
                         'have strictly one of the following values: ' +
                         '\'True\', \'true\', \'T\', \'t\' (for enabling ' +
                         'debug mode), \'False\', \'false\', \'F\', or ' +
                         '\'f\' (for disabling debug mode). If not set, ' +
                         'debug mode is disabled.')
except KeyError:
    DEBUG_MODE = False


# Logging-related global constants.
LOGGER_FMT = '[%(asctime)s, %(levelname)8s]    %(message)s'
LOGGER_DATE_FMT = '%Y-%m-%dT%H:%M:%S (%Z)'  # ISO 8601-compliant
LOGGER_NUMBER_EXAMPLES = 5
logging.basicConfig(format=LOGGER_FMT,
                    datefmt=LOGGER_DATE_FMT,
                    level=logging.DEBUG if DEBUG_MODE else logging.INFO)
LOGGER = logging.getLogger(name='Transformer Runner')


# Now that we've set up both `DEBUG_MODE` and `LOGGER`, warn the user if
# debugging is enabled, just in case.
if DEBUG_MODE:
    LOGGER.warning('Debugging mode is turned ON!')


# A device that PyTorch may target.
TorchDevice = Union[Literal['cpu'], Literal['cuda']]
# A special 'rank' to use when you wish to not use distributed training.
NO_DISTRIBUTION_RANK = -1


class NaturalLanguage(Enum):
    """A natural language.
    
    Enumeration member values are ISO 639-1 language codes.
    """
    ENGLISH = 'en'
    DUTCH = 'nl'


class QueryLanguage(Enum):
    """A querying language of any kind.
    
    Enumeration member values are file extension-like.
    """
    SPARQL = 'sparql'


def hugging_face_hub_model_exists(author: Optional[str], model: str) -> bool:
    """Determines whether the model with author `author` and name `model`
    on HuggingFace Hub.

    :param author: The author of the model. May be omitted if a root-level
        model is searched for, such as `'bert-base-uncased'`.
    :param model: The model's name.
    :returns: Whether the model exists on HuggingFace Hub.
    :throws: `RuntimeError` when connection problems arise, or `ValueError` if
        either `author` or `model` are empty strings.
    """
    id_or_path: str = model if author is None else f'{author}/{model}'
    if author is not None and author == '' or \
       model == '':
        raise ValueError(f'\'{id_or_path}\' is not a valid HuggingFace Hub ' +
                         ' model ID!')
    api = HfApi()
    try:
        infos: List[ModelInfo] = api.list_models(author=author,
                                                 search=model)
    except ConnectionError as connect_error:
        raise RuntimeError('Unable to check on HuggingFace Hub whether ' +
                           f'model \'{id_or_path}\' exists.\nReason: ' +
                           f'\"{connect_error}\".')
    for info in infos:
        if info.modelId == id_or_path:
            # Exact match: The model exists on HuggingFace Hub.
            return True
    # No match: The model does not exist (precisely, as-is) on HuggingFace Hub.
    return False


def string_is_existing_hugging_face_hub_model(string: str) -> bool:
    """Determines whether the supplied string represents an existing model ID
    on HuggingFace Hub.
    
    :param string: The string to test.
    :returns: The question's answer.
    :throws: `RuntimeError` when connection problems arise.
    """
    path = PurePosixPath(string)
    author: Optional[str]
    model: str
    if len(path.parts) == 1:
        author = None
        model = path.parts[0]
    elif len(path.parts) == 2:
        author = path.parts[0]
        model = path.parts[1]
    else:
        return False
    try:
        return hugging_face_hub_model_exists(author, model)
    except ValueError:
        # Obtained a problematic string, such as `'/bert-based-uncased'`.
        return False


class SemanticVersion(NamedTuple):
    """A semantic versioning-compliant software version.
    
    This function omits pre-release and build information.
    """
    major: int
    minor: int
    patch: int


def pytorch_version() -> SemanticVersion:
    """Returns the PyTorch version in semantic versioning format.

    :returns: The version.
    """
    match = re.search('^(([1-9]*[0-9])(\.)){2}([1-9]*[0-9])',
                      torch.__version__).group().split('.')
    tup = tuple(int(v) for v in match)
    assert(len(tup) == 3)
    major, minor, patch = tup
    return SemanticVersion(major, minor, patch)


# Permitted values for initialising pseudo-random number generators with.
LEGAL_SEEDS_RANGE: Tuple[int, int] = (1, 2 ** 32 - 1)


def set_seeds(seed: int) -> None:
    """Initialises pseudo-random number generators (PRNGs) across the program's
    components that rely on them.
    
    Note that "[c]completely reproducible results are not guaranteed across
    PyTorch releases, individual commits or different platforms. Furthermore,
    results need not be reproducible between CPU and GPU executions, even when
    using identical seeds." (PyTorch documentation, section 'Reproducibility,
    version 1.6.0). Thus, this method cannot generally guarantee determinism
    across systems; it can only help in debugging problems that persist on
    singular systems.

    :param seed: The seed to use.
    :throws: `AssertionError` if `seed` does not lie within the range of legal
        seeds.
    """
    version = pytorch_version()
    try:
        assert(seed in range(*LEGAL_SEEDS_RANGE))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if version.minor >= 8:
            torch.use_deterministic_algorithms(mode=True)
        else:
            cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        cudnn.benchmark = False
    except AssertionError:
        print('Your PRNG seed, %d, does not lie within [%d, %d]!' %
              (seed,
               LEGAL_SEEDS_RANGE[0],
               LEGAL_SEEDS_RANGE[1]))


class MLStage(Enum):
    """A stage in developing a machine learning model: 'training',
    'validating', or 'testing'.
    """
    TRAIN = 'train'
    VALIDATE = 'validate'
    TEST = 'test'
