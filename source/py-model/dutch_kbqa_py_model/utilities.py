"""Various utility symbols."""

from pathlib import Path, PurePosixPath
from requests.exceptions import ConnectionError
from huggingface_hub.hf_api import HfApi, ModelInfo
from enum import Enum
from typing import Optional, List


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
