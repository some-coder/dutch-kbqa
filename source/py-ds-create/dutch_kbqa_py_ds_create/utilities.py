"""Various utility symbols."""

import os
import json
import platform
from pathlib import Path
from enum import Enum
from typing import Union, Literal, Dict, Any, Optional, List


# Absolute file system path to the package's root (base) directory.
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


class NaturalLanguage(Enum):
    """A natural language.
    
    Enumeration member values are ISO 639-1 language codes.
    """
    ENGLISH = 'en'
    DUTCH = 'nl'


def overwritably_print(text: str) -> None:
    """Prints to standard output, allowing future output to overwrite it.
    
    :param text: The text to print.
    """
    if platform.system() == 'Darwin':
        print(f'\r{text}', flush=True, end='')
    else:
        # Disable fancy overwriting of lines; it's buggy.
        print(text, flush=True)


FileNotFoundReaction = Union[Literal['throw-error'],
                             Literal['return-none']]


def json_loaded_from_disk(location: Path,
                          upon_file_not_found: FileNotFoundReaction) -> \
        Optional[Dict[str, Any]]:
    """Returns the contents of `location`, interpreted as JSON.

    :param location: An absolute or relative file system location.
    :param upon_file_not_found: The action to perform when the file can't be
        found. `'throw-error'` throws an exception; `'return-none'` simply
        returns `None`.
    :returns: The JSON data, or `None` if the file couldn't be found and
        `upon_file_not_found` was set to `'return-none'`.
    :throws: `RuntimeError` when anything abnormal happens.
    """
    try:
        with open(location, 'r') as handle:
            contents = json.load(handle)
            assert(type(contents) == dict)
            return contents
    except FileNotFoundError:
        if upon_file_not_found == 'throw-error':
            raise RuntimeError(f'File \'{location.resolve()}\' was not found!')
        else:
            return None
    except IOError as error:
        raise RuntimeError(f'An IO error occurred: \'{error}\'.')
    except AssertionError:
        raise RuntimeError(f'File \'{location.resolve()}\' isn\'t an object!')


def save_json_to_disk(contents: Dict[str, Any],
                      location: Path) -> None:
    """Saves the `contents` as JSON to disk, at the specified `location`.

    :param contents: The contents to save to disk.
    :param location: An absolute or relative file system location to save to.
    :throws: `RuntimeError` when anything abnormal happens.
    """
    try:
        with open(location, 'w') as handle:
            json.dump(contents, handle)
    except FileNotFoundError:
        raise RuntimeError(f'File \'{location.resolve()}\' was not found!')
    except IOError as error:
        raise RuntimeError(f'An IO error occurred: \'{error}\'.')


class QuestionAnswerPair:
    """A question-answer pair of from a (modified) LC-QuAD 2.0 dataset
    split.
    """

    class Part(Enum):
        """A part of a question-answer pair: the question or the answer."""
        QUESTION = 'question'
        ANSWER = 'answer'

    def __init__(self, uid: int, question: str, answer: str) -> None:
        """Constructs a question-answer pair.
        
        :param uid: The pair's unique identifier.
        :param question: The pair's question component.
        :param answer: The pair's answer component.
        """
        self.uid = uid
        self.question = question
        self.answer = answer

    def part(self, which: Part) -> str:
        """Returns the requested part of this question-answer pair.
        
        :param which: The part to access.
        :returns: The requested part.
        :throws: `ValueError` if an unsupported `Part` is requested.
        """
        if which == QuestionAnswerPair.Part.QUESTION:
            return self.question
        elif which == QuestionAnswerPair.Part.ANSWER:
            return self.answer
        else:
            raise ValueError('Part type \'%s\' not supported!' %
                             (str(which),))


def ensure_directory_exists(location: Path) -> None:
    """Ensures the directory at location `location` exists.
    
    If it does not, this function tries to create it.

    :param location: The location the directory should be at.
    """
    if not os.path.exists(location):
        os.makedirs(location)
        print('Created directory \'%s\' as it wasn\'t there yet.' %
              (str(location.resolve()),))


def only_unique(li: List[str]) -> List[str]:
    """Returns the list with only the unique values stored within it.
    
    Importantly, this function preserves the order of `li`, unlike a simpler
    call to `list(set(li))`. Prefer this method if preserving order is
    important.

    :param li: The list to get only the unique values of.
    :returns: `li`, but with only the unique values. Order is preserved.
    """
    unique: List[str] = []
    for entry in li:
        if entry not in unique:
            unique.append(entry)
    return unique
