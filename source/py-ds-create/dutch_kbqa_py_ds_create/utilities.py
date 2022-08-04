"""Various utility symbols."""

import os
import json
from pathlib import Path
from enum import Enum
from typing import Union, Literal, Dict, Any, Optional


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
    print(text, flush=True)


FileNotFoundReaction = Union[Literal['throw-error'],
                             Literal['return-none']]


def json_loaded_from_disk(location: Path,
                          upon_file_not_found: FileNotFoundError) -> \
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
