"""Various utility symbols."""

import json
import os
import sys
from pathlib import Path
from enum import Enum


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
#    sys.stdout.write(text)
#    sys.stdout.flush()
    print(text, flush=True)


def validate_against_reference(proposal_file: Path,
                               reference_file: Path) -> bool:
    """Determines whether JSON file `proposal_file`'s key-values
    match those of `reference_file`'s.
    
    We only check for the keys existing in `proposal_file`; keys in
    `reference_file` but not in `proposal_file` are ignored.

    If any inconsistencies are encountered, these are printed to standard
    output.

    :param proposal_file: The JSON file that should be(come) valid.
    :param reference_file: The JSON file that serves as 'ground truth'.
    :returns: Whether `proposal_file` is valid with respect to
        `reference_file`.
    """
    try:
        with open(proposal_file, 'r') as handle:
            prp = json.load(handle)
    except FileNotFoundError:
        raise RuntimeError(f'Proposal file \'{proposal_file}\' not found!')
    except IOError as error:
        raise RuntimeError(f'An IO error occurred: \'{error}\'.')
    try:
        with open(reference_file, 'r') as handle:
            ref = json.load(handle)
    except FileNotFoundError:
        raise RuntimeError(f'Reference file \'{reference_file}\' not found!')
    except IOError as error:
        raise RuntimeError(f'An IO error occurred: \'{error}\'.')
    assert isinstance(prp, dict)
    assert isinstance(ref, dict)
    different_count = 0
    for key in prp.keys():
        prp_val = prp[key]
        ref_val = ref[key]
        if prp_val != ref_val:
            print('%5s: (pr.)  \'%s\',\n%s(ref.) \'%s\'.' %
                  (int(key),
                   prp_val,
                   ' ' * (5 + len(': ')),
                   ref_val))
            different_count += 1
    if different_count > 0:
        print('%d / %d (%6.2lf%%) of entries were different.' %
              (different_count,
               len(prp),
               (different_count / len(prp)) * 100.))
    return different_count == 0
