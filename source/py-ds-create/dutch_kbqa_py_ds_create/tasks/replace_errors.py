"""Symbols for replacing `ERROR`s in translated LC-QuAD 2.0 datasets."""

import re
from pathlib import Path
from dutch_kbqa_py_ds_create.lc_quad_2_0 import RESOURCES_DIR, Split
from dutch_kbqa_py_ds_create.utilities import NaturalLanguage, \
                                              json_loaded_from_disk, \
                                              save_json_to_disk
from typing import Dict, cast


def error_replacement_dict(split: Split,
                           language: NaturalLanguage) -> Dict[str, str]:
    """Returns a mapping from UIDs to `ERROR` replacements.

    :param split: The LC-QuAD 2.0 dataset split to target.
    :param language: The natural language for which to return replacements.
    :returns: The UID-to-replacements dictionary.
    :throws: `RuntimeError` when the dictionary cannot be loaded. For instance,
        it may not exist on disk.
    """
    location = RESOURCES_DIR / f'error_replacements_{split}_{language.value}.json'
    return cast(Dict[str, str],
                json_loaded_from_disk(location,
                                      upon_file_not_found='throw-error'))


def replace_errors(file_with_errors: Path,
                   file_without_errors: Path,
                   split: Split,
                   language: NaturalLanguage) -> None:
    """Replaces `ERROR`s in `file_with_errors` and writes the resultant
    file to `file_without_errors`.

    Both the input and output file (`file_with_errors` and
    `file_without_errors`, respectively) should be located in the
    `RESOURCES_DIR`.

    :param file_with_errors: The file with errors to use as starting point.
    :param file_without_errors: The file to write to once the result is
        obtained.
    :param split: The dataset split to work on.
    :param language: The natural language to work on.
    :throws: `RuntimeError` when any IO anomaly occurs in the process.
    """
    uid_question_map = json_loaded_from_disk(RESOURCES_DIR / file_with_errors,
                                             upon_file_not_found='throw-error')
    assert(uid_question_map is not None)
    for uid, replacement in error_replacement_dict(split, language).items():
        uid_question_map[uid] = re.sub('(ERROR)[0-9]+',
                                       replacement,
                                       uid_question_map[uid])
        uid_question_map[uid] = re.sub('( )?(\\?)$', '?', uid_question_map[uid])
    save_json_to_disk(uid_question_map,
                      RESOURCES_DIR / file_without_errors)
