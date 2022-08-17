"""Symbols for validating translations of LC-QuAD 2.0 datasets."""

from pathlib import Path
from dutch_kbqa_py_ds_create.utilities import json_loaded_from_disk


def validate_translation_against_reference(proposal_file: Path,
                                           reference_file: Path) -> bool:
    """Determines whether JSON file `proposal_file`'s contents match those of
    `reference_file`'s. Meant for comparison of two translations of the same
    LC-QuAD 2.0 dataset split, e.g. at different times.
    
    We only check for the keys existing in `proposal_file`; keys in
    `reference_file` but not in `proposal_file` are ignored.

    If any inconsistencies are encountered, these are printed to standard
    output.

    :param proposal_file: The JSON file that should be(come) valid.
    :param reference_file: The JSON file that serves as 'ground truth'.
    :returns: Whether `proposal_file` is valid with respect to
        `reference_file`.
    :throws: `RuntimeError` when any IO anomaly occurs during validation.
    """
    prp = json_loaded_from_disk(proposal_file,
                                upon_file_not_found='throw-error')
    ref = json_loaded_from_disk(reference_file,
                                upon_file_not_found='throw-error')
    assert(prp is not None)
    assert(ref is not None)
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
