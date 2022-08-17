"""Symbols for validating entity and property masks of LC-QuAD 2.0 dataset
question-answer pairs.
"""

import re
import copy
from pathlib import Path
from dutch_kbqa_py_ds_create.utilities import json_loaded_from_disk, \
                                              QuestionAnswerPair
from typing import Dict, Any, List, Set, Union, Literal


MaskedQAPairsMap = Dict[int, QuestionAnswerPair]
SymbolSwitchDirection = Union[Literal['forward'], Literal['backward']]


def mask_file_to_masked_qa_pairs_map(mask_file: Dict[str, Any]) -> \
        MaskedQAPairsMap:
    """Returns the contents of `mask_file`, but presented as a mapping from
    UIDs to masked question-answer pairs.
    
    :param mask_file: The raw, Python-loaded JSON object.
    :returns: The map.
    """
    masks_map: MaskedQAPairsMap = {}
    for key, value in mask_file.items():
        uid = int(key)
        masks_map[uid] = QuestionAnswerPair(uid=uid,
                                            question=value['q'],
                                            answer=value['a'])
    return masks_map


SWITCH_MAP: Dict[SymbolSwitchDirection, Dict[str, str]] = \
    {'forward': {'P': 'R',
                 'Q': 'S'},
     'backward': {'R': 'P',
                  'S': 'Q'}}


def switched_mask_symbol(mask: str,
                         direction: SymbolSwitchDirection) -> str:
    legal_keys: List[str] = list(SWITCH_MAP[direction].keys())
    assert(re.search(f'^[{"".join(legal_keys)}][0-9]+$', mask))
    for key, switch_symbol in SWITCH_MAP[direction].items():
        if mask[0] == key:
            return f'{switch_symbol}{mask[1:]}'
    raise RuntimeError('Couldn\'t fit any key. Is the `direction` correct?')


def sentence_mask_symbols_switched(sen: str,
                                   direction: SymbolSwitchDirection) -> \
        QuestionAnswerPair:
    original_keys: List[str] = list(SWITCH_MAP[direction].keys())
    to_switch = re.findall(f'[{"".join(original_keys)}][0-9]+', sen)
    for symbol_to_switch in to_switch:
        switched_symbol = switched_mask_symbol(symbol_to_switch,
                                               direction)
        sen = sen.replace(symbol_to_switch, switched_symbol)
    return sen


def reference_to_proposal_masks_map(prp_pair: QuestionAnswerPair,
                                    ref_pair: QuestionAnswerPair,
                                    part: QuestionAnswerPair.Part) -> \
        Dict[str, str]:
    """Returns a mapping from reference masked entities and properties to
    analogous ones in the proposal.
    
    Note: `prp_pair` and `ref_pair` must share the same `uid`.

    :param prp_pair: A proposal question-answer pair.
    :param ref_pair: A reference question-answer pair.
    :param part: The part of the question-answer pairs to use for building the
        mapping.
    :returns: The mapping.
    """
    assert(prp_pair.uid == ref_pair.uid)
    prp_masks: List[str] = re.findall('[QP][0-9]+', prp_pair.part(which=part))
    ref_masks: List[str] = re.findall('[QP][0-9]+', ref_pair.part(which=part))
    try:
        assert(len(prp_masks) == len(ref_masks))
    except AssertionError:
        print('Mismatch in number of masks!')
        print('\t  (UID) %d' % (prp_pair.uid,))
        print('\t(Prop.) %s' % (str(prp_masks),))
        print('\t        %s' % (prp_pair.question,))
        print('\t (Ref.) %s' % (str(ref_masks),))
        print('\t        %s' % (ref_pair.question,))
        exit(1)
    out: Dict[str, str] = {}
    already_encounterd: Set[str] = set()
    for index, mask in enumerate(ref_masks):
        if mask in already_encounterd:
            continue  # try to find masks that haven't been mapped yet
        out[mask] = switched_mask_symbol(prp_masks[index], 'forward')
        already_encounterd.add(mask)
    return out


def reference_masks_replaced_by_analogous_proposal_masks(
        prp_pair: QuestionAnswerPair,
        ref_pair: QuestionAnswerPair,
        part: QuestionAnswerPair.Part) -> str:
    """Replaces within a reference question-answer pair the requested part's
    masks by analogous ones from the proposed pair.
    
    :param prp_pair: A proposal question-answer pair.
    :param ref_pair: A reference question-answer pair.
    :param part: The part of the question-answer pairs to use for building the
        mapping.
    :returns: The mapping.
    """
    masks_map = reference_to_proposal_masks_map(prp_pair, ref_pair, part)
    out = ref_pair.part(which=part)
    for ref_mask, prp_mask in masks_map.items():
        out = out.replace(ref_mask, prp_mask)
    return out


def successful_single_masks_validation(prp_pair: QuestionAnswerPair,
                                       ref_pair: QuestionAnswerPair,
                                       part: QuestionAnswerPair.Part) -> bool:
    """Determines whether, for a single question-answer pair, the proposal and
    reference match up in terms of masks.
    
    This function can take into account different numbering schemes (zero- or
    one-based indexing, for instance).

    :param prp_pair: A proposal question-answer pair.
    :param ref_pair: A reference question-answer pair.
    :param part: The part of the question-answer pairs to use during
        validation.
    :returns: The question's answer.
    """
    replaced = reference_masks_replaced_by_analogous_proposal_masks(prp_pair,
                                                                    ref_pair,
                                                                    part)
    replaced_2 = sentence_mask_symbols_switched(replaced, 'backward')
    return prp_pair.part(part) == replaced_2


def validate_masking_against_reference(proposal_file: Path,
                                       reference_file: Path) -> bool:
    """Determines whether JSON file `proposal_file`'s contents match those of
    `reference_file`'s. Meant for comparison of two masked versions of the
    same translated LC-QuAD dataset split, e.g. from different programs.

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
    prp_masks = mask_file_to_masked_qa_pairs_map(prp)
    ref_masks = mask_file_to_masked_qa_pairs_map(ref)
    try:
        assert(set(prp_masks.keys()) == set(ref_masks.keys()))
    except AssertionError:
        s1 = set(prp_masks.keys())
        s2 = set(ref_masks.keys())
        print('In 1 but not in 2: %s' % (s1.difference(s2),))
        print('In 2 but not in 1: %s' % (s2.difference(s1),))
        exit(1)
    diff_count: Dict[QuestionAnswerPair.Part, int] = \
        {QuestionAnswerPair.Part.QUESTION: 0,
         QuestionAnswerPair.Part.ANSWER: 0}
    for part in QuestionAnswerPair.Part:
        for uid in ref_masks.keys():
            prp_pair = prp_masks[uid]
            ref_pair = ref_masks[uid]
            if not successful_single_masks_validation(prp_masks[uid],
                                                      ref_masks[uid],
                                                      part):
                print('(UID=%6d)' % (uid,))
                print('\t(Prop.) \'%s\'' % (prp_pair.part(which=part)))
                print('\t (Ref.) \'%s\'' % (ref_pair.part(which=part)))
                diff_count[part] += 1
    print('Differences summary:') 
    for part in QuestionAnswerPair.Part:
        print('\t\'%8s\': %6d/%6d different (%6.2lf%%)' %
              (part.value,
               diff_count[part],
               len(ref_masks),  # count is the same for both parts
               (diff_count[part] / len(ref_masks)) * 100.))
    # Only if zero differences are produces with both `part`s do we regard the
    # two files as being equal, and the proposal file as being successfully
    # validated.
    return sum([count for count in diff_count.values()]) == 0
