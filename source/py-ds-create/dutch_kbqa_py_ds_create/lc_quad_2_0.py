"""Symbols for working with the LC-QuAD 2.0 dataset."""

import os
import json
from dutch_kbqa_py_ds_create.utilities import ROOT_DIR
from typing import Union, Literal, TypedDict, List, Optional, cast


RESOURCES_DIR = (ROOT_DIR /
                 os.pardir /
                 os.pardir /
                 os.pardir /
                 'resources').resolve()
TRAIN_FILE = (RESOURCES_DIR / 'train.json').resolve()
TEST_FILE = (RESOURCES_DIR / 'test.json').resolve()


Split = Union[Literal['train'], Literal['test']]


class LCQuADQAPair(TypedDict):
    """A single LC-QuAD 2.0 question-answer (QA) pair.
    
    The `subgraph`, `template`, and `paraphrased_question` keys may be empty
    lists. In this case, the value should be viewed as being empty (`None`).
    """
    NNQT_question: str
    uid: int
    subgraph: Union[str, List]
    template_index: int
    question: Optional[str]
    sparql_wikidata: str
    sparql_dbpedia18: str
    template: Union[str, List]
    answer: List
    template_id: Union[str, int]
    paraphrased_question: Union[str, List]


def dataset_split(split: Split) -> List[LCQuADQAPair]:
    """Returns the desired `split` of the LC-QuAD 2.0 dataset.
    
    :param split: A dataset split. Either 'train' or 'test'.
    :returns: The desired split. It is returned as a sequence of LC-QuAD 2.0
        question-answer pairs.
    :throws: `RuntimeError` when the dataset split somehow couldn't be read.
    """
    file = TRAIN_FILE if split == 'train' else TEST_FILE
    try:
        with open(file, mode='r') as handle:
            ds_split = cast(List[LCQuADQAPair], json.load(handle))
    except FileNotFoundError:
        raise RuntimeError(f'File \'{file.resolve()}\' was not found!')
    except IOError as error:
        raise RuntimeError(f'An IO error occurred: \'{error}\'.')
    return ds_split


LCQuADQuestionType = Union[Literal['NNQT_question'],
                           Literal['question'],
                           Literal['paraphrased_question']]


def question_type_for_translation(qa_pair: LCQuADQAPair) -> LCQuADQuestionType:
    """Returns the question type to use for the given LC-QuAD 2.0 `qa_pair`.
    
    :param qa_pair: A question-answer pair.
    :returns: The question type to use.
    """
    q_type: LCQuADQuestionType
    if type(qa_pair['question']) == str and len(qa_pair['question']) > 15:
        q_type = 'question'
    elif type(qa_pair['paraphrased_question']) == str and \
         len(qa_pair['paraphrased_question']) > 20:
        q_type = 'paraphrased_question'
    else:
        q_type = 'NNQT_question'
    assert isinstance(qa_pair[q_type], str)  # should be available
    return q_type
