"""Symbols for finalising the creation of derived datasets of LC-QuAD 2.0."""

import re
import math
from dutch_kbqa_py_ds_create.lc_quad_2_0 import Split, DATASET_DIR
from dutch_kbqa_py_ds_create.utilities import NaturalLanguage, \
                                              QuestionAnswerPair, \
                                              json_loaded_from_disk, \
                                              ensure_directory_exists
from typing import List, Union, Literal, Dict, Callable, AnyStr, Tuple


FINALISED_DATASET_DIR = DATASET_DIR / "finalised"

Partition = Union[Literal['train'], Literal['validation'], Literal['test']]
PartitionedQAPairs = Dict[Partition, List[QuestionAnswerPair]]


def loaded_pre_finalised_dataset_split(split: Split,
                                       language: NaturalLanguage) -> \
        List[QuestionAnswerPair]:
    """Returns the almost-finalised question-answer pairs: the 'input data' to
    this module's various functions.
    
    :param split: The LC-QuAD 2.0 dataset split to target.
    :param language: The natural language for which to return replacements.
    :returns: A series of question-answer pairs.
    :throws: `RuntimeError` when the data cannot be loaded. For instance, it
        may not exist on disk.
    """
    location = DATASET_DIR / \
               f'{split}-{language.value}-replaced-no-errors-masked.json'
    raw = json_loaded_from_disk(location, upon_file_not_found='throw-error')
    out: List[QuestionAnswerPair] = []
    for uid_str, qa_pair in raw.items():
        uid = int(uid_str)
        out.append(QuestionAnswerPair(uid=uid,
                                      question=qa_pair['q'],
                                      answer=qa_pair['a']))
    return out


def string_with_substitutions(string: str,
                              substitutions: List[Tuple[str, str]]) -> str:
    """Returns the string, but with the requested substitutions performed on
    it.
    
    Substitutions are performed in the order of the list.
    
    :param string: The string to manipulate.
    :param substitutions: The substitutions to perform. Order matters.
    :returns: The manipulated string.
    """
    for before, after in substitutions:
        string = re.sub(before, after, string)
    return string


def answer_with_variables_replaced(answer: str) -> str:
    """Returns the given question-answer pair answer, except that the variables
    have been replaced with more word-like counterparts.
    
    :param answer: The original answer.
    :returns: The manipulated answer.
    """
    variable_reg_exp = '(\?)[^ ]+'  # e.g. `?ans_1`
    matches = set(match.group() for match in re.finditer(variable_reg_exp, answer))
    for index, match in enumerate(matches):
        answer = re.sub(f'(\\?{match[1:]})', f'var_{index + 1}', answer)
    return answer


def post_processed_question(question: str) -> str:
    """Adds space around masked entities and properties, makes question marks
    standalone, and removes duplicate spaces.
    
    :param question: The question to post-process.
    :returns: The post-processed question.
    """
    add_space: Callable[[re.Match[AnyStr]], str] = lambda match: f' {match.group()} '
    question = question.lower()
    question = re.sub('[pq][0-9]+', add_space, question)
    return string_with_substitutions(question,
                                     [('(\?)$', ' ?'),
                                      ('( ){2,}', ' ')],)


def post_processed_answer(answer: str) -> str:
    """Replaces various special symbols in the supplied answer by more
    word-like equivalents, removes mentions to WikiData namespaces, and deletes
    excessive whitespace.
    
    :param answer: The answer to post-process.
    :returns: The post-processed answer.
    """
    answer = answer.lower()
    answer = string_with_substitutions(answer,
                                       [('[\{]', ' brack_open '),
                                        ('[\}]', ' brack_close '),
                                        ('[\(]', ' attr_open '),
                                        ('[\)]', ' attr_close '),
                                        ('[\.]', ' sep_dot '),
                                        ('[,]', ' , '),
                                        ('([a-z]+:)(?=[pq][0-9]+)', '')])
    answer = answer_with_variables_replaced(answer)
    answer = string_with_substitutions(answer,
                                       [('[ ]{2,}', ' '),
                                        ('( )+$', '')])
    return answer


def post_processed_question_answer_pair(qa_pair: QuestionAnswerPair) -> \
        QuestionAnswerPair:
    """Post-processes an almost-finalised question-answer pair.
    
    :param qa_pair: The question-answer pair to post-process.
    :returns: The post-processed question-answer pair.
    """
    return QuestionAnswerPair(uid=qa_pair.uid,
                              question=post_processed_question(qa_pair.question),
                              answer=post_processed_answer(qa_pair.answer))


def partitioned_question_answer_pairs(qa_pairs: List[QuestionAnswerPair],
                                      split: Split,
                                      fraction_to_validate: float) -> \
        PartitionedQAPairs:
    """Returns the given question-answer pairs, but partitioned into one or
    more subsets.

    Note: `fraction_to_validate` only has effect if `split` is `'train'`,
    because our implementation divides the validation fraction purely from the
    original LC-QuAD 2.0 training part (and not the LC-QuAD 2.0 testing part).
    
    :param qa_pairs: The question-answer pairs to partition.
    :param split: The LC-QuAD 2.0 dataset split that `qa_pairs` belong to.
    :param fraction_to_validate: The fraction (0.0 and 1.0 both inclusive)
        of the pairs that should go to validation.
    :returns: The partitioning of the question-answer pairs.
    """
    if split == 'test':
        return {'test': qa_pairs}
    elif split == 'train':
        n_to_validate = math.floor(len(qa_pairs) * fraction_to_validate)
        return {'train': qa_pairs[n_to_validate:],
                'validate': qa_pairs[:n_to_validate]}


def save_partitioned_question_answer_pairs(partitioned: PartitionedQAPairs,
                                           language: NaturalLanguage) -> None:
    """Saves the post-processed and partitioned question-answer paris to disk.
    
    :param partitioned: The post-processed and partitioned question-answer
        pairs.
    :param language: The natural language in which the questions are expressed.
    """
    ensure_directory_exists(FINALISED_DATASET_DIR)
    for partition, qa_pairs in partitioned.items():
        questions = [qa_pair.question + '\n' for qa_pair in qa_pairs]
        answers = [qa_pair.answer + '\n' for qa_pair in qa_pairs]
        with open(FINALISED_DATASET_DIR / f'{partition}-{language.value}.txt',
                  'w') as handle:
            handle.writelines(questions)
        with open(FINALISED_DATASET_DIR / f'{partition}-sparql.txt',
                  'w') as handle:
            handle.writelines(answers)


def finalise_dataset_split(split: Split,
                           language: NaturalLanguage,
                           fraction_to_validate: float) -> None:
    """Performs two final operations on the given derived LC-QuAD 2.0 dataset
    split.

    The two operations are: (1) replacing various special characters on the
    SPARQL-side of the dataset by more word-like equivalents, and (2) splitting
    the dataset into a 'train'-'validate'-'test' tri-partition.

    :param split: The LC-QuAD 2.0 dataset split to work with.
    :param language: The natural language to work with.
    :param fraction_to_validate: The fraction of the dataset that should go to
        the validation (or, 'development') part of the dataset.
    """
    qa_pairs = loaded_pre_finalised_dataset_split(split, language)
    partitioned = partitioned_question_answer_pairs(qa_pairs,
                                                    split,
                                                    fraction_to_validate)
    save_partitioned_question_answer_pairs(partitioned, language)
