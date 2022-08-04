"""Symbols for running the main program from the command-line."""

from argparse import ArgumentParser
from pathlib import Path
from dutch_kbqa_py_ds_create.lc_quad_2_0 import RESOURCES_DIR, Split
from dutch_kbqa_py_ds_create.tasks.translate import \
    translate_complete_dataset_split_questions
from dutch_kbqa_py_ds_create.tasks.validate import \
    validate_against_reference
from dutch_kbqa_py_ds_create.tasks.replace_errors import replace_errors
from dutch_kbqa_py_ds_create.utilities import NaturalLanguage 
from enum import Enum
from typing import List, Optional, TypedDict


ARG_PARSE_BOOLEAN_TRUE = ['True', 'true', 'T', 't']
ARG_PARSE_BOOLEAN_FALSE = ['False', 'false', 'F', 'f']


def boolean_argument_parser_choices() -> List[str]:
    """Returns strings that the argument parser considers as booleans.

    :returns: Strings regarded as being boolean values.
    """
    return [*ARG_PARSE_BOOLEAN_TRUE, *ARG_PARSE_BOOLEAN_FALSE]


def interpret_boolean_argument_parser_choice(choice: str) -> bool:
    """Interprets a string as an argument parser boolean.

    :param choice: The command-line choice string to interpret.
    :returns: A Python primitive boolean value.
    :throws: `ValueError` if `choice` cannot be interpreted as a `bool`.
    """
    if choice in ARG_PARSE_BOOLEAN_TRUE:
        return True
    elif choice in ARG_PARSE_BOOLEAN_FALSE:
        return False
    else:
        raise ValueError(f'Choice \'{choice}\' cannot be interpreted as ' +
                         'boolean. Use one of the following values: ' +
                         f'{ARG_PARSE_BOOLEAN_TRUE + ARG_PARSE_BOOLEAN_FALSE}')


class TaskType(Enum):
    """A type of task to perform."""
    TRANSLATE = 'translate'
    VALIDATE = 'validate'
    REPLACE_ERRORS = 'replace-errors'


def dutch_kbqa_python_dataset_creation_argument_parser() -> ArgumentParser:
    """Returns a command-line argument parser for this program.
    
    :returns: The command-line argument parser.
    """
    parser = ArgumentParser(description='Create derived datasets of ' +
                                        'LC-QuAD 2.0 using Python.')
    parser.add_argument('-t', '--task',
                        type=str,
                        help='The manipulation to perform.',
                        choices=[task.value for task in TaskType],
                        required=True)
    parser.add_argument('--split',
                        type=str,
                        help='The dataset split to work on.',
                        choices=['train', 'test'])
    parser.add_argument('--language',
                        type=str,
                        help='The language to translate into.',
                        choices=[lang.value for lang in NaturalLanguage])
    parser.add_argument('--load_file_name',
                        type=str,
                        help='The name of the file to load from.')
    parser.add_argument('--save_file_name',
                        type=str,
                        help='The name of the file to save to.')
    parser.add_argument('--save_frequency',
                        type=int,
                        help='The number of operations to perform before ' +
                             'saving to disk.')
    parser.add_argument('--reference_file_name',
                        type=str,
                        help='A reference file to validate against.')
    parser.add_argument('--quiet',
                        type=str,
                        choices=ARG_PARSE_BOOLEAN_TRUE + 
                                ARG_PARSE_BOOLEAN_FALSE)
    return parser


class DutchKBQADSCreationDict(TypedDict):
    """A mapping from command-line arguments to values, specifically designed
    for the context of this program.
    """
    task: TaskType
    split: Optional[Split]
    language: Optional[NaturalLanguage]
    load_file_name: Optional[Path]
    save_file_name: Optional[Path]
    save_frequency: Optional[int]
    reference_file_name: Optional[Path]
    quiet: bool


def dutch_kbqa_dataset_creation_namespace_to_dict(parser: ArgumentParser) -> \
        DutchKBQADSCreationDict:
    """Returns a parsed namespace, designed specifically for this program.
    
    :param parser: A command-line argument parser from which to obtain a
        namespace.
    :returns: The parsed namespace, returned as a typed Python dictionary.
    """
    ns = parser.parse_args()
    return {'task': TaskType(ns.task),
            'split': ns.split if 'split' in ns else None,
            'language': NaturalLanguage(ns.language)
                        if ns.language is not None else None,
            'load_file_name': RESOURCES_DIR / ns.load_file_name
                              if ns.load_file_name is not None else None,
            'save_file_name': RESOURCES_DIR / ns.save_file_name
                              if ns.save_file_name is not None else None,
            'save_frequency': ns.save_frequency
                              if ns.save_frequency is not None else None,
            'reference_file_name': Path(ns.reference_file_name).resolve()
                                   if ns.reference_file_name is not None else
                                   None,
            'quiet': interpret_boolean_argument_parser_choice(ns.quiet)
                     if ns.quiet is not None else None}


def act_on_dutch_kbqa_dataset_creation_dict(di: DutchKBQADSCreationDict) -> \
        None:
    """Starts up the relevant sub-program based on the parsed namespace, `di`.
    
    :param di: The parsed namespace. A typed Python dictionary.
    """
    if di['task'] == TaskType.TRANSLATE:
        translate_complete_dataset_split_questions(di['split'],
                                                   di['language'],
                                                   di['save_file_name'],
                                                   di['save_frequency'],
                                                   di['quiet'])
    elif di['task'] == TaskType.VALIDATE:
        result = validate_against_reference(proposal_file=di['save_file_name'],
                                            reference_file=di['reference_file_name'])
        print('Are the same? %s.' % (result,))
    elif di['task'] == TaskType.REPLACE_ERRORS:
        replace_errors(di['load_file_name'],
                       di['save_file_name'],
                       di['split'],
                       di['language'])
    else:
        raise NotImplementedError(f'Task type \'{di["task"]}\' not yet ' +
                                  'supported.')


if __name__ == '__main__':
    parser = dutch_kbqa_python_dataset_creation_argument_parser()
    di = dutch_kbqa_dataset_creation_namespace_to_dict(parser)
    act_on_dutch_kbqa_dataset_creation_dict(di)
