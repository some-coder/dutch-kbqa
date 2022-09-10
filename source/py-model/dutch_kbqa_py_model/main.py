"""Symbols for running the main program from the command-line."""

from argparse import ArgumentParser
from pathlib import Path, PurePosixPath
from dutch_kbqa_py_model.model.run_transformer import SUPPORTED_MODEL_TRIPLES, \
                                                      TransformerRunner
from dutch_kbqa_py_model.utilities import TRUE_STRINGS, \
                                          FALSE_STRINGS, \
                                          NO_DISTRIBUTION_RANK, \
                                          NaturalLanguage, \
                                          QueryLanguage, \
                                          hugging_face_hub_model_exists, \
                                          string_is_existing_hugging_face_hub_model
from typing import List, Union


def boolean_argument_parser_choices() -> List[str]:
    """Returns strings that the argument parser considers as booleans.

    :returns: Strings regarded as being boolean values.
    """
    return [*TRUE_STRINGS, *FALSE_STRINGS]


def interpret_boolean_argument_parser_choice(choice: str) -> bool:
    """Interprets a string as an argument parser boolean.

    :param choice: The command-line choice string to interpret.
    :returns: A Python primitive boolean value.
    :throws: `ValueError` if `choice` cannot be interpreted as a `bool`.
    """
    if choice in TRUE_STRINGS:
        return True
    elif choice in FALSE_STRINGS:
        return False
    else:
        raise ValueError(f'Choice \'{choice}\' cannot be interpreted as ' +
                         'boolean. Use one of the following values: ' +
                         f'{TRUE_STRINGS + FALSE_STRINGS}')


def dutch_kbqa_python_model_argument_parser() -> ArgumentParser:
    """Returns a command-line argument parser for this program.
    
    :returns: The command-line argument parser.
    """
    parser = ArgumentParser(description='Fine-tune BERT-based transformers ' +
                                        'for WikiData question-answering.')
    # Required arguments.
    parser.add_argument('--enc_model_type',
                        type=str,
                        help='The encoder language model type.',
                        choices=SUPPORTED_MODEL_TRIPLES.keys(),
                        required=True)
    parser.add_argument('--dec_model_type',
                        type=str,
                        help='The decoder language model type.',
                        choices=SUPPORTED_MODEL_TRIPLES.keys(),
                        required=True)
    parser.add_argument('--enc_id_or_path',
                        type=str,
                        help='A file system path to a pre-trained encoder ' +
                             'language model (enclosing folder or ' +
                             'configuration JSON file), or a model ID of a ' +
                             'model hosted on `huggingface.co`.',
                        required=True)
    parser.add_argument('--dec_id_or_path',
                        type=str,
                        help='A file system path to a pre-trained decoder ' +
                             'language model (enclosing folder or ' +
                             'configuration JSON file), or a model ID of a ' +
                             'model hosted on `huggingface.co`.',
                        required=True)
    parser.add_argument('--dataset_dir',
                        type=str,
                        help='A file system path to a directory. The ' +
                             'directory under which the training, validation' +
                             ' and testing data resides.',
                        required=True)
    parser.add_argument('--natural_language',
                        type=str,
                        help='A natural language. The input language of the ' +
                             'transformer.',
                        choices=[lang.value for lang in NaturalLanguage],
                        required=True)
    parser.add_argument('--query_language',
                        type=str,
                        help='A query language. The output language of the ' +
                             'transformer.',
                        choices=[lang.value for lang in QueryLanguage],
                        required=True)
    parser.add_argument('--max_natural_language_length',
                        type=int,
                        help='The maximum (inclusive) number of tokens to ' +
                             'include in tokenised natural language ' +
                             'inputs. Truncation and padding occur for too ' +
                             'long and too short sequences, respectively. ' +
                             'Must be strictly positive.',
                        required=True) 
    parser.add_argument('--max_query_language_length',
                        type=int,
                        help='The maximum (inclusive) number of tokens to ' +
                             'include in tokenised query language ' +
                             'outputs. Truncation and padding occur for too ' +
                             'long and too short sequences, respectively. ' +
                             'Must be strictly positive.',
                        required=True)
    parser.add_argument('--learning_rate',
                        type=float,
                        help='The initial learning rate for the Adam ' +
                             'optimiser. Must be strictly positive.',
                        required=True)
    parser.add_argument('--beam_size',
                        type=int,
                        help='The beam size to use in the beam search at ' +
                             'the transformer\'s output layer (for queries).' +
                             ' Must be strictly positive.',
                        required=True)
    parser.add_argument('--perform_training',
                        type=str,
                        help='Whether to perform the training stage.',
                        choices=TRUE_STRINGS +
                                FALSE_STRINGS,
                        required=True)
    parser.add_argument('--perform_validation',
                        type=str,
                        help='Whether to perform the validation stage.',
                        choices=TRUE_STRINGS +
                                FALSE_STRINGS,
                        required=True)
    parser.add_argument('--perform_testing',
                        type=str,
                        help='Whether to perform the testing stage.',
                        choices=TRUE_STRINGS +
                                FALSE_STRINGS,
                        required=True)
    parser.add_argument('--save_dir',
                        type=str,
                        help='A file system path to a directory. The ' +
                             'directory under which to save transformer ' +
                             'checkpoints and model predictions.',
                        required=True)
    parser.add_argument('--seed',
                        type=int,
                        help='A pseudo-random number generator (PRNG) ' +
                             'initialisation value to use. (This argument ' +
                             'is required to encourage reproducibility in ' +
                             'model results. Take care to switch seeds if ' +
                             'it is your intention to obtain varying ' +
                             'results.) Must be an integer in the range [1, ' +
                             '2^32 - 1], both ends inclusive.',
                        required=True)
    # Optional or situationally required arguments.
    parser.add_argument('--enc_config_name',
                        type=str,
                        help='An encoder language model configuration if ' +
                             'you don\'t wish to use the default one ' +
                             'associated with `enc_model_type`.')
    parser.add_argument('--dec_config_name',
                        type=str,
                        help='A decoder language model configuration if ' +
                             'you don\'t wish to use the default one ' +
                             'associated with `dec_model_type`.')
    parser.add_argument('--enc_tokeniser_name',
                        type=str,
                        help='An encoder language model tokeniser if' +
                             ' you don\'t wish to use the default one ' +
                             'associated with `enc_model_type`.')
    parser.add_argument('--dec_tokeniser_name',
                        type=str,
                        help='A decoder language model tokeniser if' +
                             ' you don\'t wish to use the default one ' +
                             'associated with `dec_model_type`.')
    parser.add_argument('--treat_transformer_as_uncased',
                        type=str,
                        default='false',
                        choices=TRUE_STRINGS +
                                FALSE_STRINGS,
                        help='Whether to treat the transformer as an uncased' +
                             'model.')
    parser.add_argument('--use_cuda',
                        type=str,
                        default='true',
                        choices=TRUE_STRINGS +
                                FALSE_STRINGS,
                        help='Whether to use CUDA if it is available.')
    parser.add_argument('--training_batch_size',
                        type=int,
                        help='(Only required when training.) The batch size ' +
                             'per GPU or CPU during training. Must be ' +
                             'strictly positive.')
    parser.add_argument('--non_training_batch_size',
                        type=int,
                        help='(Only required when validating or testing.) ' +
                             'The batch size per GPU or CPU during ' +
                             'anything except training. Must be strictly ' +
                             'positive.')
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help='The number of parameter update steps to ' +
                             'accumulate before performing a single ' +
                             'backpropagation. Must be strictly positive.')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.,
                        help='The weight decay scalar. Must be non-negative.')
    parser.add_argument('--adam_epsilon',
                        type=float,
                        default=1e-8,
                        help='A denominator numerical stability term to ' +
                             'use for Adam. Is \'epsilon hat\' on page ' +
                             '2 of Kingma and Ba (2014). Must be strictly ' +
                             'positive.')
    parser.add_argument('--training_epochs',
                        type=int,
                        help='(Only required when training.) The number of ' +
                             'training epochs to perform. Must be strictly ' +
                             'positive.')
    parser.add_argument('--local_rank',
                        type=int,
                        default=NO_DISTRIBUTION_RANK,
                        help='A local rank for processes to use during ' +
                             'distributed training. If given explicitly, a ' +
                             'strictly non-negative integer or the special ' +
                             'value `NO_DISTRIBUTION_RANK` if you wish not ' +
                             'to use distributed execution.')
    parser.add_argument('--save_frequency',
                        type=int,
                        default=1,
                        help='The number of epochs to complete before ' +
                             'performing a(nother) save to disk. Must be ' +
                             'strictly positive.')
    parser.add_argument('--load_file',
                        type=str,
                        help='(Only required when testing.) A file ' +
                             'system path to a `.bin` file. The path to a ' +
                             'trained transformer.')
    return parser


def language_model_id_or_path(id_or_path: str) -> Union[str, Path]:
    """Returns a language model ID or path, given a namespace string that
    should represent it.
    
    :param id_or_path: The namespace string.
    :returns: The language model ID or path.
    """
    if len(id_or_path) == 0:
        raise ValueError('Language model ID or path should be non-empty!')
    path = PurePosixPath(id_or_path)
    if len(path.parts) == 1:
        # Ambiguous (case one).
        return id_or_path \
               if hugging_face_hub_model_exists(author=None,
                                                model=path.parts[0]) else \
               Path(id_or_path)
    elif len(path.parts) == 2:
        # Ambiguous (case two).
        return id_or_path \
               if hugging_face_hub_model_exists(author=path.parts[0],
                                                model=path.parts[1]) else \
               Path(id_or_path)
    else:
        # Unambiguous: Must be an file system model path.
        return Path(id_or_path)


def dutch_kbqa_model_namespace_to_runner(parser: ArgumentParser) -> \
        TransformerRunner:
    """Returns an initialised transformer runner from the parsed input
    arguments.
    
    :param parser: A command-line argument parser from which to obtain a
        namespace.
    :returns: An initialised transformer runner.
    """
    ns = parser.parse_args()
    
    train = interpret_boolean_argument_parser_choice(ns.perform_training)
    validate = interpret_boolean_argument_parser_choice(ns.perform_validation)
    test = interpret_boolean_argument_parser_choice(ns.perform_testing)
    
    assert(ns.max_natural_language_length > 0)
    assert(ns.max_query_language_length > 0)
    assert(ns.learning_rate > 0.)
    assert(ns.beam_size > 0)
    assert(ns.seed >= 0)
    if ns.enc_config_name is not None:
        assert(string_is_existing_hugging_face_hub_model(ns.enc_config_name))
    if ns.dec_config_name is not None:
        assert(string_is_existing_hugging_face_hub_model(ns.dec_config_name))
    if ns.enc_tokeniser_name is not None:
        assert(string_is_existing_hugging_face_hub_model(ns.enc_tokeniser_name))
    if ns.dec_tokeniser_name is not None:
        assert(string_is_existing_hugging_face_hub_model(ns.dec_tokeniser_name))
    assert(ns.gradient_accumulation_steps > 0)
    assert(ns.weight_decay >= 0.)
    assert(ns.adam_epsilon > 0.)
    assert(ns.local_rank == NO_DISTRIBUTION_RANK or ns.local_rank >= 0)
    assert(ns.save_frequency > 0)
    if train:
        assert(ns.training_epochs is not None)
        assert(ns.training_epochs > 0)
        assert(ns.training_batch_size is not None)
        assert(ns.training_batch_size > 0)
    if not train:
        assert(ns.non_training_batch_size is not None)
        assert(ns.non_training_batch_size > 0)
    if test:
        assert(ns.load_file is not None)
    
    return TransformerRunner(
        enc_model_type=ns.enc_model_type,
        dec_model_type=ns.dec_model_type,
        enc_id_or_path=language_model_id_or_path(ns.enc_id_or_path),
        dec_id_or_path=language_model_id_or_path(ns.dec_id_or_path),
        dataset_dir=Path(ns.dataset_dir).resolve(),
        natural_language=NaturalLanguage(ns.natural_language),
        query_language=QueryLanguage(ns.query_language),
        max_natural_language_length=ns.max_natural_language_length,
        max_query_language_length=ns.max_query_language_length,
        learning_rate=ns.learning_rate,
        beam_size=ns.beam_size,
        perform_training=train,
        perform_validation=validate,
        perform_testing=test,
        save_dir=Path(ns.save_dir).resolve(),
        seed=ns.seed,
        enc_config_name=ns.enc_config_name,
        dec_config_name=ns.dec_config_name,
        enc_tokeniser_name=ns.enc_tokeniser_name,
        dec_tokeniser_name=ns.dec_tokeniser_name,
        treat_transformer_as_uncased=
            interpret_boolean_argument_parser_choice(ns.treat_transformer_as_uncased),
        use_cuda=
            interpret_boolean_argument_parser_choice(ns.use_cuda),
        training_batch_size=ns.training_batch_size,
        non_training_batch_size=ns.non_training_batch_size,
        gradient_accumulation_steps=ns.gradient_accumulation_steps,
        weight_decay=ns.weight_decay,
        adam_epsilon=ns.adam_epsilon,
        training_epochs=ns.training_epochs,
        local_rank=ns.local_rank,
        save_frequency=ns.save_frequency,
        load_file=
            None if ns.load_file is None else Path(ns.load_file).resolve()
    )


if __name__ == '__main__':
    parser = dutch_kbqa_python_model_argument_parser()
    runner = dutch_kbqa_model_namespace_to_runner(parser)
    runner.run()
