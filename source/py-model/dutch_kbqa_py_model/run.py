"""Symbols for running BERT-based transformer fine-tuning operations."""

from pathlib import Path, PurePosixPath
from transformers import PretrainedConfig, \
                         PreTrainedModel, \
                         PreTrainedTokenizer, \
                         BertConfig, \
                         BertModel, \
                         BertTokenizer, \
                         RobertaConfig, \
                         RobertaModel, \
                         RobertaTokenizer
from dutch_kbqa_py_model.utilities import NaturalLanguage, \
                                          QueryLanguage
from typing import NamedTuple, Dict, Type, Union, Literal, Set, Optional


class ModelTriple(NamedTuple):
    """An en- and decoder language model triple.

    The triple stores three values: (1) a type of model configuration, (2) a
    type of model, and (3) a type of model tokeniser.
    """
    configuration: Type[PretrainedConfig]
    model: Type[PreTrainedModel]
    tokeniser: Type[PreTrainedTokenizer]


SupportedModelType = Union[Literal['bert'], Literal['roberta']]
SupportedArchitecture = Union[Literal['bert-random'], Literal['bert-bert']]


SUPPORTED_MODEL_TRIPLES: Dict[SupportedModelType, ModelTriple] = \
    {'bert': ModelTriple(BertConfig, BertModel, BertTokenizer),
     'roberta': ModelTriple(RobertaConfig, RobertaModel, RobertaTokenizer)}
SUPPORTED_ARCHITECTURES: Set[SupportedArchitecture] = set(['bert-random',
                                                           'bert-bert'])


class TransformerRunner:
    """A convenience class that helps you run transformer models."""

    def __init__(self,
                 model_type: SupportedModelType,
                 model_architecture: SupportedArchitecture,
                 encoder_id_or_path: Union[str, PurePosixPath],
                 decoder_id_or_path: Union[str, PurePosixPath],
                 dataset_dir: Path,
                 natural_language: NaturalLanguage,
                 query_language: QueryLanguage,
                 max_natural_language_length: int,
                 max_query_language_length: int,
                 learning_rate: float,
                 beam_size: int,
                 perform_training: bool,
                 perform_validation: bool,
                 perform_testing: bool,
                 save_dir: Path,
                 seed: int,
                 config_name: Optional[str],
                 tokeniser_name: Optional[str],
                 treat_transformer_as_uncased: bool,
                 use_cuda: bool,
                 training_batch_size: Optional[int],
                 non_training_batch_size: Optional[int],
                 gradient_accumulation_steps: int,
                 weight_decay: float,
                 adam_epsilon: float,
                 training_epochs: Optional[int],
                 local_rank: int,
                 save_frequency: int,
                 load_file: Optional[Path]) -> None:
        """Constructs a new transformer runner.
        
        :param model_type: The en- and decoder language model type.
        :param model_architecture: The transformer architecture.
        :param encoder_id_or_path: A file system path to a pre-trained encoder
            language model (enclosing folder or configuration JSON file), or a
            model ID of a model hosten on `huggingface.co`.
        :param decoder_id_or_path: A file system path to a pre-trained decoder
            language model (enclosing folder or configuration JSON file), or a
            model ID of a model hosten on `huggingface.co`.
        :param dataset_dir: A file system path to a directory. The directory
            under which the training, validation and testing data resides.
        :param natural_language: A natural language. The input language of the
            transformer.
        :param query_language: A query language. The output language of the
            transformer.
        :param max_natural_language_length: The maximum (inclusive) number of
            tokens to include in tokenised natural language inputs. Truncation
            and padding occur for too long and too short sequences,
            respectively. Must be strictly positive.
        :param learning_rate: The initial learning rate for the Adam optimiser.
            Must be strictly positive.
        :param beam_size: The beam size to use in the beam search at the
            transformer's output layer (for queries). Must be strictly
            positive.
        :param perform_training: Whether to perform the training phase.
        :param perform_validation: Whether to perfrom the validation phase.
        :param perform_testing: Whether to perform the testing phase.
        :param save_dir: A file system path to a directory. The directory under
            which to save transformer checkpoints and model predictions.
        :param seed: A pseudo-random number generator (PRNG) initialisation
            value to use. (This argument is required to encourage
            reproducibility in model results. Take care to switch seeds if it
            is your intention to obtain varying results.) Must be non-negative.
        :param config_name: An en- and decoder language model configuration if
            you don't wish to use the default one associated with `model_type`.
        :param tokeniser_name: An en- and decoder language model tokeniser if
            you don't wish to use the default one associated with `model_type`.
        :param treat_transformer_as_uncased: Whether to treat the transformer
            as an uncased model.
        :param use_cuda: Whether to use CUDA if it is available.
        :param training_batch_size: (Only required when `perform_training` is
            `True`.) The batch size per GPU or CPU during training. Must be
            strictly positive.
        :param non_training_batch_size: (Only required when `perform_validation`
            or `perform_testing` is `True`, or if both are `True`.) The batch 
            size per GPU or CPU during anything but training. Must be strictly
            positive.
        :param gradient_accumulation_steps: The number of parameter update
            steps to accumulate before performing a single backpropagation.
            Must be strictly positive.
        :param weight_decay: The weight decay scalar. Must be non-negative.
        :param adam_epsilon: A denominator numerical stability term to use for
            Adam. Is 'epsilon_hat' on page 2 of Kingma and Ba (2014). Must be
            strictly positive.
        :param training_epochs: (Only required when `perform_training` is
            `True`.) The number of training epochs to perform. Must be strictly
            positive.
        :param local_rank: A local rank for processes to use during distributed
            training. If given explicitly, a strictly non-negative integer or
            the special value `NO_LOCAL_RANK` if you wish not to use
            distributed execution.
        :param save_frequency: The number of epochs to complete before
            performing a(nother) save to disk. Must be strictly positive.
        :param load_file: (Only required when `perform_testing` is `True`.) A
            file system path to a `.bin` file. The path to a trained
            transformer.
        """
        self.model_type = model_type
        self.model_architecture = model_architecture
        self.encoder_id_or_path = encoder_id_or_path
        self.decoder_id_or_path = decoder_id_or_path
        self.dataset_dir = dataset_dir
        self.natural_language = natural_language
        self.query_language = query_language
        self.max_natural_language_length = max_natural_language_length
        self.max_query_language_length = max_query_language_length
        self.learning_rate = learning_rate
        self.beam_size = beam_size
        self.perform_training = perform_training
        self.perform_validation = perform_validation
        self.perform_testing = perform_testing
        self.save_dir = save_dir
        self.seed = seed
        self.config_name = config_name
        self.tokeniser_name = tokeniser_name
        self.treat_transformer_as_uncased = treat_transformer_as_uncased
        self.use_cuda = use_cuda
        self.training_batch_size = training_batch_size
        self.non_training_batch_size = non_training_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.weight_decay = weight_decay
        self.adam_epsilon = adam_epsilon
        self.training_epochs = training_epochs
        self.local_rank = local_rank
        self.save_frequency = save_frequency
        self.load_file = load_file
