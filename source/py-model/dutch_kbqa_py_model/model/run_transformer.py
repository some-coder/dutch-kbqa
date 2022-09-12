"""Symbols for running BERT-based transformer fine-tuning operations.

Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Adapted by GitHub user `some-coder` on 2022-09-06.
"""

import os
import random
import tqdm
import re
from pathlib import Path, PurePosixPath
import torch
import torch.distributed as torch_distrib
from torch.optim import Optimizer
from torch.utils.data import DataLoader, \
                             SequentialSampler, \
                             RandomSampler, \
                             TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import PretrainedConfig, \
                         PreTrainedModel, \
                         PreTrainedTokenizer, \
                         BertConfig, \
                         BertModel, \
                         BertTokenizer, \
                         RobertaConfig, \
                         RobertaModel, \
                         RobertaTokenizer, \
                         XLMRobertaConfig, \
                         XLMRobertaModel, \
                         XLMRobertaTokenizer, \
                         AdamW, \
                         get_linear_schedule_with_warmup
from nltk.translate.bleu_score import corpus_bleu
from dutch_kbqa_py_model.model.transformer import Transformer
from dutch_kbqa_py_model.dataset.data_points import RawDataPoint, TransformerDataPoint, \
                                                    loaded_raw_data_points, \
                                                    transformer_data_points_from_raw
from dutch_kbqa_py_model.utilities import LOGGER, \
                                          NO_DISTRIBUTION_RANK, \
                                          MLStage, \
                                          NaturalLanguage, \
                                          QueryLanguage, \
                                          set_seeds
from typing import NamedTuple, \
                   Dict, \
                   Type, \
                   Union, \
                   Set, \
                   Optional, \
                   Tuple, \
                   List, \
                   Callable, \
                   cast
from typing_extensions import Literal, TypedDict


class ModelTriple(NamedTuple):
    """An en- and decoder language model triple.

    The triple stores three values: (1) a type of model configuration, (2) a
    type of model, and (3) a type of model tokeniser.
    """
    configuration: Type[PretrainedConfig]
    model: Type[PreTrainedModel]
    tokeniser: Type[PreTrainedTokenizer]


# A supported en- and decoder language model. A `'random'` prefix indicates
# that the model's configuration, model architecture, and tokeniser will be
# used (like it would be normally), except that the model's weights are
# initialised randomly. So the model is parameterised as if its pretraining
# weights have never been imposed.
#   Note: unless present in the name, the 'base' model is implied, and not
# the 'large' or 'xl' model variants.
SupportedModelType = Union[Literal['bert'],
                           Literal['random-bert'],
                           Literal['roberta'],
                           Literal['random-roberta'],
                           Literal['xlm-roberta'],
                           Literal['random-xlm-roberta']]
# Either an encoder ('enc') or a decoder ('dec') language model.
EncOrDec = Union[Literal['enc'], Literal['dec']]
# An encoder language model for use in the first half of a transformer.
Encoder = Union[torch.nn.TransformerEncoder, PreTrainedModel]
# A decoder language model for use in the second half of a transformer.
Decoder = Union[torch.nn.TransformerDecoder, PreTrainedModel]
# A type of PyTorch `TensorDataset` sampler used by the `TransformerRunner`.
Sampler = Union[RandomSampler, DistributedSampler, SequentialSampler]
# A learning rate scheduler for transformers.
Scheduler = torch.optim.lr_scheduler.LambdaLR


class WeightDecayParamGroup(TypedDict):
    """A PyTorch `Optimizer` parameter group that also specifies a weight decay
    scalar to apply to the parameters within the group.
    """
    params: List[torch.nn.parameter.Parameter]
    weight_decay: float


class TrainInfo(TypedDict):
    """Stores various data pertraining to training transformers."""
    steps_sum: int  # The number of batches processed up until now.
    loss_sum: float  # The cumulative categorical cross-entropy loss.


class EvaluationPair(NamedTuple):
    """A pair that combines a ground-truth evaluation query language sentence
    with a prediction for said sentence made by the transformer.

    An 'evaluation' query language sentence is a query language sentence found
    within either the validation or testing machine learning stage.
    """
    idx: int
    predicted_sent: List[str]
    ground_truth_sents: List[List[str]]


SUPPORTED_MODEL_TRIPLES: Dict[SupportedModelType, ModelTriple] = \
    {'bert': ModelTriple(BertConfig, BertModel, BertTokenizer),
     'random-bert': ModelTriple(BertConfig, BertModel, BertTokenizer),
     'roberta': ModelTriple(RobertaConfig, RobertaModel, RobertaTokenizer),
     'random-roberta': ModelTriple(RobertaConfig, RobertaModel, RobertaTokenizer),
     'xlm-roberta': ModelTriple(XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer),
     'random-xlm-roberta': ModelTriple(XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer)}


class TransformerRunner:
    """A convenience class that helps you run transformer models."""

    # The number of layers to use for en- and decode language models that are
    # initialised randomly instead of via pretrained weights files.
    NUM_RANDOM_LAYERS = 6
    # The maximum number of data points to include when performing a sampling
    # operation while reading raw data points. Inclusive.
    MAX_SAMPLES = int(1e3)
    # A fraction of the number of training stage steps. The portion of these
    # steps in which a linear optimizer learning rate warmup is applied.
    WARMUP_STEPS_FRACTION = .1
    # The worst possible BLEU score. Used as a starting value for tracking
    # improvements to the transformer's BLEU score over epochs.
    WORST_BLEU_SCORE = 0
    # A subdirectory in `save_dir` that stores transformer parameters that
    # yielded the best BLEU score up until some point during training.
    BEST_BLEU_CKPT_DIR = Path('best-bleu-checkpoint')
    # A separator string to use for separating ground-truth query language
    # sentences within evaluation pair files.
    GROUND_TRUTH_SENTS_SEP = '   ;   '
    # The name of the file in which transformer states are saved.
    SAVE_FILE_NAME = 'pytorch-model'

    def __init__(self,
                 enc_model_type: SupportedModelType,
                 dec_model_type: SupportedModelType,
                 enc_id_or_path: Union[str, PurePosixPath],
                 dec_id_or_path: Union[str, PurePosixPath],
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
                 enc_config_name: Optional[str],
                 dec_config_name: Optional[str],
                 enc_tokeniser_name: Optional[str],
                 dec_tokeniser_name: Optional[str],
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
                 load_file: Optional[Path],
                 follow_spbert_seed_protocol: bool) -> None:
        """Constructs a new transformer runner.
        
        :param enc_model_type: The encoder language model type.
        :param dec_model_type: The decoder language model type.
        :param enc_id_or_path: A file system path to a pre-trained encoder
            language model (enclosing folder or configuration JSON file), or a
            model ID of a model hosted on `huggingface.co`.
        :param dec_id_or_path: A file system path to a pre-trained decoder
            language model (enclosing folder or configuration JSON file), or a
            model ID of a model hosted on `huggingface.co`.
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
        :param perform_training: Whether to perform the training stage.
        :param perform_validation: Whether to perform the validation stage.
        :param perform_testing: Whether to perform the testing stage.
        :param save_dir: A file system path to a directory. The directory under
            which to save transformer checkpoints and model predictions.
        :param seed: A pseudo-random number generator (PRNG) initialisation
            value to use. (This argument is required to encourage
            reproducibility in model results. Take care to switch seeds if it
            is your intention to obtain varying results.) Must be an integer in
            the range [1, 2^32 - 1], both ends inclusive.
        :param enc_config_name: An encoder language model configuration if you
            don't wish to use the default one associated with `enc_model_type`.
        :param dec_config_name: A decoder language model configuration if you
            don't wish to use the default one associated with `dec_model_type`.
        :param enc_tokeniser_name: An encoder language model tokeniser if you
            don't wish to use the default one associated with `enc_model_type`.
        :param dec_tokeniser_name: A decoder language model tokeniser if you
            don't wish to use the default one associated with `dec_model_type`.
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
            the special value `NO_DISTRIBUTION_RANK` if you wish not to use
            distributed execution.
        :param save_frequency: The number of epochs to complete before
            performing a(nother) save to disk. Must be strictly positive.
        :param load_file: (Only required when `perform_testing` is `True`.) A
            file system path to a `.bin` file. The path to a trained
            transformer.
        :param follow_spbert_seed_protocol: Whether to follow the same PRNG
            seed protocol as is done in Tran et al. (2021)'s SPBERT paper.
        """
        self.enc_model_type = enc_model_type
        self.dec_model_type = dec_model_type
        self.enc_id_or_path = enc_id_or_path
        self.dec_id_or_path = dec_id_or_path
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
        self.enc_config_name = enc_config_name
        self.dec_config_name = dec_config_name
        self.enc_tokeniser_name = enc_tokeniser_name
        self.dec_tokeniser_name = dec_tokeniser_name
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
        self.follow_spbert_seed_protocol = follow_spbert_seed_protocol

        self.log_arguments()

        device, number_gpus = self.device_and_number_of_gpus_to_use()
        self.device: torch.device = device
        self.number_gpus: int = number_gpus
        self.log_device_and_number_of_gpus_in_use()

        set_seeds(seed=self.seed,
                  follow_spbert_seed_protocol=self.follow_spbert_seed_protocol)
        self.ensure_save_dir_exists()

        self.trf: Transformer
        self.enc_tokeniser: PreTrainedTokenizer
        self.dec_tokeniser: PreTrainedTokenizer
        self.trf, self.enc_tokeniser, self.dec_tokeniser = \
            self.initialised_transformer_and_tokenisers()
        self.prepare_transformer_for_distributed_training()

    def log_arguments(self) -> None:
        """Logs this transformer runner's initialisation arguments."""
        msg = 'Arguments passed to transformer runner:\n'
        sub_msg_fmt = '\t%28s: %s'
        members_dict = self.__dict__.items()
        for idx, (member, value) in enumerate(members_dict):
            if member == 'local_rank' and value == NO_DISTRIBUTION_RANK:
                str_val = '(No rank)'
            else:
                str_val = f'\'{value}\'' if type(value) == str else f'{value}'
            msg += sub_msg_fmt % (member, str_val)
            msg += ',\n' if idx < len(members_dict) - 1 else '.'
        LOGGER.info(msg)

    def device_and_number_of_gpus_to_use(self) -> Tuple[torch.device, int]:
        """Returns the PyTorch device to use, as well as the number of GPUs in
        use, depending on the initialisation arguments.

        :returns: A pair. First, the PyTorch device to use for this Python
            process. Second, the number of GPUs in use by this Python process:
            zero or more if not distributed, and zero or one if distributed.
        """
        device: torch.device
        number_gpus: int
        if self.local_rank != NO_DISTRIBUTION_RANK and self.use_cuda:
            # Use distributed training. Entails the use of GPUs.
            torch.cuda.set_device(self.local_rank)
            device = torch.device('cuda', index=self.local_rank)
            torch_distrib.init_process_group(backend=torch_distrib.Backend.NCCL)
            number_gpus = 1
        else:
            # Use local training. Uses the GPU depending on `self.use_cuda`.
            device = torch.device('cuda') \
                     if torch.cuda.is_available() and self.use_cuda else \
                     torch.device('cpu')
            number_gpus = torch.cuda.device_count()
        return device, number_gpus

    def log_device_and_number_of_gpus_in_use(self) -> None:
        """Logs this transformer runner Python process' PyTorch device and
        number of GPUs used.
        """
        msg = '\n'
        sub_msg_fmt = '\t%35s: %s'
        sub_messages: List[Tuple[str, Union[bool, int, torch.device]]] = \
            [('Distributed training', self.local_rank != NO_DISTRIBUTION_RANK),
             ('Process rank', self.local_rank),
             ('PyTorch device', self.device),
             ('Number of GPUs used by this process', self.number_gpus)]
        for index, (label, value) in enumerate(sub_messages):
            if label == 'Process rank' and value == NO_DISTRIBUTION_RANK:
                msg += sub_msg_fmt % (label, '(No rank)')
            else:
                msg += sub_msg_fmt % (label, str(value))
            msg += ',\n' if index != len(sub_messages) - 1 else '.'
        LOGGER.info(msg)

    def ensure_save_dir_exists(self) -> None:
        """Makes sure that `self.save_dir` exists: if it does not yet exist, it
        will be created.
        
        :throws: `OSError` if a file system-related problem occurs.
        """
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def is_random_model_type(self, model_type: SupportedModelType) -> bool:
        """Determines whether the supplied `model_type` is one of which the
        model weights should be initialised randomly, instead of imposing them
        from a pretrained version of the model.

        :param model_type: The en- or decoder model type. Note: it must be the
            string, not a `Type[PreTrainedModel]`!
        :returns: The question's answer.
        """
        return model_type.startswith('random-')

    def instantiated_configs(self) -> Tuple[PretrainedConfig, PretrainedConfig]:
        """Returns instantiated transformer en- and decoder configurations.
        
        :returns: A pair. The en- and decoder configurations, respectively.
        """
        configs: Tuple[PretrainedConfig, ...] = tuple()
        prefix: EncOrDec
        for prefix in ('enc', 'dec'):
            model_type_str: SupportedModelType = \
                getattr(self, f'{prefix}_model_type')
            id_or_path: Union[str, PurePosixPath] = \
                getattr(self, f'{prefix}_id_or_path')
            config_name: Optional[str] = \
                getattr(self, f'{prefix}_config_name')
            cfg_cls, _, _ = SUPPORTED_MODEL_TRIPLES[model_type_str]
            config = cfg_cls.from_pretrained(id_or_path
                                             if config_name is None else
                                             config_name)
            if prefix == 'dec':
                # Add additional details when considering the decoder.
                config.is_decoder = True
                config.add_cross_attention = True
            if self.is_random_model_type(model_type_str):
                assert(hasattr(config, 'hidden_size'))
                assert(type(config.hidden_size) == int)
                assert(hasattr(config, 'num_attention_heads'))
                assert(type(config.num_attention_heads) == int)
            configs += config,
        assert(len(configs) == 2)
        return cast(Tuple[PretrainedConfig, PretrainedConfig],
                    tuple(configs))
    
    def instantiated_en_or_decoder(self,
                                   config: PretrainedConfig,
                                   enc_or_dec: EncOrDec) -> \
            Union[Encoder, Decoder]:
        """Returns an instantiated en- or decoder language model.
        
        :param config: The en- or decoder configuration from which to
            initialise the model.
        :param enc_or_dec: Whether to initialise an encoder or a decoder.
        :returns: The instantiated en- or decoder.
        """
        model_type_str: SupportedModelType = \
            getattr(self, f'{enc_or_dec}_model_type')
        if self.is_random_model_type(model_type_str):
            layer_cls = torch.nn.EncoderLayer \
                        if enc_or_dec == 'enc' else \
                        torch.nn.DecoderLayer
            coder_cls = torch.nn.TransformerEncoder \
                        if enc_or_dec == 'enc' else \
                        torch.nn.TransformerDecoder
            layer = layer_cls(d_model=config.hidden_size,
                              nhead=config.num_attention_heads)
            return coder_cls(layer,
                             num_layers=TransformerRunner.NUM_RANDOM_LAYERS)
        else:
            _, mdl_cls, _ = SUPPORTED_MODEL_TRIPLES[model_type_str]
            id_or_path: Union[str, PurePosixPath] = \
                getattr(self, f'{enc_or_dec}_id_or_path')
            return mdl_cls.from_pretrained(id_or_path, config=config)

    def instantiated_tokeniser(self,
                               enc_or_dec: EncOrDec) -> PreTrainedTokenizer:
        """Returns an instantiated tokeniser.
        
        :param enc_or_dec: Whether the tokeniser is meant for tokenisation at
            the en- or decoding side (first and second half of the transformer,
            respectively).
        :returns: The tokeniser.
        """
        model_type_str: SupportedModelType = \
            getattr(self, f'{enc_or_dec}_model_type')
        _, _, tok_cls = SUPPORTED_MODEL_TRIPLES[model_type_str]
        id_or_path: Union[str, PurePosixPath] = \
            getattr(self, f'{enc_or_dec}_id_or_path')
        tok_name: Optional[str] = \
            getattr(self, f'{enc_or_dec}_tokeniser_name')
        name = id_or_path if tok_name is None else tok_name
        return tok_cls.from_pretrained(name,
                                       do_lower_case=self.treat_transformer_as_uncased)

    def initialised_transformer_and_tokenisers(self) -> \
            Tuple[Transformer, PreTrainedTokenizer, PreTrainedTokenizer]:
        """Initialises the requested transformer model and the en- and decoder
        tokenisers.

        'Initialising a model' here means: (1) instantiating the `Transformer`
        class, (2) loading existing transformer parameters if needed, and (3)
        sending the model to the right PyTorch device. Importantly,
        instantiation does not involve preparing the transformer for multi-GPU
        or distributed training; that must be done in a separate function call.

        :returns: A triple. First, the initialised transformer model. Second,
            the encoder tokeniser. Third and last, the decoder tokeniser.
        """
        enc_config, dec_config = self.instantiated_configs()
        enc_msg = 'Initialising encoder model. ' + \
                  'Note that there may be warnings of `cls.predictions` ' + \
                  'weights not being set; this is normal.'
        dec_msg = 'Initialising decoder model. ' + \
                  'Note that there may be warnings of (1) `cls.predictions` ' + \
                  'and (2) `crossattention` weights not being set; this ' + \
                  'is normal.'
        LOGGER.info(enc_msg)
        encoder = self.instantiated_en_or_decoder(enc_config, enc_or_dec='enc')
        LOGGER.info('Successfully initialised encoder.')
        LOGGER.info(dec_msg)
        decoder = self.instantiated_en_or_decoder(dec_config, enc_or_dec='dec')
        LOGGER.info('Successfully initialised decoder.')
        enc_tokeniser = self.instantiated_tokeniser(enc_or_dec='enc')
        dec_tokeniser = self.instantiated_tokeniser(enc_or_dec='dec')
        trf = Transformer(encoder,
                          decoder,
                          enc_config,
                          dec_config,
                          tie_weights=self.enc_id_or_path == self.dec_id_or_path,
                          beam_size=self.beam_size,
                          max_length=self.max_query_language_length,
                          sos_id=dec_tokeniser.cls_token_id,
                          eos_id=dec_tokeniser.sep_token_id)
        if self.load_file is None:
            LOGGER.info('No transformer file to load: starting fine-tuning.')
        else:
            msg_before = 'Reloading transformer from location \'%s\'.'
            LOGGER.info(msg_before % (str(self.load_file),))
            trf.load_state_dict(torch.load(self.load_file))
            LOGGER.info('Transformer reloading successful.')
        trf.to(self.device)
        return trf, enc_tokeniser, dec_tokeniser

    def prepare_transformer_for_distributed_training(self) -> None:
        """Sets up this transformer runner's transformer for either operating
        distributed over multiple processes, or over multiple GPUs within a
        single process.

        This method silently no-operates if, during instantiation of the
        transformer runner, no request was made to perform distributed
        operations over both processes and GPUs.

        Note that distribution over multiple processes and distribution over
        multiple GPUs are mutually exclusive. If both are requested,
        distribution over processes attains precedence.
        """
        # We force a type cast in the two returns of this method, because the
        # design of both `apex.parallel.DistributedDataParallel` and
        # `torch.nn.DataParallel` intend to keep using the class as before,
        # similar to how you would wrap OpenAI Gym `Env`s.
        if self.local_rank != NO_DISTRIBUTION_RANK:
            # Distribute over multiple processes.
            try:
                from apex.parallel import DistributedDataParallel
            except ImportError:
                msg = 'You need to have NVIDIA Apex installed in order to ' + \
                      'run the transformer over multiple processes, as you ' + \
                      'requested. For installation instructions, see ' + \
                      'https://github.com/NVIDIA/apex#from-source.'
                raise ImportError(msg)
            self.trf = cast(Transformer,
                            DistributedDataParallel(module=self.trf))
        elif self.number_gpus > 1:
            # Distribute over multiple GPUs.
            self.trf = cast(Transformer,
                            torch.nn.DataParallel(module=self.trf))

    def data_points_location(self,
                             ml_stage: MLStage,
                             language: Union[NaturalLanguage, QueryLanguage]) -> \
            Path:
        """Returns the location of the requested data points.
        
        :param ml_stage: The machine learning stage for which the data points
            are meant.
        :param language: Either a natural or a query language. The language of
            the data points.
        :returns: An absolute file system location to the data points.
        :throws: `FileNotFoundError` if the data points are not present on the
            file system; `RuntimeError` if path resolution leads to an infinite
            recursion. (The latter may happen if two symlinks point to one
            another, for instance.)
        """
        return (self.dataset_dir /  
                f'{ml_stage.value}-{language.value}.txt').resolve(strict=True)

    def raw_data_points_for_ml_stage(self,
                                     ml_stage: MLStage,
                                     perform_sampling: bool = False) -> \
            List[RawDataPoint]: 
        """Returns unprocessed ('raw') data points for the requested stage of
        machine learning.
        
        :param ml_stage: The machine learning stage to get 'raw' data points
            for.
        :param perform_sampling: Whether to draw a random sample from the
            complete set of raw data points. Note that the sample's size is
            capped to a maximum of `MAX_SAMPLES` data points, so if the file
            you requested has more data points, some may not appear in the
            sample. Defaults to `False`.
        :returns: 'Raw' data points for the requested ML stage.
        """
        natural_language_loc = self.data_points_location(ml_stage,
                                                         self.natural_language)
        query_language_loc = self.data_points_location(ml_stage,
                                                       self.query_language)
        data_points = loaded_raw_data_points(natural_language_loc,
                                             query_language_loc)
        if perform_sampling:
            sample_size = min(TransformerRunner.MAX_SAMPLES,
                              len(data_points))
            data_points = random.sample(data_points, k=sample_size)
        return data_points

    def transformer_data_points_for_ml_stage(self,
                                             raw_data_points: List[RawDataPoint],
                                             ml_stage: MLStage) -> \
            List[TransformerDataPoint]:
        """Returns transformer-ready data points for the requested stage of
        machine learning, derived from unprocessed ('raw') data points.
        
        :param ml_stage: The machine learning stage to return data points for.
        :param raw_data_points: The 'raw' data points: those just read in from
            a text file without any post-processing whatsoever.
        :returns: Transformer-ready data points for the requested ML stage.
        """
        return transformer_data_points_from_raw(raw_data_points,
                                                self.enc_tokeniser,
                                                self.dec_tokeniser,
                                                self.max_natural_language_length,
                                                self.max_query_language_length,
                                                ml_stage)

    def tensor_dataset_for_ml_stage(self,
                                    trf_dps: List[TransformerDataPoint],
                                    ml_stage: MLStage) -> TensorDataset:
        """Returns a PyTorch tensor version of the requested dataset split.
        
        :param trf_dps: Transformer-ready data points for the requested data
            split. Are not yet encoded in PyTorch tensors.
        :param ml_stage: The machine learning stage to get a tensor dataset
            for. Should match the dataset split from which `trf_dps` were
            obtained.
        :returns: The tensor dataset.
        """
        attr_prefixes = ('inp',)
        if ml_stage == MLStage.TRAIN:
            attr_prefixes += 'out',
        tensors: List[torch.Tensor] = []
        for attr_prefix in attr_prefixes:
            for attr_suffix in ('ids', 'att_mask'):
                attr = f'{attr_prefix}_{attr_suffix}'
                ids_or_att_masks: List[List[int]] = [getattr(dp, attr) 
                                                     for dp in trf_dps]
                tensors.append(torch.tensor(ids_or_att_masks,
                                            dtype=torch.long))
        return TensorDataset(*tensors)

    def sampler_for_ml_stage(self,
                             tensor_ds: TensorDataset,
                             ml_stage: MLStage) -> Sampler:
        """Returns a data point sampler for the specified machine learning
        stage.
        
        :param tensor_ds: The tensor dataset from which the sampler should
            obtain data points.
        :param ml_stage: The machine learning stage to get a sampler for.
        :returns: The data point sampler.
        """
        if ml_stage == MLStage.TRAIN:
            return RandomSampler(tensor_ds) \
                   if self.local_rank == NO_DISTRIBUTION_RANK else \
                   DistributedSampler(tensor_ds)
        else:
            return SequentialSampler(tensor_ds)
    
    def batch_size_for_ml_stage(self, ml_stage: MLStage) -> int:
        """Returns the batch size to use for the given machine learning stage.
        
        :param ml_stage: The machine learning stage to get a batch size for.
        :returns: The batch size.
        """
        return self.training_batch_size // self.gradient_accumulation_steps \
               if ml_stage == MLStage.TRAIN else \
               self.non_training_batch_size
    
    def data_loader_for_ml_stage(self,
                                 ml_stage: MLStage,
                                 raw_dps:
                                     Optional[List[RawDataPoint]] = None) -> \
            Tuple[DataLoader, int]:
        """Returns a data loader for the requested stage of machine learning.
        
        :param ml_stage: The machine learning stage to get a data loader for.
        :param raw_dps: (Only required when `ml_stage` is set to
            `MLStage.VALIDATE` or `MLStage.TEST`.) Already-computed raw data
            points to use as input to the data loader.
        :returns: A pair. First, the data loader. Second, the number of raw
            data points that went into the data loader.
        :throws: `AssertionError` if `raw_dps` is not given while `ml_stage`
            is either `MLStage.VALIDATE` or `MLStage.TEST`.
        """
        if ml_stage in (MLStage.VALIDATE, MLStage.TEST):
            assert(raw_dps is not None)
        else:
            raw_dps = self.raw_data_points_for_ml_stage(ml_stage)
        trf_dps = self.transformer_data_points_for_ml_stage(raw_dps, ml_stage)
        tensor_ds = self.tensor_dataset_for_ml_stage(trf_dps, ml_stage)
        sampler = self.sampler_for_ml_stage(tensor_ds, ml_stage)
        batch_size = self.batch_size_for_ml_stage(ml_stage)
        return DataLoader(tensor_ds, batch_size, sampler=sampler), len(raw_dps)

    def optimiser_for_training_stage(self) -> Optimizer:
        """Returns an optimiser for the transformer's training stage.
        
        :returns: An optimiser.
        """
        is_decayable_param: Callable[[str], bool] = \
            lambda param_name: not any((kw in param_name) for kw in
                                       ['bias', 'LayerNorm.weight'])
        param_groups: List[WeightDecayParamGroup] = \
            [{'params': [param for name, param in self.trf.named_parameters()
                         if is_decayable_param(name)],
              'weight_decay': self.weight_decay},
             {'params': [param for name, param in self.trf.named_parameters()
                         if not is_decayable_param(name)],
              'weight_decay': 0.}]
        return AdamW(param_groups, lr=self.learning_rate, eps=self.adam_epsilon)

    def scheduler_for_training_stage(self,
                                     optimiser: Optimizer,
                                     train_dl: DataLoader) -> Scheduler:
        """Returns a learning rate scheduler for the transformer's training
        stage.
        
        :param optimiser: The optimiser to apply the adapting learning rates to.
        :param train_dl: The training stage data loader.
        :returns: A learning rate scheduler.
        """
        total_steps = (len(train_dl) // self.gradient_accumulation_steps) * \
                      self.training_epochs
        number_warmup_steps = int(total_steps *
                                  TransformerRunner.WARMUP_STEPS_FRACTION)
        return get_linear_schedule_with_warmup(optimiser,
                                               number_warmup_steps,
                                               num_training_steps=total_steps)

    def log_start_of_training(self, number_data_points: int) -> None:
        """Logs relevant parameters that affect upcoming training stages.
        
        :param number_data_points: The number of data points present in the
            training dataset split.
        """
        msg = 'Performing training stage.\n'
        sub_msg_fmt = '\t%21s: %d'
        sub_messages: List[Tuple[str, int]] = \
            [('Number of data points', number_data_points),
             ('Batch size', self.batch_size_for_ml_stage(MLStage.TRAIN)),
             ('Number of epochs', self.training_epochs)]
        for idx, (label, value) in enumerate(sub_messages):
            msg += sub_msg_fmt % (label, value)
            msg += ',\n' if idx != len(sub_messages) - 1 else '.'
        LOGGER.info(msg)

    def run_single_training_epoch(self,
                                  epoch: int,
                                  dl: DataLoader,
                                  optimiser: Optimizer,
                                  scheduler: Scheduler) -> TrainInfo:
        """Runs the transformer through a single training epoch.
        
        :param epoch: The current training epoch.
        :param dl: The training stage data loader.
        :param optimiser: An optimiser that revises transformer weights.
        :param scheduler: A learning rate scheduler for the optimiser.
        :returns: Training information on the completed training epoch.
        """
        self.trf.train()
        result: TrainInfo = {'steps_sum': 0, 'loss_sum': 0.}
        progress_bar = tqdm.tqdm(dl, total=len(dl))
        batch: List[torch.Tensor]
        for batch in progress_bar:
            # Compute losses per each batch in the data loader.
            batch = tuple(t.to(self.device) for t in batch)
            inp_ids, inp_att_mask, out_ids, out_att_mask = batch
            ce_loss: torch.Tensor
            ce_loss, _, _ = self.trf(inp_ids,
                                     inp_att_mask,
                                     out_ids,
                                     out_att_mask)
            if self.number_gpus > 1:
                ce_loss = ce_loss.mean()
            if self.gradient_accumulation_steps > 1:
                ce_loss /= self.gradient_accumulation_steps
            result['loss_sum'] += ce_loss.item()
            running_loss = round((result['loss_sum'] *
                                  self.gradient_accumulation_steps) /
                                 (result['steps_sum'] + 1),
                                 ndigits=4)
            dsc_fmt = 'Epoch %3d, running cross-entropy loss %7.4lf.'
            progress_bar.set_description(dsc_fmt % (epoch, running_loss))
            result['steps_sum'] += 1
            ce_loss.backward()
            if (result['steps_sum'] + 1) % self.gradient_accumulation_steps == 0:
                # Update the transformer's parameters.
                optimiser.step()
                optimiser.zero_grad()
                scheduler.step() 
        return result
    
    def predicted_query_language_sentences(self,
                                           dl: DataLoader,
                                           ml_stage: MLStage) -> List[str]:
        """Returns the transformer's predicted query language sentences for
        each of the data points in `dl`.

        This method only supports prediction during the validation and testing
        stages; providing an `MLStage.TRAIN` `dl` is an invalid operation.

        :param dl: The data loader.
        :param ml_stage: The machine learning stage that `dl` samples data
            points from.
        :returns: Transformer predictions for each of these data points.
        :throws: `AssertionError` if `dl` is meant for `MLStage.Train`.
        """
        assert(ml_stage in (MLStage.VALIDATE, MLStage.TEST))
        sents: List[str] = []
        progress_bar = tqdm.tqdm(dl, total=len(dl))
        batch: List[torch.Tensor]
        for batch in progress_bar:
            dsc = f'Predicting in ML stage \'{ml_stage.value.title()}\''
            progress_bar.set_description(dsc)
            batch = tuple(t.to(self.device) for t in batch)
            inp_ids, inp_att_mask = batch
            with torch.no_grad():
                predictions: torch.Tensor = self.trf(inp_ids, inp_att_mask)
                for prediction in predictions:
                    tkn_ids = list(prediction[0, :].cpu().numpy())
                    if 0 in tkn_ids:
                        # Remove any zero-padding tokens to the right.
                        tkn_ids = tkn_ids[:tkn_ids.index(0)]
                    sent = self.dec_tokeniser.decode(tkn_ids,
                                                     clean_up_tokenization_spaces=False)
                    sents.append(sent)
        return sents 

    def evaluation_pairs(self,
                         predicted_sents: List[str],
                         ground_truth_raw_dps: List[RawDataPoint]) -> \
            List[EvaluationPair]:
        """Returns evaluation pairs derived from combining predicted
        sentences and 'raw' ground-truth data points in a one-on-one fashion.
        
        The pairs are not simply combined from the two source lists; some
        spacing is added and removed in places where this is needed.

        :param predicted_sents: Predicted sentences made by the transformer
            for each of the entries in `ground_truth_raw_dps`.
        :param ground_truth_raw_dps: 'Raw' ground-truth validation data points.
        :returns: The evaluation pairs.
        """
        out: List[EvaluationPair] = []
        prd: str
        gt: RawDataPoint
        assert(len(predicted_sents) == len(ground_truth_raw_dps))
        for prd, gt in zip(predicted_sents, ground_truth_raw_dps):
            prd = prd.strip().replace('< ', '<').replace(' >', '>')
            prd = re.sub(r' ?([!"#$%&\'(’)*+,-./:;=?@\\^_`{|}~]) ?',
                         r'\1',
                         prd)
            prd = prd.replace('attr_close>', 'attr_close >')
            prd = prd.replace('_attr_open', '_ attr_open')
            prd = prd.replace(' [ ', ' [').replace(' ] ', '] ')
            prd = prd.replace('_obd_', ' _obd_ ')
            prd = prd.replace('_oba_', ' _oba_ ')
            ground_truth_sent = gt.query_language.strip().split()
            out.append(EvaluationPair(idx=gt.idx,
                                      predicted_sent=prd.split(),
                                      ground_truth_sents=[ground_truth_sent]))
        return out

    def save_evaluation_pairs(self,
                              e_pairs: List[EvaluationPair],
                              ml_stage: MLStage) -> None:
        """Saves evaluation pairs of the specified machine learning stage to
        disk.

        By definition, evaluation pairs can only belong to `MLStage.VALIDATE`
        or `MLStage.TEST`.
        
        :param e_pairs: The evaluation pairs to save.
        :param ml_stage: The machine learning stage to which the evaluation
            pairs belong.
        :throws: `OSError` if one or both of the validation pair files cannot
            be opened (e.g. because of insufficient file system permissions),
            and `AssertionError` if `ml_stage` is not either `MLStage.VALIDATE`
            or `MLStage.TEST`.
        """
        assert(ml_stage in (MLStage.VALIDATE, MLStage.TEST))
        prefix = 'validate' if ml_stage == MLStage.VALIDATE else 'test'
        with open(self.save_dir / f'{prefix}-predicted.txt', 'w') as f_prd, \
             open(self.save_dir / f'{prefix}-ground-truth.txt', 'w') as f_gt:
            for pair in e_pairs:
                gt_sep = TransformerRunner.GROUND_TRUTH_SENTS_SEP
                predicted = ' '.join(pair.predicted_sent)
                joined_gt_sents: List[str] = \
                    [' '.join(sent) for sent in pair.ground_truth_sents]
                ground_truth = gt_sep.join(joined_gt_sents)
                f_prd.write(f'{pair.idx}\t{predicted}\n')
                f_gt.write(f'{pair.idx}\t{ground_truth}\n')

    def transformer_evaluation_bleu_score(self,
                                          e_pairs: List[EvaluationPair]) -> float:
        """Returns the transformer's BLEU score on the evaluation set, given
        already-computed evaluation pairs of said set.
        
        :param e_pairs: The evaluation pairs of the evaluation set.
        :returns: The transformer's BLEU score. A value between `0.` and
            `100.`, both ends inclusive. (The closer to `100.`, generally the
            better.)
        """
        references = [pair.ground_truth_sents for pair in e_pairs]
        hypotheses = [pair.predicted_sent for pair in e_pairs]
        return cast(float, corpus_bleu(references, hypotheses)) * 100.

    def log_transformer_evaluation_bleu_score(self,
                                              bleu_score: float,
                                              ml_stage: MLStage) -> None:
        """Logs the BLEU score the transformer obtained on the evaluation set.
        
        :param bleu_score: The transformer's BLEU score.
        :param ml_stage: The machine learning stage that the evaluation BLEU
            score belongs to. By definition, it must either be
            `MLStage.VALIDATE` or `MLStage.TEST`.
        :throws: `AssertionError` if `ml_stage` is not one of
            `MLStage.VALIDATE` and `MLStage.TEST`.
        """
        assert(ml_stage in (MLStage.VALIDATE, MLStage.TEST))
        msg = 'Machine learning stage \'%s\' BLEU score: %8.4lf.'
        LOGGER.info(msg % (ml_stage.value.title(),
                           round(bleu_score, ndigits=4),))

    def log_validation_bleu_score_update(self, old: float, new: float) -> None:
        """Logs how the validation BLEU score of the transformer updated from
        the previous epoch to the current one.

        The current implementation only reports 'improvement updates': changes
        in BLEU score where `new` is strictly greater than `old`.
        
        :param old: The previous epoch's best BLEU score.
        :param new: The current epoch's best BLEU score.
        """
        if new > old:
            msg = 'Validation BLEU score improved!\n'
            sub_msg_fmt = '\t%5s: %8.4lf'
            sub_messages: List[Tuple[str, float]] = \
                [('Old', old),
                 ('New', new),
                 ('Delta', new - old)]
            for index, (label, value) in enumerate(sub_messages):
                msg += sub_msg_fmt % (label, value)
                msg += ',\n' if index != len(sub_messages) - 1 else '.'
            LOGGER.info(msg)

    def save_transformer(self) -> None:
        """Saves the transformer to disk.
        
        In practice, we save the state (all `torch.nn.parameter.Parameter`s)
        to disk, and require the user to re-specify the exact same architecture
        at the command-line the next time the transformer needs to be loaded.
        """
        best_ckpt_dir = \
            (self.save_dir / \
             TransformerRunner.BEST_BLEU_CKPT_DIR).resolve()
        if not os.path.exists(best_ckpt_dir):
            os.makedirs(best_ckpt_dir)
        trf_to_save: torch.nn.Module = self.trf.module \
                                       if hasattr(self.trf, 'module') else \
                                       self.trf
        torch.save(trf_to_save.state_dict(),
                   best_ckpt_dir / f'{TransformerRunner.SAVE_FILE_NAME}.zip')

    def run_single_evaluation_epoch(self,
                                    raw_dps: List[RawDataPoint],
                                    ml_stage: MLStage) -> float:
        """Runs the transformer through a single evaluation epoch.

        :param raw_dps: The 'raw' evaluation data points. Should remain
            constant when calling this method multiple times (during the
            validation stage).
        :param ml_stage: The machine learning stage to which this evaluation
            stage belongs. By definition, `ml_stage` may be either
            `MLStage.VALIDATE` or `MLStage.TEST`; other stages are invalid.
        :returns: The BLEU score obtained by the transformer in its current
            state, that is: in the current epoch, with its current weights
            parameterisation.
        :throws: `AssertionError` if `ml_stage` is not either
            `MLStage.VALIDATE` or `MLStage.TEST`.
        """
        assert(ml_stage in (MLStage.VALIDATE, MLStage.TEST))
        dl, _ = self.data_loader_for_ml_stage(ml_stage, raw_dps)
        self.trf.eval()
        prd_sents = self.predicted_query_language_sentences(dl, ml_stage)
        self.trf.train()
        e_pairs = self.evaluation_pairs(predicted_sents=prd_sents,
                                        ground_truth_raw_dps=raw_dps)
        self.save_evaluation_pairs(e_pairs, ml_stage)
        bleu_score = self.transformer_evaluation_bleu_score(e_pairs)
        self.log_transformer_evaluation_bleu_score(bleu_score, ml_stage)
        return bleu_score

    def ensure_training_stage_begins_with_empty_save_dir(self) -> None:
        """Ensures that, when starting a training stage, the saving directory
        `save_dir` does not have any contents. Specifically, it should not
        contain any evaluation pair text files, nor any transformer
        checkpoints.
        
        This method does not clear the directory for you; this is on purpose.
        You should decide what to do with the contents of the directory if it
        is not empty. 

        :throws: `RuntimeError` when `save_dir` has contents.
        """
        if os.path.exists(self.save_dir) and \
           len(os.listdir(self.save_dir)) != 0:
            LOGGER.critical('You want to train a transformer under ' +
                            f'\'{self.save_dir}\', but this directory is not ' +
                            'empty! Either (1) rename or move the directory, ' +
                            '(2) move the contents of the directory ' +
                            'somewhere else, or (3) delete the directory.')
            raise RuntimeError('Encountered a non-empty training stage saving ' +
                               'directory! See the logger message above for ' +
                               'instructions on how to resolve this issue.')

    def run_training_stage(self) -> None:
        """Runs the transformer through a complete training stage.
        
        A training stage consists of training epochs, possibly interleaved
        with validation epochs (if `perform_validation` was set to `True`);
        `training_epochs` is the number of epochs run.
        """
        self.ensure_training_stage_begins_with_empty_save_dir()
        dl, number_train = self.data_loader_for_ml_stage(MLStage.TRAIN)
        optimiser = self.optimiser_for_training_stage()
        scheduler = self.scheduler_for_training_stage(optimiser, train_dl=dl)
        self.log_start_of_training(number_data_points=number_train)
        train_info: TrainInfo = {'steps_sum': 0, 'loss_sum': 0.}
        best_bleu = TransformerRunner.WORST_BLEU_SCORE
        validation_raw_dps: Optional[List[RawDataPoint]] = None  # filled later
        for epoch in range(self.training_epochs):
            epoch_info = self.run_single_training_epoch(epoch,
                                                        dl,
                                                        optimiser,
                                                        scheduler)
            train_info['steps_sum'] += epoch_info['steps_sum']
            train_info['loss_sum'] += epoch_info['loss_sum']
            if self.perform_validation and (epoch + 1) % self.save_frequency == 0:
                train_info['steps_sum'] = 0
                train_info['loss_sum'] = 0.  # TODO(Niels): Move outside of `if`?
                if validation_raw_dps is None:
                    validation_raw_dps = \
                        self.raw_data_points_for_ml_stage(MLStage.VALIDATE,
                                                          perform_sampling=True)
                bleu_score = self.run_single_evaluation_epoch(validation_raw_dps,
                                                              MLStage.VALIDATE)
                if  bleu_score > best_bleu:
                    self.log_validation_bleu_score_update(old=best_bleu, new=bleu_score)
                    best_bleu = bleu_score
                    self.save_transformer()

    def run_testing_stage(self) -> None:
        """Runs the transformer through a complete testing stage.
        
        A testing stage consists of a single testing epoch, potentially
        preceded by a single validation epoch (if `perform_validation` was set
        to `True`). In the testing epoch, the transformer translates
        never-before-seen natural language sentences into predicted query
        language sentences, which are subsequently compared to ground-truth
        test query language sentences.
        """
        if self.perform_validation:
            validation_raw_dps = self.raw_data_points_for_ml_stage(MLStage.VALIDATE)
            _ = self.run_single_evaluation_epoch(validation_raw_dps,
                                                 MLStage.VALIDATE)
        test_raw_dps = self.raw_data_points_for_ml_stage(MLStage.TEST)
        _ = self.run_single_evaluation_epoch(test_raw_dps, MLStage.TEST)

    def run(self) -> None:
        """Applies the transformer to the task of transforming natural language
        input sequences into query language output sequences.
        
        This method performs training, validation, and testing, depending on
        what was requested during the initialisation of the transformer runner.
        """
        if self.perform_training:
            self.run_training_stage()
        if self.perform_testing:
            self.run_testing_stage()
