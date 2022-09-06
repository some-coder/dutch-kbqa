"""Symbols for loading in and pre-processing language model data points."""

from pathlib import Path
from transformers import PreTrainedTokenizer
from dutch_kbqa_py_model.utilities import LOGGER, \
                                          LOGGER_NUMBER_EXAMPLES, \
                                          MLStage
from typing import NamedTuple, List, Tuple, Optional, Union, Literal


class RawDataPoint(NamedTuple):
    """A single 'raw' natural language-query language data point.
    
    A data point is considered to be 'raw' when it has not yet been processed
    into a structure that is more directly useful to transformer models.
    """
    idx: int
    natural_language: str
    query_language: str


def loaded_raw_data_points(natural_language_file: Path,
                           query_language_file: Path) -> List[RawDataPoint]:
    """Returns 'raw' natural language-query language data points from
    respective text files.

    :param natural_language_file: A file system path to a text file. Should
        contain, per line, a single natural language sentence.
    :param query_language_file: A file system path to a text file. Should
        contain, per line, a single query language sentence.
    :returns: A list of 'raw' natural language-query language data points.
    :throws: `ValueError` if the number of sentences in `natural_language_file`
        and `query_language_file` are not equal.
    """
    data_points: List[RawDataPoint] = []
    with open(natural_language_file, mode='r', encoding='utf-8') as nl_handle, \
         open(query_language_file, mode='r', encoding='utf-8') as ql_handle:
        nl_lines = nl_handle.readlines()
        ql_lines = ql_handle.readlines()
        assert(len(nl_lines) == len(ql_lines))
        for idx, (question, query) in enumerate(zip(nl_lines, ql_lines)):
            data_points.append(RawDataPoint(idx=idx,
                                            natural_language=question.strip(),
                                            query_language=query.strip()))
    return data_points


class TransformerDataPoint(NamedTuple):
    """A single natural language-query language data point that is appropriate
    for training, validating, and testing a transformer model with.

    The difference between a `RawDataPoint` and a `TransformerDataPoint` is
    that the latter (1) encodes sentences as series of token IDs, and (2)
    includes attention masks.
    """
    idx: int
    inp_ids: List[int]
    inp_att_mask: List[int]
    out_ids: List[int]
    out_att_mask: List[int]


class TransformerDataPointHalf(NamedTuple):
    """Either the natural language or query language half of a
    natural language-query language data point that is appropriate for
    training, validating, and testing a transformer model with.
    """
    ids: List[int]
    att_mask: List[int]


DataPointHalf = Union[Literal['input'], Literal['output']]


def transformer_data_point_from_raw(raw_data_point: RawDataPoint,
                                    tokeniser: PreTrainedTokenizer,
                                    max_length: int,
                                    half: DataPointHalf,
                                    ml_stage: Optional[MLStage] = None) -> \
        Tuple[TransformerDataPointHalf, List[str]]:
    """Returns a transformer model-ready half of a data point from a raw data
    point. Additionally returns the sentence's tokenised counterpart.
    
    :param raw_data_point: The raw data point to process the input or output
        sentence of, depending on the value of `half`.
    :param tokeniser: A tokeniser to help in obtaining token IDs for the data
        point's sentence.
    :param max_length: The maximum length (in tokens) that the natural language
        or query language sentence may assume, including the start- and
        end-of-sentence tokens. Inclusive.
    :param half: The half of `raw_data_point` to process. `'input'` refers to
        the natural language half; `'output'` refers to the query language
        half.
    :param ml_stage: Only required when `half` equals `'output'`. The machine
        learning model stage for which `raw_data_points` are meant to be used.
    :returns: A pair of two results. First, the data point half. Second, a list
        of strings, representing the tokenised sentence.
    """
    if half == 'output':
        assert(ml_stage is not None)
    if half == 'output' and ml_stage in (MLStage.VALIDATE, MLStage.TEST):
        tokens = tokeniser.tokenize(text='None')
    else:
        text = raw_data_point.natural_language if half == 'input' else \
               raw_data_point.query_language
        tokens = tokeniser.tokenize(text)
        tokens = tokens[:(max_length - 2)]  # make room for SOS, EOS tokens
    tokens = [tokeniser.cls_token] + tokens + [tokeniser.sep_token] 
    ids = tokeniser.convert_tokens_to_ids(tokens)
    pad_length = max_length - len(ids)
    ids += [tokeniser.pad_token_id] * pad_length
    mask = ([1] * len(tokens)) + ([0] * pad_length)
    assert(len(ids) == len(mask))
    return TransformerDataPointHalf(ids=ids, att_mask=mask), tokens


def log_first_data_points(data_points: List[TransformerDataPoint],
                          tokens_pairs: List[Tuple[List[str], List[str]]],
                          number: int) -> None:
    """Logs the first `number` of data points in `data_points`.
    
    :param data_points: The transformer model-ready data points.
    :param tokens_pairs: A list of pairs. For each pair, the first entry
        contains the tokenised input (natural language) sentence, and the
        second entry contains the tokenised output (query language) sentence.
        Must correspond one-to-one with `data_points`.
    :param number: The number of data points to log. A strictly positive
        integer.
    :throws: `AssertionError` if `number` is not a strictly positive integer,
        and `AssertionError` if the lengths of `data_points` and `token_pairs`
        are not equal.
    """
    assert(number > 0)
    assert(len(data_points) == len(tokens_pairs)) 
    msg = 'First %d data points:\n' % (number,)
    sub_msg_fmt = '\t(Data point %d)\n' + \
                  ('\t\t%6s %9s: \'%s\',\n' * 5) + \
                  '\t\t%6s %9s: \'%s\'.'
    counter = 0  # for determining whether to print a newline
    for data_point, tokens_pair in zip(data_points[:number], tokens_pairs):
        str_inp_ids = [str(inp_id) for inp_id in data_point.inp_ids]
        str_out_ids = [str(out_id) for out_id in data_point.out_ids]
        str_inp_mask = [str(msk_part) for msk_part in data_point.inp_att_mask]
        str_out_mask = [str(msk_part) for msk_part in data_point.out_att_mask]
        sub_msg = sub_msg_fmt % (data_point.idx,
                                 'Input', 'tokens', ' '.join(tokens_pair[0]),
                                 '', 'token IDs', ' '.join(str_inp_ids),
                                 '', 'mask', ' '.join(str_inp_mask),
                                 'Output', 'tokens', ' '.join(tokens_pair[1]),
                                 '', 'token IDs', ' '.join(str_out_ids),
                                 '', 'mask', ' '.join(str_out_mask))
        msg += '%s%s' % (sub_msg, '\n' if counter < number - 1 else '')
        counter += 1
    LOGGER.info(msg)


def transformer_data_points_from_raw(raw_data_points: List[RawDataPoint],
                                     tokeniser: PreTrainedTokenizer,
                                     max_natural_language_length: int,
                                     max_query_language_length: int,
                                     ml_stage: MLStage) -> \
        List[TransformerDataPoint]:
    """Returns raw data points, but processed so as to be useful to transformer
    models.
    
    :param raw_data_points: The raw data points to process.
    :param tokeniser: A tokeniser to help in obtaining token IDs for the data
        point's sentences.
    :param ml_stage: The machine learning model stage for which
        `raw_data_points` are meant to be used.
    :returns: Processed, transformer model-ready data points.
    """
    data_points: List[TransformerDataPoint] = []
    tokens_pairs: List[Tuple[List[str], List[str]]] = []  # used for logging
    for raw_data_point in raw_data_points:
        inp_half, inp_tokens = \
            transformer_data_point_from_raw(raw_data_point,
                                            tokeniser,
                                            max_length=max_natural_language_length,
                                            half='input')
        out_half, out_tokens = \
            transformer_data_point_from_raw(raw_data_point,
                                            tokeniser,
                                            max_length=max_query_language_length,
                                            half='output',
                                            ml_stage=ml_stage)
        data_points.append(TransformerDataPoint(idx=raw_data_point.idx,
                                                inp_ids=inp_half.ids,
                                                inp_att_mask=inp_half.att_mask,
                                                out_ids=out_half.ids,
                                                out_att_mask=out_half.att_mask))
        tokens_pairs.append((inp_tokens, out_tokens))
    
    if ml_stage == MLStage.TRAIN:
        log_first_data_points(data_points,
                              tokens_pairs,
                              number=LOGGER_NUMBER_EXAMPLES)
    return data_points
