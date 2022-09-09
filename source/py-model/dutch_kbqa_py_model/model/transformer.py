"""Symbols for training, validating, and testing transformer models.

Copyright (c) Microsoft Corporation.
Licensed by Microsoft under the MIT license.
Adapted by GitHub user `some-coder` on 2022-09-06.
"""

import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.file_utils import ModelOutput
from dutch_kbqa_py_model.utilities import LOGGER, \
                                          TorchDevice
from dutch_kbqa_py_model.model.beam_search import TokenBeamSearcher
from typing import Optional, NamedTuple, Union, Tuple, List
from typing_extensions import Literal


class LabelSmoothingLoss(torch.nn.Module):
    """A PyTorch module for computing label-smoothed losses on predictions.
    
    Label smoothing simulates uncertainty in label predictions by taking a
    fraction from a selected label and distributing it uniformly over all
    non-selected label classes.
    """

    def __init__(self,
                 number_classes: int,
                 smoothing: float = 0.,
                 dimension: int = -1) -> None:
        """Constructs a label smoothing loss PyTorch module.
        
        :param number_classes: The number of possible classes that labels can
            be chosen from.
        :param smoothing: A smoothing fraction (0.0 and 1.0 both inclusive).
            The degree of uncertainty in label proposals.
        :param dimension: A tensor axis to smooth labels over.
        """
        super().__init__()
        assert(0. <= smoothing <= 1.)
        self.confidence = 1. - smoothing
        self.smoothing = smoothing
        self.num_cls = number_classes
        self.dim = dimension
    
    def forward(self,
                prediction: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """Computes the label smoothing loss between `prediction` and `target`.
        
        :param prediction: The predicted label vector.
        :param target: The ground-truth label vector.
        :returns: The label smoothing loss.
        """
        prediction = prediction.log_softmax(dim=self.dim)
        with torch.no_grad():
            # Build a label-smoothed ground-truth label distribution.
            ground_truth_dist = torch.zeros_like(prediction)
            ground_truth_dist.fill_(self.smoothing / (self.num_cls - 1))
            ground_truth_dist.scatter_(dim=1,
                                       index=target.unsqueeze(dim=1),
                                       value=self.confidence)
        summed = torch.sum(-ground_truth_dist * prediction, dim=self.dim)
        return torch.mean(summed)


class TransformerNonTestingOutput(NamedTuple):
    """A transformer output during training and validation."""
    # The average of summed-over-query-sentence-tokens cross-entropy losses in 
    # the batch.
    loss: torch.Tensor
    # `loss`, but scaled by the cumulative length of all batch query sentences.
    length_scaled_loss: torch.Tensor
    # The cumulative length of all batch query sentences.
    length: torch.Tensor


# A transformer output during testing: A PyTorch tensor of shape (`batch_size`,
# `max_out_length`) and of type `torch.long`. Predicted query language
# sentences.
TransformerTestingOutput = torch.Tensor

TransformerOutput = Union[TransformerNonTestingOutput,
                          TransformerTestingOutput]

DecodeType = Union[Literal['pytorch'], Literal['hugging-face']]


class Transformer(torch.nn.Module):
    """A sequence-to-sequence deep machine learning model inspired by the
    original architecture of Vaswani et al. (2017).
    """

    # The degree of label smoothing during computation of the loss.
    LABEL_SMOOTHING_SCALAR = .1

    def __init__(self,
                 encoder: PreTrainedModel,
                 decoder: Union[torch.nn.TransformerDecoder,
                                PreTrainedModel],
                 config: PretrainedConfig,
                 beam_size: Optional[int] = None,
                 max_length: Optional[int] = None,
                 sos_id: Optional[int] = None,
                 eos_id: Optional[int] = None,
                 device: Optional[TorchDevice] = None) -> None:
        """Constructs a transformer model.
        
        :param encoder: The encoder language model to use in the first half of
            the transformer.
        :param decoder: The decoder language model to use in the second half of
            the transformer. Either a PyTorch transformer decoder model or a
            HuggingFace decoder model.
        :param config: A configuration to specify parts of the architecture of
            this transformer with.
        :param beam_size: The beam size to use. Minimally 1, maximally the
            output vocabulary size, both ends inclusive.
        :param max_length: The maximum (inclusive) number of tokens to include
            in tokenised query language outputs from this transformer.
            Truncation and padding occur for too long and too short sequences,
            respectively. Must be strictly positive.
        :param sos_id: The start-of-sentence (SOS) output vocabulary ID.
        :param eos_id: The end-of-sentence (EOS) output vocabulary ID.
        :param device: Optional. The device to place any PyTorch tensors on.
        """
        super().__init__()

        self.bias: torch.Tensor
        
        # Some type-hinted arguments must not only be instances of
        # subclasses from some base class; they must also have some attributes
        # defined on them. Without an intersection construct, this notion can't
        # currently be cleanly conferred in Python. Thus, we resort to runtime
        # `hasattr` checks.
        assert(hasattr(config, 'hidden_size'))
        assert(type(config.hidden_size) == int)
        assert(hasattr(config, 'vocab_size'))
        assert(type(config.vocab_size) == int)
        assert(hasattr(encoder, 'embeddings'))
        assert(isinstance(encoder, torch.nn.Module))
        assert(hasattr(encoder.embeddings, 'word_embeddings'))
        assert(isinstance(encoder.embeddings.word_embeddings,
                          torch.nn.Embedding))
        
        # Further business-logic checks.
        assert(config.return_dict is True)
        
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.register_buffer(name='bias',
                             tensor=torch.tril(torch.ones(2048, 2048)))
        self.dense = torch.nn.Linear(in_features=config.hidden_size,
                                     out_features=config.hidden_size)
        self.lm_head = torch.nn.Linear(in_features=config.hidden_size,
                                       out_features=config.vocab_size,
                                       bias=False)
        self.lsm = torch.nn.LogSoftmax(dim=-1)
        self.tie_weights()

        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id

        self.device = device
        self.decode_type: DecodeType = \
            'pytorch' \
            if type(decoder) == torch.nn.TransformerDecoder else \
            'hugging-face'

    def tie_or_clone_weights(self,
                             module_1: torch.nn.Module,
                             module_2: torch.nn.Module) -> None:
        """Ties (shares) the weights between two weight-tieable PyTorch modules
        if this is possible, and copies the weights of `module_2` to `module_1`
        otherwise.

        A PyTorch module is considered weight-tieable if it has a `weight`
        attribute that is a `torch.nn.parameter.Parameter`.
        
        :param module_1: One of the modules to weight-tie. When cloning, this
            module will receive the weights from `module_2`.
        :param module_2: One of the modules to weight-tie. When cloning, this
            module will impose its weights onto `module_1`.
        """
        assert(hasattr(module_1, 'weight') and
               type(module_1.weight) == torch.nn.parameter.Parameter)
        assert(hasattr(module_2, 'weight') and
               type(module_2.weight) == torch.nn.parameter.Parameter)
        if self.config.torchscript:
            module_1.weight = torch.nn.Parameter(module_2.weight.clone())
        else:
            module_1.weight = module_2.weight

    def tie_weights(self) -> None:
        """Ties the weights between the in- and output layers of the
        transformer, forcing them to share an identical embedding.
        """
        self.tie_or_clone_weights(self.lm_head,
                                  self.encoder.embeddings.word_embeddings)

    def non_testing_stage_forward(self,
                                  inp_ids: Optional[torch.Tensor] = None,
                                  inp_att_mask: Optional[torch.Tensor] = None,
                                  out_ids: Optional[torch.Tensor] = None,
                                  out_att_mask: Optional[torch.Tensor] = None) -> \
            TransformerNonTestingOutput:
        """Performs a single non-testing stage forward-pass in this
        transformer.
        
        :param inp_ids: Natural language input sequences, encoded as
            sequences of numerical tokens ("token IDs").
        :param inp_att_mask: Per-token attention masks of the natural
            language input sequences.
        :param out_ids: The ground-truth query language sentences, encoded as
            sequences of numerical tokens ("token IDs").
        :param out_att_mask: Per-token attention masks for the ground-truth
            query language output sequences.
        :returns: Three losses: (1) The average of
            summed-over-query-sentence-tokens cross-entropy losses in the
            batch, (2) the same loss, but scaled by the cumulative length of
            all batch query sentences, and (3) the cumulative length of all
            batch query sentences.
        """
        outputs: ModelOutput = \
            self.encoder(input_ids=inp_ids,
                         attention_mask=inp_att_mask)
        if self.decode_type == 'pytorch':
            encoder_output = outputs[0].permute(1, 0, 2).contiguous()
            att_mask = -1.e4 * (1. - self.bias[:out_ids.shape[1], :out_ids.shape[1]])
            out_embeddings = self.encoder.embeddings(input_ids=out_ids).permute(1, 0, 2).contiguous()
            out: torch.Tensor = self.decoder(tgt=out_embeddings,
                                             memory=encoder_output,
                                             tgt_mask=att_mask,
                                             memory_key_padding_mask=(1. - inp_att_mask).bool())
            hidden_states = torch.tanh(self.dense(out)).permute(1, 0, 2).contiguous()
        else:
            encoder_output = outputs[0]
            out: ModelOutput = self.decoder(input_ids=out_ids,
                                            attention_mask=out_att_mask,
                                            encoder_hidden_states=encoder_output,
                                            encoder_attention_mask=inp_att_mask)
            hidden_states = torch.tanh(self.dense(out[0]))
        lm_log_its = self.lm_head(hidden_states)

        active_loss = out_att_mask[..., 1:].ne(0).view(-1) == 1
        shift_log_its = lm_log_its[..., :-1, :].contiguous()
        shift_labels = out_ids[..., 1:].contiguous()

        loss_layer = LabelSmoothingLoss(number_classes=self.config.vocab_size,
                                        smoothing=Transformer.LABEL_SMOOTHING_SCALAR)
        loss = loss_layer(shift_log_its.view(-1, shift_log_its.size(-1))[active_loss],
                          shift_labels.view(-1)[active_loss])
        length = active_loss.sum()
        return TransformerNonTestingOutput(loss=loss,
                                           length_scaled_loss=loss * length,
                                           length=length)

    def zero_device(self) -> TorchDevice:
        """Returns the PyTorch device on which zero-valued scalar tensors
        should be placed.

        :returns: The PyTorch device.
        """
        if self.device is None:
            return torch.device('cuda'
                                if torch.cuda.is_available() else
                                'cpu')
        else:
            return self.device
    
    def testing_stage_forward(self,
                              inp_ids: Optional[torch.Tensor] = None,
                              inp_att_mask: Optional[torch.Tensor] = None) -> \
            TransformerTestingOutput:
        """Performs a single testing stage forward-pass in this transformer.
        
        :param inp_ids: The natural language input sequences, encoded as
            sequences of numerical tokens ("token IDs").
        :param inp_att_mask: Per-token attention masks of the natural
            language input sequences.
        :returns: Predicted query language sentences.
        """
        outputs: ModelOutput = \
            self.encoder(input_ids=inp_ids,
                         attention_mask=inp_att_mask)
        encoder_output = outputs[0].permute(1, 0, 2).contiguous() \
                         if self.decode_type == 'pytorch' else \
                         outputs[0]
        predictions: List[torch.Tensor] = []
        zero = torch.zeros(1, device=self.zero_device())
        for idx in range(inp_ids.shape[0]):
            context = encoder_output[:, idx:(idx + 1)] \
                      if self.decode_type == 'pytorch' else \
                      encoder_output[idx:(idx + 1), :]
            context_mask = inp_att_mask[idx:(idx + 1), :]
            beam = TokenBeamSearcher(beam_size=self.beam_size,
                                     sos_id=self.sos_id,
                                     eos_id=self.eos_id,
                                     device=self.device)
            input_ids = beam.current_state()
            context = context.repeat(1, self.beam_size, 1) \
                      if self.decode_type == 'pytorch' else \
                      context.repeat(self.beam_size, 1, 1)
            context_mask = context_mask.repeat(self.beam_size, 1)
            for _ in range(self.max_length):
                if beam.is_done():
                    break
                if self.decode_type == 'pytorch':
                    att_mask = -1.e4 * (1. - self.bias[:input_ids.shape[1],
                                                       :input_ids.shape[1]])
                    out_embeddings = self.encoder.embeddings(input_ids).permute(1, 0, 2).contiguous()
                    out: torch.Tensor = self.decoder(tgt=out_embeddings,
                                                     memory=context,
                                                     tgt_mask=att_mask,
                                                     memory_key_padding_mask=(1. - context_mask).bool())
                    out = torch.tanh(self.dense(out))
                    hidden_states = out.permute(1, 0, 2).contiguous()[:, -1, :]
                else:
                    att_mask = input_ids > 0
                    out: ModelOutput = self.decoder(input_ids=input_ids,
                                                    attention_mask=att_mask,
                                                    encoder_hidden_states=context,
                                                    encoder_attention_mask=context_mask)
                    hidden_states = torch.tanh(self.dense(out[0]))[:, -1, :]
                out = self.lsm(self.lm_head(hidden_states))
                beam.advance(suc_probs=out)
                input_ids.copy_(input_ids.index_select(0, beam.previous_beams_for_current_state()))
                input_ids = torch.cat(tensors=(input_ids, beam.current_state()),
                                      dim=-1)
            hypotheses = beam.hypotheses(beams=beam.final_token_beams())
            prediction = beam.target_tokens(hypotheses)[:self.beam_size]
            prediction = [torch.cat(tensors=[tkn.view(-1) for tkn in p] +
                                            [zero] * (self.max_length - len(p))).view(1, -1)
                          for p in prediction]
            predictions.append(torch.cat(prediction, 0).unsqueeze(0))
        predictions: torch.Tensor = torch.cat(predictions, 0)
        return predictions

    def forward(self,
                inp_ids: Optional[torch.Tensor] = None,
                inp_att_mask: Optional[torch.Tensor] = None,
                out_ids: Optional[torch.Tensor] = None,
                out_att_mask: Optional[torch.Tensor] = None) -> TransformerOutput:
        """Performs a single forward-pass in this transformer.
        
        The pass comes down to feeding the transformer a natural language
        input sentence, encoded as a sequence of numerical tokens
        ("token IDs"), along with attention masks that allows the transformer
        to ignore null-padding at the end of input sequences. Simultaneously,
        a pair of ground-truth query language token IDs and attention mask
        are supplied in order to compare the transformer's output vocabulary
        predictions to what was expected.

        :param inp_ids: The natural language input sentences, encoded as
            sequences of numerical tokens ("token IDs"). Is of shape 
            (`batch_size`, `max_inp_sequence_length`); entries are of type
            `torch.long`.
        :param inp_att_mask: Per-token attention masks for the natural language
            input sequences. Is of shape (`batch_size`, 
            `max_inp_sequence_length`); entries are of type `torch.float`.
            Each entry is a fraction that represents the degree of a token's
            attention: `0.` means 'ignore', whereas `1.` means 'pay full
            attention to'.
        :param out_ids: The ground-truth query language sentences, encoded as
            sequences of numerical tokens ("token IDs"). Of identical
            specification as `inp_ids`.
        :param out_att_mask: Per-token attention-masks for the ground-truth
            query language output sequences. Of identical specification as
            `inp_att_mask`.
        :returns: Various losses when training or validating; predicted
            query language sentences when testing.
        """
        if out_ids is None:
            # Testing stage: predict.
            return self.testing_stage_forward(inp_ids,
                                              inp_att_mask)
        else:
            # Training or validation stage: predict and compare.
            return self.non_testing_stage_forward(inp_ids,
                                                  inp_att_mask,
                                                  out_ids,
                                                  out_att_mask)
