"""Symbols for performing beam searches on token sequences."""

import torch
from dutch_kbqa_py_model.utilities import TorchDevice
from typing import NamedTuple, List, Optional


class TokenBeam(NamedTuple):
    """A token beam for use in a token beam search algorithm."""
    # A scalar `torch.float` tensor. The summed log-probability of this output
    # vocabulary token sequence.
    summed_log_prb: torch.Tensor
    # The number of non-end-of-string (EOS) tokens present on this token beam.
    # Note that this also includes the start-of-string (SOS) token.
    length: int
    # An integer to uniquely identify this beam with.
    beam_idx: int


# A sequence of scalar `torch.long` PyTorch tensors. Each tensor represents a
# single output vocabulary token ID. The complete sequence represents a
# 'sentence'. It may either be a prediction ('hypothesis') or a ground-truth.
Sentence = List[torch.Tensor]


class TokenBeamSearcher:
    """A convenience class that helps in finding "good" output token sequences
    for transformer models.
    """

    # A log-probability that represents a highly improbable situation.
    IMPROBABLE_LOG_PRB = -1e20

    def __init__(self,
                 beam_size: int,
                 sos_id: int,
                 eos_id: int,
                 device: Optional[TorchDevice] = None) -> None:
        """Constructs a token sequence finder that utilises beam search.
        
        :param beam_size: The beam size to use. Minimally 1, maximally the
            output vocabulary size, both ends inclusive.
        :param sos_id: The start-of-sequence (SOS) output vocabulary ID.
        :param eos_id: The end-of-sequence (EOS) output vocabulary ID.
        :param device: Optional. The device to place any PyTorch tensors on.
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.beam_size = beam_size
        self.eos = eos_id
        self.eos_top = False

        # `self.sum_probs` keeps track of the summed log-probabilities of
        # each beam as the beam search algorithm advances.
        self.sum_probs = torch.zeros(self.beam_size,
                                     dtype=torch.float,
                                     device=self.device)
        
        # `self.dep_beams_list` stores a series of tensors. Successive tensors
        # correspond to successive steps in the beam search algorithm.
        # Each individual tensor gives the (index of) the departed-from beam
        # for each of the tensor's beams.
        self.dep_beams_list: List[torch.Tensor] = []
        
        # `self.sel_tokens_list` stores a series of tensors. Successive tensors
        # correspond to successive steps in the beam search algorithm.
        # Each individual tensor stores `self.beam_size` output vocabulary
        # token IDs. These are the tokens selected per each beam at a step.
        self.sel_tokens_list: List[torch.Tensor] = \
            [torch.zeros(self.beam_size,
                         dtype=torch.long,
                         device=self.device)]
        self.sel_tokens_list[0][0] = sos_id  # TODO(Niels): Not [0][:] (so, for ALL beams)?
        
        # Token beams that have reached an end-of-string (EOS) token.
        self.cpl_beams: List[TokenBeam] = []

    def current_state(self) -> torch.Tensor:
        """Returns the summed log-probabilities of each beam of this token
        beam searcher as it currently stands.
        
        :returns: The summed log-probabilities per beam, returned as a column
            vector.
        """
        return self.sel_tokens_list[-1].view(-1, 1)

    def previous_beams_for_current_state(self) -> torch.Tensor:
        """Returns indices to departed-from beams for the current state.
        
        :returns: The departed-from beam indices for the beams, as it currently
            stands for this token beam searcher.
        """
        return self.dep_beams_list[-1]

    def advance(self, suc_probs: torch.Tensor) -> None:
        """Incorporates the log-probabilities of all beams into the token beam
        searcher's state, effectively advancing the algorithm by one 'step'.

        :param suc_probs: A PyTorch tensor with `self.beam_size` rows and the
            cardinality of the output vocabulary number of columns.
            Per beam (row), the predicted log-probability of observing the
            successor token (column) given the sequence of previous tokens of
            the beam.
        """
        num_tokens: int = suc_probs.size(dim=1)  # output vocabulary size

        # Compute, per beam, summed log-probabilities to successor tokens.
        if len(self.dep_beams_list) > 0:
            beam_probs = self.sum_probs.unsqueeze(dim=1).expand_as(suc_probs) + \
                         suc_probs
            for idx in range(self.beam_size):
                if self.sel_tokens_list[-1][idx] == self.eos:
                    # Successors of EOS tokens should not be chosen.
                    beam_probs[idx, :] = TokenBeamSearcher.IMPROBABLE_LOG_PRB
        else:
            # All beams start from the same location, so they all share the
            # same beginning set of token log-probabilities.
            beam_probs = suc_probs[0, :]

        # Pick the top `self.beam_size` beams with greatest log-probability.
        best_probs: torch.Tensor
        best_probs_idx: torch.Tensor
        flat_beam_probs = beam_probs.view(-1)
        best_probs, best_probs_idx = flat_beam_probs.topk(k=self.beam_size,
                                                          dim=0,
                                                          largest=True,
                                                          sorted=True)
        
        # Per each current-step beam: the previous, departed-from beam index.
        dep_beams = best_probs_idx // num_tokens
        
        # Update the token beam searcher's internal state.
        self.sum_probs = best_probs
        self.dep_beams_list.append(dep_beams)
        self.sel_tokens_list.append(best_probs_idx - dep_beams * num_tokens)
        for idx in range(self.beam_size):
            if self.sel_tokens_list[-1][idx] == self.eos:
                # If a beam has completed, add it to a special list.
                sum_prob = self.sum_probs[idx]
                cpl_beam = TokenBeam(summed_log_prb=sum_prob,
                                     length=len(self.sel_tokens_list) - 1,
                                     beam_idx=idx)
                self.cpl_beams.append(cpl_beam) 
        if self.sel_tokens_list[-1][0] == self.eos:
            # If the 'best' token beam has completed, take note of this.
            self.eos_top = True

    def is_done(self) -> bool:
        """Determines whether the token beam search algorithm is done.
        
        :returns: The question's answer.
        """
        return self.eos_top and len(self.cpl_beams) >= self.beam_size

    def final_token_beams(self) -> List[TokenBeam]:
        """Obtains the final list of token beams.
        
        This method allows for the situation in which the token beam searcher
        has not yet finished obtaining `self.beam_size` completed beams; it
        will return a 'best effort' list of beams in that case.

        :returns: The list of final token beams.
        """
        if len(self.cpl_beams) == 0:
            self.cpl_beams.append(TokenBeam(summed_log_prb=self.sum_probs[0],
                                            length=len(self.sel_tokens_list) - 1,
                                            beam_idx=0))
        self.cpl_beams.sort(key=lambda beam: -beam.summed_log_prb)
        if len(self.cpl_beams) != self.beam_size:
            not_completed: List[TokenBeam] = []
            for idx in range(self.beam_size):
                if self.sel_tokens_list[-1][idx] != self.eos:
                    sum_prob = self.sum_probs[idx]
                    not_completed.append(TokenBeam(summed_log_prb=sum_prob,
                                                   length=len(self.sel_tokens_list) - 1,
                                                   beam_idx=idx))
            not_completed.sort(key=lambda beam: -beam.summed_log_prb)
            self.cpl_beams += not_completed[:self.beam_size - len(self.cpl_beams)]
        return self.cpl_beams[:self.beam_size]

    def hypotheses(self, beams: List[TokenBeam]) -> List[Sentence]:
        """Returns, per each beam in `beams`, a hypothesised (predicted) output
        token sequence.
        
        :param beams: The token beams to construct hypotheses for.
        :returns: The hypothesised token sequences, represented as a
            list-of-lists. The outer list corresponds to the `beams`; each
            inner list stores a series of scalar PyTorch `torch.long` tensors.
            Those latter's scalar tensors represent output vocabulary token
            IDs.
        """
        hypotheses: List[Sentence] = []
        for _, beam_len, beam_idx in beams:
            # Reconstruct a single, hypothesised token sequence per beam.
            hyp = []
            for step in range(len(self.dep_beams_list[:beam_len]) - 1, -1, -1):
                # Trace back from the beam's EOS to the beam's SOS.
                hyp.append(self.sel_tokens_list[step + 1][beam_idx])
                beam_idx = self.dep_beams_list[step][beam_idx]
            hypotheses.append(hyp[::-1])
        return hypotheses

    def target_tokens(self, predictions: List[Sentence]) -> List[Sentence]:
        """Returns target token sequences built from `predictions`.

        This method simply ensures that each prediction in `predictions` stops
        the moment an end-of-string token is encountered.
        
        :param predictions: A list of token sequences to convert into proper
            target token sequences.
        :returns: The target token sequences.
        """
        targets: List[Sentence] = []
        for prd in predictions:
            target: Sentence = []
            for token in prd:
                if token == self.eos:
                    break  # the added value of this method
                else:
                    target.append(token)
            targets.append(target)
        return targets
