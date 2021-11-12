from pre_processing.question import QuestionForm
from pre_processing.translation import NaturalLanguage, translate_texts
from pre_processing.dataset.lc_quad import LCQuAD, create_lc_quad_translations

from typing import Dict, Tuple


def translate_lc_quad_bracket_entity_relation_pairs(
		lcq: LCQuAD,
		language: NaturalLanguage,
		idx_range: Tuple[int, int],
		save_file: str) -> None:
	texts: Dict[int, str] = {}
	for qa_pair in lcq.qa_pairs:
		try:
			texts[qa_pair.metadata['uid']] = qa_pair.q.in_form(
					QuestionForm.BRACKETED_ENTITIES_RELATIONS,
					NaturalLanguage.ENGLISH,
					question_mark=True)
		except KeyError:
			pass
	translate_texts([text for text in texts.items()][idx_range[0]:idx_range[1]], language, save_file)


if __name__ == '__main__':
	translate_lc_quad_bracket_entity_relation_pairs(LCQuAD(), NaturalLanguage.DUTCH, (0, 1), 'dutch-brackets.json')
