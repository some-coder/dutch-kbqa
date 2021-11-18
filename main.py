from pre_processing.question import QuestionForm
from pre_processing.answer import AnswerForm, FormalLanguage
from pre_processing.translation import NaturalLanguage, translate_texts
from pre_processing.dataset.lc_quad import LCQuAD, create_lc_quad_translations

from typing import Dict, Tuple


def translate_lc_quad_bracket_entity_relation_pairs(
		lcq: LCQuAD,
		language: NaturalLanguage,
		idx_range: Tuple[int, int]) -> None:
	texts: Dict[int, str] = {}
	for qa_pair in lcq.qa_pairs:
		try:
			texts[qa_pair.metadata['uid']] = qa_pair.q.in_form(
					QuestionForm.BRACKETED_ENTITIES_RELATIONS,
					NaturalLanguage.ENGLISH,
					question_mark=True)
		except KeyError:
			pass
	if idx_range[0] < 0 or idx_range[1] > len(texts):
		raise ValueError('Indices must lie between 0 and %6d.' % (len(texts),))
	translate_texts(
		[text for text in texts.items()][idx_range[0]:idx_range[1]],
		language,
		'%s_%s.json' % (QuestionForm.BRACKETED_ENTITIES_RELATIONS.value, language.value))


if __name__ == '__main__':
	quad = LCQuAD()
	# for qa in quad.qa_pairs:
	# 	print('QA %d' % (qa.metadata['uid']))
	# 	try:
	# 		print(qa.a.in_form(AnswerForm.WIKIDATA_BRACKETED_ENTITIES, FormalLanguage.SPARQL))
	# 	except KeyError:
	# 		print('(not available)')
