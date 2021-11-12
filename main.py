from pre_processing.question import QuestionForm
from pre_processing.translation import NaturalLanguage
from pre_processing.dataset.lc_quad import LCQuAD, create_lc_quad_translations


if __name__ == '__main__':
	lcq = LCQuAD()
	for qa_pair in lcq.qa_pairs:
		try:
			print('%6d: %s' % (
				qa_pair.metadata['uid'],
				qa_pair.q.in_form(
					QuestionForm.BRACKETED_ENTITIES_RELATIONS,
					NaturalLanguage.ENGLISH,
					question_mark=True)
				)
			)
		except KeyError:
			print('%6d: %s' % (qa_pair.metadata['uid'], '(no entry available)'))
