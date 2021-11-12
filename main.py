from pre_processing.question import QuestionForm
from pre_processing.translation import NaturalLanguage
from pre_processing.dataset.lc_quad import LCQuAD, create_lc_quad_translations


if __name__ == '__main__':
	lcq = LCQuAD()
	for index, qa_pair in enumerate(lcq.qa_pairs):
		try:
			print('%6d: %s' % (
				index,
				qa_pair.q.in_form(
					QuestionForm.BRACKETED_ENTITIES,
					NaturalLanguage.ENGLISH,
					question_mark=True)
				)
			)
		except KeyError:
			print('%6d: %s' % (index, '(no entry available)'))
