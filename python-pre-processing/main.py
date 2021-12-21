from pre_processing.question import QuestionForm, StringQuestion
from pre_processing.answer import AnswerForm, FormalLanguage, StringAnswer
from pre_processing.translation import NaturalLanguage, translate_texts
from pre_processing.dataset.lc_quad import LCQuAD, QualityGroup

from typing import Dict, Optional, Tuple

from pre_processing.translation import translate_text, _confirm_access, _configure_credentials
import json
import os
import re
from utility.wikidata import symbol_labels, WikiDataSymbol



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


def quick_and_dirty_translation():
	data_dir = '/home/niels/Documents/programming/dutch-ml/symbol-strings/data'
	d: Dict[int, str] = {}
	cursed_indices = []
	content: Dict
	_confirm_access()
	_configure_credentials()
	with open(os.path.join(data_dir, 'data.json'), 'r') as handle:
		content = json.load(handle)
	old_d: Dict
	entries = []
	for entry in content:
		if entry['question'] is None or len(entry['question']) < 15:
			entries.append(entry)
	for index, entry in enumerate(entries):
		if index % 10 == 0:
			print('\t%5d, or %6.3lf%%' % (index + 1, ((index + 1) / len(entries)) * 1e2))
		print('Considering UID %d with \'%s\'' % (int(entry['uid']), entry['question'] if entry['question'] is not None else '(None)'))
		if entry['paraphrased_question'] is not None and len(entry['paraphrased_question']) > 20:
			d[entry['uid']] = translate_text(entry['paraphrased_question'], NaturalLanguage.DUTCH)
		else:
			d[entry['uid']] = translate_text(entry['NNQT_question'], NaturalLanguage.DUTCH)
		# if entry['question'] is not None and len(entry['question']) > 0:
		# 	d[entry['uid']] = translate_text(entry['question'], NaturalLanguage.DUTCH)
		# elif entry['paraphrased_question'] is not None and len(entry['paraphrased_question']) > 0:
		# 	d[entry['uid']] = translate_text(entry['paraphrased_question'], NaturalLanguage.DUTCH)
		# else:
		# 	try:
		# 		d[entry['uid']] = translate_text(entry['NNQT_question'], NaturalLanguage.DUTCH)  # we have no choice
		# 	except:
		# 		print('Index %5d is cursed!' % (index,))
		# 		cursed_indices.append(index)
		if index % 100 == 0:
			try:
				with open(os.path.join(data_dir, 'data_nl.json'), 'r') as handle:
					old_d = json.load(handle)
			except:
				print('Failed to load in previous data. Using `d` instead.')
				old_d = d
			with open(os.path.join(data_dir, 'data_nl.json'), 'w') as handle:
				old_d.update(d)
				json.dump(old_d, handle)
			print('(Saved data.)')
	try:
		with open(os.path.join(data_dir, 'data_nl.json'), 'r') as handle:
			old_d = json.load(handle)
	except:
		print('Failed to load in previous data. Using `d` instead.')
		old_d = d
	with open(os.path.join(data_dir, 'data_nl.json'), 'w') as handle:
		old_d.update(d)
		json.dump(old_d, handle)
	print('(Saved data one last time.)')


def quick_and_dirty_wikidata_symbol_injection():
	data_dir = '/home/niels/Documents/programming/dutch-ml/symbol-strings/data'
	content: Dict
	with open(os.path.join(data_dir, 'data_test_nl.json'), 'r') as handle:
		content = json.load(handle)
	s = set()
	for k, v in content.items():
		if re.search('(?<![a-zA-Z-])[QP][0-9]+', v):
			for mt in re.finditer('(?<![a-zA-Z-])[QP][0-9]+', v):
				s.add(mt.group())
	s_map = {e: '' for e in s}
	for key in s_map.keys():
		s_map[key] = symbol_labels(WikiDataSymbol(key), NaturalLanguage.DUTCH)[0]
	for k, v in content.items():
		if re.search('(?<![a-zA-Z-])[QP][0-9]+', v):
			print('Replacing for \'' + v + '\'.')
			for key in s_map.keys():
				content[k] = re.subn('(' + key + ')', s_map[key], content[k])[0]
	with open(os.path.join(data_dir, 'data_test_nl.json'), 'w') as handle:
		json.dump(content, handle)


if __name__ == '__main__':
	# quad = LCQuAD(brackets_sparql_preferred_language=NaturalLanguage.DUTCH)
	# q: StringQuestion
	# a: StringAnswer
	# quality: QualityGroup
	# qaq: Tuple[Optional[Tuple[QuestionForm, AnswerForm, QualityGroup]], ...] = \
	# 	(
	# 		(QuestionForm.BRACKETED_ENTITIES_RELATIONS, AnswerForm.WIKIDATA_BRACKETED_ENTITIES_RELATIONS, QualityGroup.Q_AND_P),
	# 		(QuestionForm.BRACKETED_ENTITIES, AnswerForm.WIKIDATA_BRACKETED_ENTITIES, QualityGroup.Q),
	# 		None  # for no entity or relation matches at all
	# 	)
	# for qa in quad.qa_pairs:
	# 	print('QUESTION-ANSWER PAIR %5d' % (qa.metadata['uid']))
	# 	for triple in qaq:
	# 		if triple is None:
	# 			# we hit the last QAQ-triple
	# 			print('\t(no entity or relation matches)')
	# 			break
	# 		try:
	# 			q = qa.q.in_form(triple[0], NaturalLanguage.DUTCH)
	# 			a = qa.a.in_form(triple[1], FormalLanguage.SPARQL)
	# 			quality = triple[2]
	# 			print('\tQ: %s\n\tA: %s\n\tQuality: %s' % (q, a, quality.value))
	# 			break
	# 		except KeyError:
	# 			pass

