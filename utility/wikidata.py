"""
Utility methods to access WikiData's knowledge base.
"""


import requests as rq
import time

from typing import Dict, Tuple

from utility.language import NaturalLanguage
from utility.typing import HTTPAddress, WikiDataSymbol


WIKIDATA_URL = HTTPAddress('https://query.wikidata.org/sparql')
WIKIDATA_QUERY_WAIT = float(60 // 21)


def wikidata_query(query: str, variables: Tuple[str, ...], wait: bool = False) -> Dict[str, Tuple[str, ...]]:
	req = rq.get(WIKIDATA_URL, params={'format': 'json', 'query': query})
	if not req.ok:
		raise RuntimeError('[query_wikidata] Encountered a problem!\nQuery:\n%s\nCode: %d.' % (query, req.status_code))
	d: Dict[str, Tuple[str, ...]] = {variable: tuple() for variable in variables}
	for bind in req.json()['results']['bindings']:
		for variable in variables:
			d[variable] = d[variable] + (bind[variable]['value'],)
	if wait:
		time.sleep(WIKIDATA_QUERY_WAIT)
	return d


def symbol_labels(sym: WikiDataSymbol, language: NaturalLanguage, wait: bool = False) -> Tuple[str, ...]:
	template: str = 'select distinct ?label where {\n\t%s %s ?label .\n\tfilter(lang(?label) = "%s") .\n}'
	prefixed: str = 'wd:%s' % (sym,)
	q1 = wikidata_query(template % (prefixed, 'rdfs:label', language.value), ('label',))
	q2 = wikidata_query(template % (prefixed, 'skos:altLabel', language.value), ('label',))
	if wait:
		time.sleep(WIKIDATA_QUERY_WAIT)
	return q1['label'] + q2['label']
