"""
Utility methods to access WikiData's knowledge base.
"""


import requests as rq
import time

from enum import Enum
from typing import Dict, Optional, Tuple

from utility.language import NaturalLanguage
from utility.typing import HTTPAddress, WikiDataSymbol


WIKIDATA_URL = HTTPAddress('https://query.wikidata.org/sparql')
WIKIDATA_QUERY_WAIT = float(60 // 21)
WIKIDATA_TOO_MANY_REQUESTS_WAIT = 5
CONNECTION_DROPPED_WAIT = 30


class WikiDataType(Enum):
	ENTITY = 'Q'
	RELATION = 'P'


def wikidata_query(query: str, variables: Tuple[str, ...], wait: bool = False) -> Dict[str, Tuple[str, ...]]:
	connect_succeeded: bool = False
	req: Optional[rq.Response] = None
	while not connect_succeeded:
		try:
			req = rq.get(WIKIDATA_URL, params={'format': 'json', 'query': query})
			connect_succeeded = True
		except rq.exceptions.ConnectionError:
			print(
				'[query_wikidata] Encountered a connection problem. Retrying in %d seconds...' %
				(CONNECTION_DROPPED_WAIT,))
			time.sleep(CONNECTION_DROPPED_WAIT)
	if not req.ok and req.status_code != 429:
		raise RuntimeError('[query_wikidata] Encountered a problem!\nQuery:\n%s\nCode: %d.' % (query, req.status_code))
	while req.status_code == 429:
		print(
			'[query_wikidata] Encountered a \'429 - Too Many Requests\' error. Retrying in %d seconds...' %
			(WIKIDATA_TOO_MANY_REQUESTS_WAIT,))
		time.sleep(WIKIDATA_TOO_MANY_REQUESTS_WAIT)
		req = rq.get(WIKIDATA_URL, params={'format': 'json', 'query': query})
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
