"""
Definitions for languages relevant for the KBQA task.
"""


from enum import Enum


class NaturalLanguage(Enum):
	ENGLISH = 'en'
	DUTCH = 'nl'


class FormalLanguage(Enum):
	SPARQL = 'sparql'
