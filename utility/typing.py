"""
Types for this project that fit nowhere else.
"""


from typing import NewType


HTTPAddress = NewType('HTTPAddress', str)
WikiDataSymbol = NewType('WikiDataSymbol', str)  # a Q or P value on WikiData
