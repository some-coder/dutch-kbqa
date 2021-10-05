from pre_processing.language import NaturalLanguage
from pre_processing.translation import translate_for_dataset
from pre_processing.dataset.lc_quad import LCQuAD


if __name__ == '__main__':
	lcq = LCQuAD()
	translate_for_dataset(lcq, NaturalLanguage.DUTCH, 0, append=True)
