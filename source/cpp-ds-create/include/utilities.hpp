/* Various utility symbols (header). */

#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <unordered_map>
#include <string>

namespace DutchKBQADSCreate {
    enum NaturalLanguage {
        ENGLISH,
        DUTCH
    };
    const std::unordered_map<std::string, DutchKBQADSCreate::NaturalLanguage> string_to_natural_language_map = {
        {"en", DutchKBQADSCreate::NaturalLanguage::ENGLISH},
        {"nl", DutchKBQADSCreate::NaturalLanguage::DUTCH}
    };
    enum LCQuADSplit {
        TRAIN,
        TEST
    };
    const std::unordered_map<std::string, DutchKBQADSCreate::LCQuADSplit> string_to_lc_quad_split_map = {
        {"train", DutchKBQADSCreate::LCQuADSplit::TRAIN},
        {"test", DutchKBQADSCreate::LCQuADSplit::TEST}
    };
}

#endif  /* UTILITIES_HPP */
