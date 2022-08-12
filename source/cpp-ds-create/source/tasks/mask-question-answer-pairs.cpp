/* Symbols for masking question-answer pairs. */

#include <iostream>
#include "tasks/mask-question-answer-pairs.hpp"
#include "suffix-trees/longest-common-substring.hpp"

using namespace DutchKBQADSCreate;

void DutchKBQADSCreate::mask_question_answer_pairs(const po::variables_map &vm) {
    const std::string first("abc");
    const std::string second("ab");
    auto result = SuffixTrees::longest_common_substring(first, second);
    if (result.has_value()) {
        std::cout << "LCS: " << result.value() << std::endl;
    } else {
        std::cout << "No LCS!" << std::endl;
    }
}
