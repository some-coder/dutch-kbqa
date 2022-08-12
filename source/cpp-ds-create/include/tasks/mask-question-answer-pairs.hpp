/* Symbols for masking question-answer pairs (header). */

#ifndef MASK_QUESTION_ANSWER_PAIRS_HPP
#define MASK_QUESTION_ANSWER_PAIRS_HPP

#include <boost/program_options.hpp>

namespace DutchKBQADSCreate {
    namespace po = boost::program_options;
    void mask_question_answer_pairs(const po::variables_map &vm);
}

#endif
