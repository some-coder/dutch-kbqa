/* Symbols for parsing standard input at the command-line (header). */

#ifndef COMMAND_LINE_HPP
#define COMMAND_LINE_HPP

#include <unordered_map>
#include <string>
#include <boost/program_options.hpp>

namespace DutchKBQADSCreate {
    namespace po = boost::program_options;
    enum TaskType {
        REPLACE_SPECIAL_SYMBOLS,
        GENERATE_QUESTION_TO_ENTITIES_PROPERTIES_MAP,
        LABEL_ENTITIES_AND_PROPERTIES,
        MASK_QUESTION_ANSWER_PAIRS
    };
    const std::unordered_map<std::string, DutchKBQADSCreate::TaskType> string_to_task_type_map = {
        {"replace-special-symbols",
         DutchKBQADSCreate::REPLACE_SPECIAL_SYMBOLS},
        {"generate-question-entities-properties-map",
         DutchKBQADSCreate::GENERATE_QUESTION_TO_ENTITIES_PROPERTIES_MAP},
        {"label-entities-and-properties",
         DutchKBQADSCreate::LABEL_ENTITIES_AND_PROPERTIES},
        {"mask-question-answer-pairs",
         DutchKBQADSCreate::MASK_QUESTION_ANSWER_PAIRS}
    };
    using vm_desc_pair = std::pair<DutchKBQADSCreate::po::variables_map,
                                   DutchKBQADSCreate::po::options_description>;
    DutchKBQADSCreate::vm_desc_pair dutch_kbqa_vm_desc_pair(int argc, char *argv[]);
    void execute_dutch_kbqa_subprogram(po::variables_map &vm);
}

#endif  /* COMMAND_LINE_HPP */
