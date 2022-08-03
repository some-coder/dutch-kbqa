/* Symbols for parsing standard input at the command-line (header). */

#ifndef COMMAND_LINE_HPP
#define COMMAND_LINE_HPP

#include <unordered_map>
#include <string>
#include <boost/program_options.hpp>

namespace DutchKBQADSCreate {
    namespace po = boost::program_options;
    enum TaskType {
        REPLACE_SPECIAL_SYMBOLS
    };
    const std::unordered_map<std::string, DutchKBQADSCreate::TaskType> string_to_task_type_map = {
        {"replace-special-symbols", DutchKBQADSCreate::REPLACE_SPECIAL_SYMBOLS}
    };
    po::variables_map dutch_kbqa_variables_map(int argc, char *argv[]);
    void execute_dutch_kbqa_subprogram(po::variables_map &vm);
}

#endif  /* COMMAND_LINE_HPP */
