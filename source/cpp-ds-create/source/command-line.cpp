/* Symbols for parsing standard input at the command-line. */

#include <stdexcept>
#include "command-line.hpp"
#include "replace-special-symbols.hpp"

using namespace DutchKBQADSCreate;

/**
 * @brief Returns a populated command-line variables map for the program.
 * 
 * A 'variables map' is a concept from the Boost library's `program_options`
 * module. Variable maps map command-line flag names to values supplied to them
 * by the user.
 * 
 * @param argc The number (count) of command-line arguments, including the
 *   name of the program.
 * @param argv An array of character pointers. The first entry contains the
 *   name of the program; all successive entries contain command-line
 *   arguments.
 * @return The variables map.
 */
po::variables_map DutchKBQADSCreate::dutch_kbqa_variables_map(int argc, char *argv[]) {
    po::options_description desc(std::string("Post-process LC-QuAD 2.0 ") +
                                 "datasets. Options");
    
    desc.add_options()
        ("help", "Show this help message.")
        ("task,t", po::value<std::string>(), "The manipulation to perform.")
        ("split", po::value<std::string>(), "The dataset split to work on.")
        ("language",
         po::value<std::string>(),
         "The natural language of the file's contents.")
        ("load-file-name",
         po::value<std::string>(),
         "The name of the file to load from.")
        ("save-file-name",
         po::value<std::string>(),
         "The name of the file to save to."); 
    
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm); 
    return vm;
}

/**
 * @brief Defers control to a subprogram based on command-line input values.
 * 
 * @param vm The variables map with which to determine which subprogram to
 *   run, and in what manner.
 */
void DutchKBQADSCreate::execute_dutch_kbqa_subprogram(po::variables_map &vm) {
    if (vm.count("task") == 0) {
        throw std::invalid_argument(std::string(R"(The "-task" ("-t") flag )") +
                                    "is required.");
    }
    auto it = string_to_task_type_map.find(vm["task"].as<std::string>());
    assert(it != string_to_task_type_map.end());
    TaskType task_type = it->second;
    if (task_type == TaskType::REPLACE_SPECIAL_SYMBOLS) {
        replace_special_symbols_in_resources_file(vm);
    } else {
        throw std::invalid_argument(std::string("Task type \"") +
                                    vm["task"].as<std::string>() +
                                    "\" is not supported.");
    }
}
