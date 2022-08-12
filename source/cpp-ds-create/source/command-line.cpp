/* Symbols for parsing standard input at the command-line. */

#include <stdexcept>
#include "command-line.hpp"
#include "tasks/replace-special-symbols.hpp"
#include "tasks/collect-entities-properties.hpp"
#include "tasks/label-entities-properties.hpp"
#include "tasks/mask-question-answer-pairs.hpp"

using namespace DutchKBQADSCreate;

/**
 * @brief Returns a pair of objects. First, a populated command-line variables
 *   map for the program; second, a description of the command-line options.
 * 
 * A 'variables map' is a concept from the Boost library's `program_options`
 * module. Variable maps map command-line flag names to values supplied to them
 * by the user.
 *
 * A 'command-line options description' lists all command-line options available
 * to the end-user.
 * 
 * @param argc The number (count) of command-line arguments, including the
 *   name of the program.
 * @param argv An array of character pointers. The first entry contains the
 *   name of the program; all successive entries contain command-line
 *   arguments.
 * @return The pair of (1) a variables map and (2) a description of
 *   command-line options.
 */
vm_desc_pair DutchKBQADSCreate::dutch_kbqa_vm_desc_pair(int argc, char *argv[]) {
    po::options_description desc(std::string("Post-process LC-QuAD 2.0 ") +
                                 "datasets. Options");
    
    desc.add_options()
        ("help", "Show this help message.")
        ("task,t", po::value<std::string>(), "The manipulation to perform.")
        ("split", po::value<std::string>(), "The dataset split to work on.")
        ("language",
         po::value<std::string>(),
         "The natural language of the file's contents.")
        ("part-size",
         po::value<int>(),
         "The number of entities and properties to label before saving to disk. Minimally 1.")
        ("quiet",
         po::value<bool>(),
         "Whether to report progress ('false') or not ('true').")
        ("load-file-name",
         po::value<std::string>(),
         "The name of the file to load from.")
        ("save-file-name",
         po::value<std::string>(),
         "The name of the file to save to."); 
    
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm); 
    return { vm, desc };
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
        replace_special_symbols_in_dataset_file(vm);
    } else if (task_type == TaskType::GENERATE_QUESTION_TO_ENTITIES_PROPERTIES_MAP) {
        generate_question_entities_properties_map(vm);
    } else if (task_type == TaskType::LABEL_ENTITIES_AND_PROPERTIES) {
        label_entities_and_properties(vm);
    } else if (task_type == TaskType::MASK_QUESTION_ANSWER_PAIRS) {
        mask_question_answer_pairs(vm);
    } else {
        throw std::invalid_argument(std::string("Task type \"") +
                                    vm["task"].as<std::string>() +
                                    "\" is not supported.");
    }
}
