#include <cstdlib>
#include <iostream>
#include "utilities.hpp"
#include "command-line.hpp"

using namespace DutchKBQADSCreate;

/**
 * @brief Runs the main program.
 * 
 * @param argc The number (count) of command-line arguments, including the
 *   name of the program.
 * @param argv An array of character pointers. The first entry contains the
 *   name of the program; all successive entries contain command-line
 *   arguments.
 * @return int An exit signal. `0` on success; non-zero on failure.
 */
int main(int argc, char *argv[]) {
    std::cout << DutchKBQADSCreate::resources_dir << std::endl;
    po::variables_map vm = dutch_kbqa_variables_map(argc, argv);
    execute_dutch_kbqa_subprogram(vm);
    return EXIT_SUCCESS;
}
