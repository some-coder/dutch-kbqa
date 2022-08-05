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
    vm_desc_pair vdp = dutch_kbqa_vm_desc_pair(argc, argv);
    if (vdp.first.count("help") != 0) {
        std::cout << vdp.second << std::endl;
    } else {
        execute_dutch_kbqa_subprogram(vdp.first);
    }
    return EXIT_SUCCESS;
}
