#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <sstream>

namespace utils {
    /*
    Helper function: Prints the name of the example in a fancy banner.
    */
    void print_example_banner(std::string title);

    /*
    Helper function: Print line number.
    */
    void print_line(int line_number);
}