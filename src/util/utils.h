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

    /*
    Helper function: Turn a vector of int64_t that contains positive integers into uint64_t.
    */
    std::vector<uint64_t> int64_to_uint64(std::vector<int64_t> vec);
    
    /*
    Helper function: Print a vector of any type.
    */
    template <typename T>
    inline void print_vec(std::vector<T> const &input, uint size, std::string name = "")
    {   
        std::cout << name << " = ";
        for (int i = 0; i < size; i++) {
            std::cout << input.at(i) << ' ';
        }
        std::cout << std::endl;
    }

}