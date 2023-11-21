#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <sstream>

namespace utils
{
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
    inline void print_vec(std::vector<T> const &input, size_t size, std::string name = "", std::string separator = " ")
    {
        std::cout << name << " = ";
        for (int i = 0; i < size; i++)
        {
            std::cout << input.at(i) << separator;
        }
        std::cout << std::endl;
    }
    /*
    Helper function: Print a vector of any type.
        1 (𝑥 ≤ −128)
        𝑥/8 + 20 (−128 < 𝑥 ≤ −75)
        𝑥/2 + 48 (−75 < 𝑥 ≤ −32)
        𝑥 + 64 (−32 < 𝑥 ≤ 31)
        𝑥/2 + 80 (31 < 𝑥 ≤ 74)
        𝑥/8 + 108 (74 < 𝑥 ≤ 127)
        127 (127 < 𝑥)
    */
    int simple_pocket_sigmoid(int x);

    /*
    Helper function to print time in ms, s, min, h.
    */
    void print_time(std::string name, size_t time_in_ms);

}
