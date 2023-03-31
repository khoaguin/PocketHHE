#include "utils.h"

namespace utils
{
    /*
    Helper function: Prints the name of the example in a fancy banner.
    */
    void print_example_banner(std::string title)
    {
        if (!title.empty())
        {
            std::size_t title_length = title.length();
            std::size_t banner_length = title_length + 2 * 10;
            std::string banner_top = "+" + std::string(banner_length - 2, '-') + "+";
            std::string banner_middle = "|" + std::string(9, ' ') + title + std::string(9, ' ') + "|";

            std::cout << std::endl
                      << banner_top << std::endl
                      << banner_middle << std::endl
                      << banner_top << std::endl;
        }
    }

    /*
    Helper function: Print line number.
    */
    void print_line(int line_number)
    {
        std::cout << "Line " << std::setw(3) << line_number << " --> ";
    }

    /*
    Helper function: Turn a vector of int64_t that contains positive integers into uint64_t.
    */
    std::vector<uint64_t> int64_to_uint64(std::vector<int64_t> vec)
    {
        std::vector<uint64_t> result;
        for (auto i : vec)
        {
            result.push_back(static_cast<uint64_t>(i));
        }
        return result;
    }

    /*
        Helper function: A simple pocket sigmoid function
        according to the PocketNN paper.
            1 (𝑥 ≤ −128)
            𝑥/8 + 20 (−128 < 𝑥 ≤ −75)
            𝑥/2 + 48 (−75 < 𝑥 ≤ −32)
            𝑥 + 64 (−32 < 𝑥 ≤ 31)
            𝑥/2 + 80 (31 < 𝑥 ≤ 74)
            𝑥/8 + 108 (74 < 𝑥 ≤ 127)
            127 (127 < 𝑥)
    */
    int simple_pocket_sigmoid(int x)
    {
        int y = 0;

        if (x < -127)
            y = 1;
        else if (x < -74)
            y = x / 8 + 20;
        else if (x < -31)
            y = x / 2 + 48;
        else if (x < 32)
            y = x + 64;
        else if (x < 75)
            y = x / 2 + 80;
        else if (x < 128)
            y = x / 8 + 108;
        else
            y = 127;

        return y;
    }

    /*
    Helper function to print time in ms, s, min.
    */
    void print_time(std::string name, size_t time_in_ms)
    {
        std::cout << name << ": " << time_in_ms << " (ms) = "
                  << (float)(time_in_ms)*1e-3 << " (s) = "
                  << (float)(time_in_ms)*1e-3 / 60 << " (min)" << std::endl;
    }
}
