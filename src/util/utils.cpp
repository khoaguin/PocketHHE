#include "utils.h"

namespace utils {
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

            std::cout << std::endl << banner_top << std::endl << banner_middle << std::endl << banner_top << std::endl;
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
    std::vector<uint64_t> int64_to_uint64(std::vector<int64_t> vec) {
        std::vector<uint64_t> result;
        for (auto i : vec) {
            result.push_back(static_cast<uint64_t>(i));
        }
        return result;
    }

}
