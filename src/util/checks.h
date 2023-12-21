#include <vector>
#include <stdexcept>
#include <iostream>

#include "matrix.h"

namespace checks
{

    template <typename T>
    inline void are_same_vectors(matrix::vector v1, std::vector<T> v2)
    {
        bool is_same = true;
        for (size_t i = 0; i < v1.size(); i++)
        {
            if (v1[i] != v2[i])
            {
                is_same = false;
                break;
            }
        }
        if (!is_same)
        {
            throw std::runtime_error("Assertion failed: Vectors are not the same");
        }
        else
        {
            std::cout << "Check pass: Vectors are the same" << std::endl;
        }
    }

    inline void are_same_matrices(matrix::matrix M1, matrix::matrix M2,
                                  std::string name1 = "", std::string name2 = "")
    {
        if (M1.size() != M2.size() || M1[0].size() != M2[0].size())
        {
            throw std::runtime_error("Assertion failed: Matrices are not the same size");
        }

        for (size_t i = 0; i < M1.size(); i++)
        {
            for (size_t j = 0; j < M1[0].size(); j++)
            {
                if (M1[i][j] != M2[i][j])
                {
                    throw std::runtime_error("Assertion failed: Matrices are not the same");
                }
            }
        }

        std::cout << "Check pass: Matrices (" << name1 << " and " << name2 << ") are the same" << std::endl;
    }

}