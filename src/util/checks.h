#include <vector>
#include <stdexcept>
#include <iostream>

#include <seal/seal.h>
#include "matrix.h"

namespace checks
{

    template <typename T>
    inline void are_same_vectors(const matrix::vector &v1, const std::vector<T> &v2)
    {
        if (v1.size() != v2.size())
        {
            throw std::runtime_error("2 vectors have different sizes");
        }

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
            throw std::runtime_error("Assertion failed: Vectors are not the same");
        std::cout << "Check pass: Vectors are the same" << std::endl;
    }

    inline void are_same_matrices(const matrix::matrix &M1,
                                  const matrix::matrix &M2,
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

    inline void are_same_he_sk(const seal::SecretKey &sk1, const seal::SecretKey &sk2)
    {
        // Serialize the keys
        std::stringstream stream1, stream2;
        sk1.save(stream1);
        sk2.save(stream2);

        // Compare serialized keys
        bool are_keys_equal = (stream1.str() == stream2.str());
        if (are_keys_equal)
            throw std::runtime_error("HE secret keys are equal: ");

        std::cout << "Check pass: HE secret keys are different" << std::endl;
    }
}