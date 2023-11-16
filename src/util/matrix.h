#pragma once

#include <stdint.h>
#include <limits>
#include <variant>
#include <vector>

namespace matrix
{
    typedef std::vector<uint64_t> vector;
    typedef std::vector<int64_t> int_vector;
    typedef std::vector<std::vector<uint64_t>> matrix;
    typedef std::vector<std::vector<int64_t>> int_matrix;
    typedef __uint128_t uint128_t;

    static void matMul(vector &vo, const matrix &M, const vector &vi,
                       size_t modulus)
    {
        size_t cols = vi.size();
        size_t rows = M.size();

        if (vo.size() != rows)
            vo.resize(rows);

        for (size_t row = 0; row < rows; row++)
        {
            vo[row] = (((uint128_t)vi[0]) * M[row][0]) % modulus;
            for (size_t col = 1; col < cols; col++)
            {
                vo[row] += ((((uint128_t)vi[col]) * M[row][col]) % modulus);
                vo[row] %= modulus;
            }
        }
    }

    // vo = vi + b
    static void vecAdd(vector &vo, const vector &vi, const vector &b,
                       size_t modulus)
    {
        size_t rows = vi.size();
        if (vo.size() != rows)
            vo.resize(rows);

        for (size_t row = 0; row < vi.size(); row++)
            vo[row] = (vi[row] + b[row]) % modulus;
    }

    // vo = M * vi + b
    static void affine(vector &vo, const matrix &M, const vector &vi,
                       const vector &b, size_t modulus)
    {
        matMul(vo, M, vi, modulus);
        vecAdd(vo, vo, b, modulus);
    }

    // vo = M * vi + b
    static void square(vector &vo, const vector &vi, size_t modulus)
    {
        size_t rows = vi.size();
        if (vo.size() != rows)
            vo.resize(rows);

        for (size_t row = 0; row < vi.size(); row++)
            vo[row] = ((((uint128_t)vi[row]) * vi[row]) % modulus);
    }

    template <typename T>
    static void print_matrix_shape(const std::vector<std::vector<T>> &m)
    {
        if (m.empty() || m[0].empty())
        {
            std::cout << "Matrix is empty" << std::endl;
            return;
        }

        size_t rows = m.size();
        size_t cols = m[0].size();
        std::cout << "rows = " << rows << "; cols = " << cols << std::endl;
    }

    template <typename T>
    static void print_matrix(const std::vector<std::vector<T>> &m)
    {
        if (m.empty() || m[0].empty())
        {
            std::cout << "Matrix is empty" << std::endl;
            return;
        }

        size_t rows = m.size();
        size_t cols = m[0].size();
        for (size_t row = 0; row < rows; row++)
        {
            for (size_t col = 0; col < cols; col++)
            {
                std::cout << m[row][col] << " ";
            }
            std::cout << "\n";
        }
    }

    template <typename T>
    static std::vector<std::vector<T>> read_from_csv(const std::string &filename)
    {
        std::vector<std::vector<T>> mat;
        std::ifstream file(filename);
        std::string line;

        if (!file.is_open())
        {
            throw std::runtime_error("Could not open file: " + filename);
        }

        while (getline(file, line))
        {
            std::stringstream ss(line);
            std::string cell;
            std::vector<T> row;

            while (getline(ss, cell, ','))
            {
                if constexpr (std::is_same<T, uint64_t>::value)
                {
                    row.push_back(std::stoull(cell));
                }
                else if constexpr (std::is_same<T, int64_t>::value)
                {
                    row.push_back(std::stoll(cell));
                }
            }
            mat.push_back(row);
        }
        file.close();
        return mat;
    }

    template <typename T>
    static void print_matrix_stats(const std::vector<std::vector<T>> &m)
    {
        if (m.empty() || m[0].empty())
        {
            std::cout << "Matrix is empty" << std::endl;
            return;
        }

        T maxVal = std::numeric_limits<T>::lowest();
        T minVal = std::numeric_limits<T>::max();
        double sum = 0;
        size_t count = 0;

        for (const auto &row : m)
        {
            for (T val : row)
            {
                maxVal = std::max(maxVal, val);
                minVal = std::min(minVal, val);
                sum += val;
                ++count;
            }
        }

        double avg = (count > 0) ? (sum / count) : 0;

        std::cout << "Maximum Value: " << maxVal << std::endl;
        std::cout << "Minimum Value: " << minVal << std::endl;
        std::cout << "Average Value: " << avg << std::endl;
    }

} // namespace matrix
