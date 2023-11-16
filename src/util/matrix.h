#pragma once

#include <stdint.h>
#include <limits>
#include <variant>
#include <vector>

namespace matrix
{
    using vector = std::vector<int64_t>;
    using matrix = std::vector<std::vector<int64_t>>;
    using uint128_t = __uint128_t;

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

    static void print_matrix_shape(const matrix &m)
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

    static void print_matrix(const matrix &m)
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

    static matrix read_from_csv(const std::string &filename)
    {
        matrix mat;
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
            std::vector<int64_t> row;

            while (getline(ss, cell, ','))
            {
                row.push_back(std::stoll(cell));
            }
            mat.push_back(row);
        }
        file.close();
        return mat;
    }

    static void print_matrix_stats(const matrix &m)
    {
        if (m.empty() || m[0].empty())
        {
            std::cout << "Matrix is empty" << std::endl;
            return;
        }

        int64_t maxVal = std::numeric_limits<int64_t>::lowest();
        int64_t minVal = std::numeric_limits<int64_t>::max();
        double sum = 0;
        size_t count = 0;

        for (const auto &row : m)
        {
            for (int64_t val : row)
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
