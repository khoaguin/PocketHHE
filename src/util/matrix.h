#pragma once

#include <stdint.h>
#include <limits>
#include <variant>
#include <vector>
#include <algorithm>

namespace matrix
{
    using vector = std::vector<int64_t>;
    using matrix = std::vector<std::vector<int64_t>>;
    using uint128_t = __uint128_t;

    // vo = M * vi: multiplication between a matrix and a vector (with modulus operation)
    static void matMul(vector &vo, const matrix &M, const vector &vi,
                       size_t modulus)
    {
        size_t cols = vi.size();
        size_t rows = M.size();

        std::cout << "\nrows = " << rows << "; ";
        std::cout << "cols = " << cols << "\n";

        if (vo.size() != rows)
            vo.resize(rows);

        for (size_t row = 0; row < rows; row++)
        {
            vo[row] = ((vi[0]) * M[row][0]) % modulus;
            for (size_t col = 1; col < cols; col++)
            {
                vo[row] += (((vi[col]) * M[row][col]) % modulus);
                vo[row] %= modulus;
            }
        }
    }

    // vo = M * vi: multiplication between a matrix and a vector (without modulus operation)
    static void matMulVecNoModulus(vector &vo, const matrix &M, const vector &vi)
    {
        size_t cols = vi.size();
        size_t rows = M.size();
        size_t matrix_cols = M[0].size();
        if (vo.size() != rows)
            vo.resize(rows);

        std::cout << "[" << rows << ", " << matrix_cols << "] * [" << cols << "]" << std::endl;

        for (size_t row = 0; row < rows; row++)
        {
            vo[row] = ((vi[0]) * M[row][0]);
            for (size_t col = 1; col < cols; col++)
            {
                vo[row] += (((vi[col]) * M[row][col]));
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

    // vo = vi * vi
    static void square(vector &vo, const vector &vi, size_t modulus)
    {
        size_t rows = vi.size();
        if (vo.size() != rows)
            vo.resize(rows);

        for (size_t row = 0; row < vi.size(); row++)
            vo[row] = (((vi[row]) * vi[row]) % modulus);
    }

    // vo = vi * vi
    static void square(vector &vo, const vector &vi)
    {
        size_t rows = vi.size();
        if (vo.size() != rows)
            vo.resize(rows);

        for (size_t row = 0; row < vi.size(); row++)
            vo[row] = (((vi[row]) * vi[row]));
    }

    static void print_matrix_shape(const matrix &m, std::string name = "")
    {
        if (m.empty() || m[0].empty())
        {
            std::cout << "Matrix is empty" << std::endl;
            return;
        }
        size_t rows = m.size();
        size_t cols = m[0].size();
        std::cout << name << " shape = [" << rows << ", " << cols << "]" << std::endl;
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

        std::cout << "Maximum Value: " << maxVal << " | ";
        std::cout << "Minimum Value: " << minVal << " | ";
        std::cout << "Average Value: " << avg << std::endl;
    }

    static matrix transpose(const matrix &m)
    {
        // Check if the matrix is empty
        if (m.empty())
            return {};

        size_t rows = m.size();
        size_t cols = m[0].size();

        // Create a new matrix with flipped dimensions
        matrix m_t(cols, std::vector<int64_t>(rows));

        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                m_t[j][i] = m[i][j];
            }
        }

        return m_t;
    }

    static int argmax(const vector &vi)
    {
        if (vi.empty())
        {
            return -1; // Return -1 or some other value to indicate the vector is empty
        }

        auto max_it = std::max_element(vi.begin(), vi.end());
        return std::distance(vi.begin(), max_it);
    }

} // namespace matrix
