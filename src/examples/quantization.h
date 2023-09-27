// Code based on https://github.com/benja263/Integer-Only-Inference-for-Deep-Learning-in-Native-C/

#include "../util/utils.h"
#include "../../configs/config.h"

namespace quantization
{
    void square_nn_inference(const int *x, const unsigned int N,
                             unsigned int *class_indices);

    void linear_layer(const int *x, const int8_t *w, int *output,
                      const int x_scale_factor, const int *w_scale_factor_inv, const int x_scale_factor_inv,
                      const unsigned int N, const unsigned int K,
                      const unsigned int M, const unsigned int not_output_layer);
    /**
     * @brief A neural network linear layer withthout bias  Y = Square(XW)
     *  x is quantized before multiplication with w and then dequantized per-row granulity prior to the activation function
     *
     * @param x - NxK input matrix
     * @param w - KxM layer weight matrix
     * @param output - NxM output matrix
     * @param x_amax_quant - amax value for quantization of input matrix
     * @param x_w_amax_dequant - 1XM amax values for dequantization of Z=XW
     * @param N
     * @param K
     * @param M
     * @param hidden_layer - boolean value if layer is a hidden layer (activation)
     *
     * @return Void
     */

    void quantize(const int *tensor_in, int8_t *tensor_q, const int scale_factor,
                  const int scale_factor_inv, const unsigned int size);
    /**
     * @brief Scale quantization of a tensor by a single amax value
     *
     * @param tensor_in - input tensor
     * @param tensor_q - output quantized tensor
     * @param scale_factor - 127 / amax
     * @param scale_factor_inv - 1 / scale_factor
     * @param size - size of flattened tensor
     * @return Void
     */

    void mat_mult(const int8_t *mat_l, const int8_t *mat_r,
                  int *result, const unsigned int N,
                  const unsigned int K, const unsigned int M);
    /**
     * @brief Calculates matrix multiplication as: Y = XW
     *
     *
     * @param mat_l - left matrix (X), size NxK
     * @param mat_r - right matrix (W), size (K+1)xM, the last row of W contains the bias vector
     * @param result - output matrix (Y), size NxM
     * @param N - number of rows in X
     * @param K - number of columns/rows in X/W
     * @param M - number of columns in W
     * @return Void
     */

    void dequantize_per_row(int *mat_in, const int *scale_factor_w_inv, const int scale_factor_x_inv, const unsigned int N, const unsigned int M);
    /**
     * @brief Scale dequantization with per-row granulity
     * Each row is multiplied by the corresponding column amax value
     * offline calculate reciprocal(amax) so we can replace division by multiplication
     *
     * @param mat_in - NxM input matrix to dequantize
     * @param scale_factor_w_inv -1XM row vector of layer's weight matrix scale factor values
     * @param scale_factor_x_inv - input inverse scale factor
     * @param N
     * @param M
     * @return Void
     */

    void square(int *tensor, const unsigned int size)
    {
        for (unsigned int i = 0; i < size; i++)
            tensor[i] = i * i;
    }
    /**
     * @brief ReLU activation function
     *
     * @param tensor_in - input tensor
     * @param size - size of flattened tensor
     * @return Void
     */

    void argmax_over_cols(const int *mat_in,
                          unsigned int *indices,
                          const unsigned int N,
                          const unsigned int M);
    /**
     * @brief Calculate argmax per columns of an NxM matrix
     *
     * @param mat_in - NxM input matrix
     * @param indices - 1xM indices to store argmax of each column
     * @param N
     * @param M
     * @return Void
     */
}