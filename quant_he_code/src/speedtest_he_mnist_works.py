import numpy as np
from Pyfhel import Pyfhel
import time
import math

from numpy import logical_and
import tensorflow as tf
import argparse

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = (x_train / 255.0 * 3).astype(int), (x_test / 255.0 * 3).astype(int)

# Best results, however, are with 0.31 or .3 or 0.29
scale_factor_input = 1 / 3

x_test = np.expand_dims(x_test, axis=1)


def quantise(element, s, b):
    upper = 2 ** (b - 1) - 1
    lower = -2 ** (b - 1)

    value = int(round(element / s))

    if value > upper:
        return upper
    elif value < lower:
        return lower
    else:
        return value


def activation(input_layer):
    return np.square(input_layer)


def convolution(input_image, conv2d_kernel, bias, stride):
    x = 0
    y = 0

    input_shape = input_image.shape
    kernel_shape = conv2d_kernel.shape
    output_shape = (kernel_shape[0],
                    int(np.floor((input_shape[1] - kernel_shape[2] - 1) // stride[0] + 1)),
                    int(np.floor((input_shape[2] - kernel_shape[3] - 1) // stride[1] + 1)))

    output = np.zeros(output_shape, dtype=np.double)

    for i_o in range(kernel_shape[0]):
        for o_x in range(output_shape[1]):
            for o_y in range(output_shape[2]):
                for i_i in range(input_shape[0]):
                    kernel = np.reshape(conv2d_kernel[i_o, i_i, :, :], (kernel_shape[2], kernel_shape[3]))

                    output[i_o][o_x][o_y] += math.fsum(
                        np.multiply(input_image[i_i, x: x + kernel_shape[2], y: y + kernel_shape[3]], kernel)
                        .flatten().tolist())

                output[i_o][o_x][o_y] += bias[i_o]

                y += stride[1]
            x += stride[0]
            y = 0
        x = 0

    return output


def scaled_avg_layer(average_pooling2d_poolsize, average_pooling2d_strides,
                     input_layer):
    x = 0
    y = 0

    input_size = input_layer.shape

    output_shape = (input_size[0],
                    int(np.floor((input_size[1] - average_pooling2d_poolsize[2] - 1) // average_pooling2d_strides[0] + 1)),
                    int(np.floor((input_size[2] - average_pooling2d_poolsize[3] - 1) // average_pooling2d_strides[1] + 1)))

    avg = np.zeros(output_shape, dtype=np.double)

    for i_i in range(output_shape[1]):
        for i_o in range(output_shape[2]):
            for c in range(output_shape[0]):
                avg[c][i_i][i_o] += math.fsum(
                    input_layer[c, x: x + average_pooling2d_poolsize[0],
                    y: y + average_pooling2d_poolsize[1]].flatten().tolist()
                )
            y += average_pooling2d_strides[0]
        x += average_pooling2d_strides[1]
        y = 0
    # x = 0

    return avg


def get_diagonal(pos, mat):
    max_size = max(mat.shape[0], mat.shape[1])
    min_size = min(mat.shape[0], mat.shape[1])

    diag = np.zeros(min_size, dtype=np.int64)
    j = pos
    i = 0
    k = 0

    while (i < max_size - pos) and (i < min_size) and (j < max_size):
        diag[k] = mat[i, j]
        k += 1
        i += 1
        j += 1

    i = max_size - pos
    j = 0

    while (i < mat.shape[0]) and (j < pos):
        diag[k] = mat[i, j]
        k += 1
        i += 1
        j += 1

    return diag


def diagonal_repr(matrix):
    n_o = matrix.shape[0]
    n_i = matrix.shape[1]

    div_oi = int(n_i / n_o)

    diag = np.zeros((n_o, n_i), dtype=np.int64)

    for j in range(n_o):
        tmp = get_diagonal(j, matrix)
        if div_oi > 1:
            for i in range(int(div_oi)):
                tmp = np.append(tmp, get_diagonal(j + i * n_o, matrix))

        diag[j] = tmp

    return diag


def duplicate(vec, n):
    return np.tile(vec, n)


def rot_plain_nohe(a, b, n_o, n_i, fact):
    div_oi = int(n_i / n_o)
    slot_size = int(fact * 2**(np.ceil(np.log2(n_i))))

    diag = np.zeros((n_o, n_i), dtype=np.int64)

    for j in range(n_o):
        tmp = get_diagonal(j, a)
        if div_oi > 1:
            for i in range(div_oi - 1):
                tmp = np.append(tmp, get_diagonal(j + (i + 1) * n_o, a))

        diag[j] = tmp

    output = np.zeros(slot_size, dtype=np.int64)
    pad_length = 240 + 7*1024

    for i in range(n_o):
        # output += duplicate(np.append(diag[i], np.zeros(pad_length, dtype=np.int64)), fact) * np.roll(b, -i)
        # output += diag[i] * np.roll(b, -i)
        output += np.tile(diag[i], fact) * np.roll(b, -i)
        # Rotation to the left; i.e. no fold of the end is done

    result = np.zeros(slot_size, dtype=np.int64)

    for i in range(div_oi):
        result += np.roll(output, - (8192 - (i * n_o + pad_length)))
        # Rotation to the right; i.e. taking the padding into account
    return result


def rot_plain(a, b, n_o, n_i, fact, HE):
    div_oi = int(n_i / n_o)
    slot_size = fact * n_i

    diag = np.zeros((n_o, n_i), dtype=np.int64)

    for j in range(n_o):
        tmp = get_diagonal(j, a)
        if div_oi > 1:
            for i in range(div_oi - 1):
                tmp = np.append(tmp, get_diagonal(j + (i + 1) * n_o, a))

        diag[j] = tmp

    output = HE.encodeInt(np.zeros(slot_size, dtype=np.int64))
    pad_length = 240 + 7*1024

    counter = 0
    for i in range(n_o):
        if np.sum(np.abs(diag[i])) != 0:
            output += HE.encodeInt(duplicate(diag[i], fact)) * (b << i)
            counter += 1
        # output += diag[i] * np.roll(b, -i)
        # output += np.tile(diag[i], fact) * np.roll(b, -i)

    result = HE.encodeInt(np.zeros(slot_size, dtype=np.int64))
    if counter != 0:
        for i in range(div_oi):
            result += (output << (8192 - (i * n_o + pad_length)))
        return result
    else:
        return None


def rotation_avg(input_image, input_shape, poolsize, pool_stride, data_stride, HE):
    x = 0
    y = 0

    original_shape = input_image.shape
    if len(original_shape) != 3:
        original_shape = (original_shape[0], int(np.sqrt(original_shape[1])), int(np.sqrt(original_shape[1])))

    output_shape = (original_shape[0],
                    int(np.floor((input_shape[1] - poolsize[2]-1) // pool_stride[0] + 1)),
                    int(np.floor((input_shape[2] - poolsize[3] - 1) // pool_stride[1] + 1)))

    n = original_shape[1]
    n2 = n ** 2

    # input_flat = input_image.flatten()

    result = np.zeros((original_shape[0], original_shape[1] * original_shape[2]), dtype=np.int64)

    if data_stride[0] == 1:
        for c_i in range(original_shape[0]):
            tmp = np.zeros(original_shape[1] * original_shape[2], dtype=np.int64)
            for j in range(poolsize[0]):
                for i in range(poolsize[1]):
                    tmp += np.roll(input_image[c_i, :].flatten(), -i - n * j)
            result[c_i] = tmp
    elif data_stride[0] == 2:
        for c_i in range(original_shape[0]):
            tmp = np.zeros(original_shape[1] * original_shape[2], dtype=np.int64)
            for j in range(poolsize[0]):
                for i in range(poolsize[1]):
                    tmp += np.roll(input_image[c_i, :].flatten(), -(i * 2) - n * (j * 2))
            result[c_i] = tmp
    elif data_stride[0] == 3:
        for c_i in range(original_shape[0]):
            tmp = np.zeros(original_shape[1] * original_shape[2], dtype=np.int64)
            for j in range(poolsize[0]):
                for i in range(poolsize[1]):
                    tmp += np.roll(input_image[c_i, :].flatten(), -(i * 4) - n * (j * 4))
            result[c_i] = tmp

    # Dimension of the output C_O, C_I, W, H
    mask = np.ones(output_shape, dtype=np.int64)

    # ToDo: Now only works for max conv_stride = 2
    # Diliation type of spread out
    if pool_stride[0] > 1:
        # When stride of a convolution is >1, the data is split out with a factor of data_stride
        zeros = 2 ** data_stride[0]
        for i in range(output_shape[1] - 1):
            for k in range(zeros - 1):
                mask = np.insert(mask, zeros * i + k + 1, 0, axis=2)
        for i in range(output_shape[1] - 1):
            for k in range(zeros - 1):
                mask = np.insert(mask, zeros * i + k + 1, 0, axis=1)

    mask = np.pad(mask, ((0, 0),
                         (0, original_shape[1] - mask.shape[1]),
                         (0, original_shape[2] - mask.shape[2])))

    result = result * mask.reshape(output_shape[0], original_shape[1] * original_shape[2])
    return result


def rotation_conv(input_image, data_shape, input_shape, conv2d_kernel, bias, conv_stride, data_stride, fact, HE):

    original_shape = data_shape
    if len(original_shape) != 3:
        original_shape = (original_shape[0], int(np.sqrt(original_shape[1])), int(np.sqrt(original_shape[1])))

    kernel_shape = conv2d_kernel.shape
    output_shape = (int(np.floor((input_shape[1] - kernel_shape[2]-1) // conv_stride[0] + 1)),
                    int(np.floor((input_shape[2] - kernel_shape[3]-1) // conv_stride[1] + 1)))

    n = original_shape[1]
    n2 = n ** 2

    # input_flat = input_image.flatten()

    result = []

    if data_stride[0] == 1:
        for c_o in range(kernel_shape[0]):
            tmp = HE.encodeInt(np.zeros(4096, dtype=np.int64))
            for c_i in range(kernel_shape[1]):
                tmp_ency = input_image[c_i]
                for j in range(kernel_shape[2]):
                    for i in range(kernel_shape[3]):
                        # Extra check, else a transpartent Ctxt is created
                        if conv2d_kernel[c_o, c_i, j, i] != 0:
                            conv_kernel = HE.encodeInt(np.repeat(conv2d_kernel[c_o, c_i, j, i], 4096))
                            tmp_mult = tmp_ency << (i + n * j)
                            tmp_mult *= conv_kernel
                            tmp += tmp_mult
            # if c_o % 10 == 0:
            #     print("Kernel " + str(c_o))
            result.append(tmp)
    # TODO: Add extra statement for CiFAR
    elif data_stride[0] == 2:
        for c_o in range(kernel_shape[0]):
            tmp = HE.encodeInt(np.zeros(original_shape[1] * original_shape[2], dtype=np.int64))
            for c_i in range(kernel_shape[1]):
                tmp_ency = input_image[c_i]
                for j in range(kernel_shape[2]):
                    for i in range(kernel_shape[3]):
                        if conv2d_kernel[c_o, c_i, j, i] != 0:
                            conv_kernel = HE.encodeInt(np.repeat(conv2d_kernel[c_o, c_i, j, i], 4096))
                            tmp_mult = tmp_ency << ((i * 2) + n * (j * 2))
                            tmp_mult *= conv_kernel
                            tmp += tmp_mult

            result.append(tmp)

    # Need to be placed here, now the mask will turn everything that need to be zero to zero
    for i in range(kernel_shape[0]):
        result[i] += HE.encode(bias[i])

    # Dimension of the output C_O, C_I, W, H
    mask = np.ones((kernel_shape[0], 1, output_shape[0], output_shape[1]), dtype=np.int64)

    # ToDo: Now only works for max conv_stride = 2
    # Diliation type of spread out
    if conv_stride[0] > 1:
        # When stride of a convolution is >1, the data is split out with a factor of data_stride
        zeros = 2 ** data_stride[0]
        for i in range(output_shape[1] - 1):
            for j in range(zeros - 1):
                mask = np.insert(mask, zeros * i + j + 1, 0, axis=3)
        for i in range(output_shape[1] - 1):
            for j in range(zeros - 1):
                mask = np.insert(mask, zeros * i + j + 1, 0, axis=2)

    mask = np.pad(mask, ((0, 0),
                         (0, 0),
                         (0, original_shape[1] - mask.shape[2]),
                         (0, original_shape[2] - mask.shape[3])))

    mask = np.append(mask[0, :].flatten(), np.zeros(240, dtype=np.int64))
    mask = HE.encodeInt(duplicate(mask, fact))

    for i in range(len(result)):
        result[i] *= mask

    # result = result * HE.encodeInt(mask.reshape(kernel_shape[0], original_shape[1] * original_shape[2]).flatten())
    return result

    # Clean up the zeros and return in form as would expect
    # return np.reshape(result[result != 0], (kernel_shape[0], output_shape[0], output_shape[1]))


def expand_mat(mat, mat_shape, data_shape, data_stride):
    if data_stride[0] > 1:
        zeros = 2 ** (data_stride[0] - 1)

        for j in range(data_shape[2]):
            for i in range(data_shape[1]):
                for k in range(zeros - 1):
                    mat = np.insert(mat, zeros * mat_shape[1] * j + zeros * i + k + 1, 0, axis=1)

            for i in range(zeros * mat_shape[1] - zeros * data_shape[1]):
                mat = np.insert(mat, zeros * mat_shape[1] * j + zeros * data_shape[1] + i, 0, axis=1)

        diff = mat_shape[1] * mat_shape[1] - mat.shape[1]

        mat = np.append(mat, np.zeros((mat.shape[0], diff), dtype=np.int64), axis=1)

    else:
        diff = mat_shape[1] - data_shape[1]

        for i in range(data_shape[1]):
            for j in range(diff):
                # Always want to have 20 data point, followed by 8 zeros, and then back again 20 data points
                mat = np.insert(mat, data_shape[1] * (i + 1) + i * diff + j, 0, axis=1)

        mat = np.append(mat, np.zeros((mat.shape[0], diff * mat_shape[1]), dtype=np.int64), axis=1)
    return mat


def main():
    HE = Pyfhel()  # Creating empty Pyfhel object

    # 16384; 8192
    HE.contextGen(scheme='bfv', n=16384,
                  t_bits=47)  # Generate context for 'bfv'/'ckks' scheme)  # Generate context for bfv scheme
    HE.keyGen()  # Key Generation: generates a pair of public/secret keys
    HE.rotateKeyGen()  # Rotate key generation --> Allows rotation/shifting
    HE.relinKeyGen()  # Relinearization key generation
    HE.batchEnabled()

    # Lowest possible MNIST: 16384; 38

    # Timings
    # 16384; 55; 333.13s
    # 16384; 35; 311.13s

    # 8192; 24; 58.04s

    cnn_py = __import__("hcnn_mnist_4_stride_data")

    acc, acc_he = 0, 0
    # nb_tests = 1000
    nb_tests = 10

    np_quantise = np.vectorize(quantise)

    bit_width = 4

    conv1_ind_scale = np.max(np.abs(cnn_py.conv2d)) / (2 ** (bit_width - 1) - 1)
    conv1_scale = scale_factor_input * conv1_ind_scale

    # Activation
    conv1_scale_act = conv1_scale ** 2

    # the 1/4 for the scaled average pooling
    conv2_ind_scale = np.max(np.abs(cnn_py.conv2d_1)) / (2 ** (bit_width - 1) - 1)
    conv2_scale = conv1_scale_act * conv2_ind_scale
    # Activation
    conv2_scale_act = conv2_scale ** 2

    dense_ind_scale = np.max(np.abs(cnn_py.dense)) / (2 ** (bit_width - 1) - 1)
    dense_scale = conv2_scale_act * dense_ind_scale

    conv1_bias = np_quantise(cnn_py.conv2d_bias, conv1_scale, 1024)
    conv2_bias = np_quantise(cnn_py.conv2d_1_bias, conv2_scale, 1024)
    dense_bias = np_quantise(cnn_py.dense_bias, dense_scale, 1024)

    conv1_q = np_quantise(cnn_py.conv2d, conv1_ind_scale, bit_width)
    conv2_q = np_quantise(cnn_py.conv2d_1, conv2_ind_scale, bit_width)
    dense1_q = np_quantise(cnn_py.dense, dense_ind_scale, bit_width)

    ################################################################################

    duplicate_factor = int((HE.get_nSlots() / 2) / 1024)

    for i in range(nb_tests):

        # Input image need to be duplicated due to batching properties of FHE
        input_image = duplicate(np.append(x_test[i].flatten(), np.zeros(240, dtype=np.int64)), duplicate_factor)
        input_image = HE.encodeInt(input_image)
        input_image = [HE.encrypt(input_image)]

        start_time_one = time.time()

        # conv1 = convolution(x_test[i], conv1_q, conv1_bias, cnn_py.conv2d_stride)
        conv1_rot = rotation_conv(input_image, (1, 28, 28), (1, 28, 28),
                                  conv1_q, conv1_bias, cnn_py.conv2d_stride, data_stride=(1, 1), fact=duplicate_factor, HE=HE)

        # result = []
        conv1_act_rot = []
        for j in range(len(conv1_rot)):
            # result.append(HE.decryptInt(conv1_rot[j]))
            conv1_act_rot.append((HE.power(conv1_rot[j], 2)))

        # for i in range(len(conv1_rot)):
        #     conv1_act_rot[i] = HE.relinearize(conv1_act_rot[i])

        # result = np.array(result)

        # conv2 = convolution(np.square(conv1), conv2_q, conv2_bias, cnn_py.conv2d_1_stride)  # + conv2_bias

        conv2_rot = rotation_conv(conv1_act_rot, (1, 28, 28), (5, 12, 12), conv2_q, conv2_bias,
                                  cnn_py.conv2d_1_stride, data_stride=(2, 2), fact=duplicate_factor, HE=HE)

        # print("Conv done " + str(i))

        # result = []
        conv2_act_rot = []
        for j in range(len(conv2_rot)):
            # result.append(HE.decryptInt(conv2_rot[j]))
            conv2_act_rot.append(HE.power(conv2_rot[j], 2))

        # for i in range(len(conv2_rot)):
        #     conv2_act_rot[i] = HE.relinearize(conv2_act_rot[i])

        # result = np.array(result)
        # conv2_act = np.square(conv2)
        # real_dense = np.dot(conv2_act.flatten(), dense1_q.T) + dense_bias

        #dense = HE.encodeInt(np.zeros(16, dtype=np.int64))
        dense = []
        slide_size = int(cnn_py.dense_input / cnn_py.conv2d_1_out_channels)

        for j in range(50):
            w_tmp = expand_mat(dense1_q[:, j * slide_size:(j + 1) * slide_size], (1, 28, 28), (1, 4, 4),
                               data_stride=(3, 3))
            w_tmp = np.append(w_tmp, np.zeros((10, 240)), axis=1)
            w_tmp = np.append(w_tmp, np.zeros((6, 1024)), axis=0)
            rot_out = rot_plain(w_tmp, conv2_act_rot[j], 16, 1024, duplicate_factor, HE=HE)
            if rot_out != None:
                dense.append(rot_out)
            # rot_out_nohe = rot_plain_nohe(w_tmp, HE.decryptInt(conv2_act_rot[j])[:8192], 16, 1024, duplicate_factor)
            # if j % 10 == 0:
            #     print("Dense " + str(j))
            # rd = np.dot(dense1_q[:, :400], conv2_act[:400])
            # print(dense[:10] - rd)

        dense_dec = np.zeros(10, dtype=np.int64)
        for k in range(len(dense)):
            # print(HE.noise_level(dense[k]))
            decrypt = HE.decryptInt(dense[k])
            dense_dec += decrypt[:10]


        dense_dec += dense_bias


        if np.argmax(dense_dec[:10]) == y_test[i]:
            acc_he += 1

        if i % 1 == 0:
            single_time = time.time() - start_time_one
            print("Single time " + str(single_time))
            # est_end_time = time.time() + nb_tests * single_time
            # print(time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(est_end_time)))

        # if i % 1 == 0:
            print(str(i) + "\t" + str(acc) + "\t" + str(acc_he) + "\t" + str(HE.noise_level(dense[0])))

    
if __name__ == '__main__':
    t1 = time.perf_counter()
    main()
    t2 = time.perf_counter()
    print(f"Timings {t2 - t1}")

