#include <stdint.h>
#include <cstddef>

#include <string>

namespace config
{
    bool debugging = false;
    bool verbose = false;
    // HE parameters
    uint64_t plain_mod = 65537; // 2^16 + 1
    uint64_t mod_degree = 16384;
    int seclevel = 128;
    bool use_bsgs = false;
    bool USE_BATCH = true;
    // MNIST parameters
    int num_test_samples = 1;
    int num_classes = 10;
    int mnist_rows = 28;
    int mnist_cols = 28;
    // Neural network parameters
    int dim_input = mnist_rows * mnist_cols;
    int dim_layer1 = 100; // only for MNIST neural net
    int dim_layer2 = 50;  // only for the MNIST neural net
    int epoch = 100;
    int mini_batch_size = 4; // CAUTION: Too big minibatch size can cause overflow
    int lr_inv = 50;
    int weight_lower_bound = -4095;
    int weight_upper_bound = 4096;
    std::string save_path = "weights/ecg/";
    // Parameters to run experiments
    // uint64_t NUM_RUN = 50;
    // uint64_t NUM_VEC = 1;
    // bool USE_BATCH = true;
    // size_t user_vector_size = 4;
}