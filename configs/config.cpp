#include <stdint.h>
#include <cstddef>

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
    int dim_layer1 = 100;
    int dim_layer2 = 50;
    int epoch = 1000;
    int mini_batch_size = 20; // CAUTION: Too big minibatch size can cause overflow
    int lr_inv = 1000;
    // Parameters to run experiments
    // uint64_t NUM_RUN = 50;
    // uint64_t NUM_VEC = 1;
    // bool USE_BATCH = true;
    // size_t user_vector_size = 4;
}