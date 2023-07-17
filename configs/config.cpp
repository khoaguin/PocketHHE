#include <stdint.h>
#include <cstddef>

#include <string>

namespace config
{
    // General parameters
    bool debugging = false;
    bool verbose = false;
    int dry_run = true;          // if true, only run a few examples
    int dry_run_num_samples = 2; // num samples to run when dry_run is true
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
    int dim_layer1 = 100; // only for the 3-layer MNIST neural net
    int dim_layer2 = 50;  // only for the 3-layer MNIST neural net
    int epoch = 100;
    int mini_batch_size = 4; // CAUTION: Too big minibatch size can cause overflow
    int lr_inv = 50;
    int weight_lower_bound = -2047;
    int weight_upper_bound = 2048;
    std::string save_weight_path = "weights/hypnogram/int/fc1_weight_100epochs_bz4_clamp1024.csv";
    std::string save_bias_path = "weights/hypnogram/int/fc1_bias_100epochs_bs4_clamp1024.csv";
    // Parameters to run experiments
    // uint64_t NUM_RUN = 50;
    // uint64_t NUM_VEC = 1;
    // bool USE_BATCH = true;
    // size_t user_vector_size = 4;
}