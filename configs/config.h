#ifndef CONFIG_H
#define CONFIG_H

#include <string>

namespace config
{
    extern bool debugging;
    extern bool verbose;
    // HE parameters
    extern uint64_t plain_mod;
    extern uint64_t mod_degree;
    extern int seclevel;
    extern bool use_bsgs; // used when creating the galois key
    extern bool USE_BATCH;
    // MNIST parameters
    extern int num_test_samples;
    extern int num_classes;
    extern int mnist_rows;
    extern int mnist_cols;
    // Neural network parameters
    extern int dim_input;
    extern int dim_layer1;
    extern int dim_layer2;
    extern int epoch;
    extern int mini_batch_size; // CAUTION: Too big minibatch size can cause overflow
    extern int lr_inv;
    extern int weight_lower_bound;
    extern int weight_upper_bound;
    extern std::string save_path;
    // Parameters to run experiments
    // extern uint64_t NUM_RUN;  // Number of runs to average over and get the final experimental results
    // extern uint64_t NUM_VEC;  // Number of vectors that the user has
    // extern size_t user_vector_size;  // the length of each user's vector data
}

#endif