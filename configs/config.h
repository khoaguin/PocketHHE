#ifndef CONFIG_H
#define CONFIG_H

namespace config
{
    // HE parameters
    extern uint64_t plain_mod;
    extern uint64_t mod_degree;
    extern int seclevel;
    extern bool use_bsgs;  // used when creating the galois key
    // // Parameters for experimentations
    // extern uint64_t NUM_RUN;  // Number of runs to average over and get the final experimental results
    // extern uint64_t NUM_VEC;  // Number of vectors that the user has
    // extern bool USE_BATCH;
    // extern size_t user_vector_size;  // the length of each user's vector data
}

#endif