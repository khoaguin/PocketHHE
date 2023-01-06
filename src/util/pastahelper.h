#pragma once

#include <vector>
#include <seal/seal.h>

namespace pastahelper {
    /*
    Helper function: create galois keys indices to create HE galois keys
    */
    std::vector<int> add_gk_indices(bool use_bsgs, const seal::BatchEncoder &benc);
}