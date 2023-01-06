#include "pastahelper.h"

namespace pastahelper {
    
    /*
    Helper function: create galois keys indices to create HE galois keys
    */
    std::vector<int> add_gk_indices(bool use_bsgs, const seal::BatchEncoder &benc) {
        size_t PASTA_T_COPPIED = 128;  // get this constant from the file 'pasta_3_plain.h
        uint64_t BSGS_N1 = 16; // coppied from 'pasta_seal_3.h'
        uint64_t BSGS_N2 = 8;
        std::vector<int> gk_indices;
        gk_indices.push_back(0);
        gk_indices.push_back(-1);
        if (PASTA_T_COPPIED * 2 != benc.slot_count())
            gk_indices.push_back(((int)PASTA_T_COPPIED));
        if (use_bsgs) {
            for (uint64_t k = 1; k < BSGS_N2; k++) gk_indices.push_back(-k * BSGS_N1);
        }
        return gk_indices;
    }
}

