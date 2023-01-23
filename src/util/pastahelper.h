#pragma once

#include <vector>
#include <seal/seal.h>
#include <pocketnn/pktnn.h>

#include "utils.h"
#include "../pasta/pasta_3_plain.h"


namespace pastahelper {
    /*
    Helper function: Create galois keys indices to create HE galois keys
    */
    std::vector<int> add_gk_indices(bool use_bsgs, const seal::BatchEncoder &benc);

    /*
    Helper function: Get the symmetric key
    */
    std::vector<uint64_t> get_symmetric_key();

    /*
    Helper function: Symmetric encryption
    */
    std::vector<std::vector<uint64_t>> symmetric_encrypt(const pasta::PASTA &encryptor, const pktnn::pktmat &plaintext) ;

    /*
    Helper function: Symmetric decryption
    */
    std::vector<std::vector<uint64_t>> symmetric_decrypt(const pasta::PASTA &encryptor, const std::vector<std::vector<uint64_t>> &ciphertext);

}