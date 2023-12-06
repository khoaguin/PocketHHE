#pragma once

#include <vector>
#include <seal/seal.h>
#include <pocketnn/pktnn.h>

#include "utils.h"
#include "matrix.h"
#include "../pasta/pasta_3_plain.h"
#include "../pasta/pasta_3_seal.h"

namespace pastahelper
{
    /*
    Helper function: Create galois keys indices to create HE galois keys
    */
    std::vector<int> add_gk_indices(bool use_bsgs, const seal::BatchEncoder &benc);

    /*
    Helper function: Create galois keys indices to create HE galois keys
    */
    std::vector<int> add_some_gk_indices(std::vector<int> gk_indices, std::vector<int> &gk_ind);

    /*
    Helper function: Get the symmetric key
    */
    std::vector<uint64_t> get_symmetric_key();

    /*
    Helper function: Symmetric encryption. Loop through the input matrix and encrypt each row.
    Note that the input needs to be a matrix of positive integers.
    */
    std::vector<std::vector<uint64_t>> symmetric_encrypt(const pasta::PASTA &encryptor, const pktnn::pktmat &plaintext);

    /*
    Helper function: Symmetric encryption of an input vector
    */
    std::vector<uint64_t> symmetric_encrypt_vec(const pasta::PASTA &encryptor, const matrix::vector &plaintext);
    /*
    Helper function: Symmetric decryption
    */
    std::vector<std::vector<uint64_t>> symmetric_decrypt(const pasta::PASTA &encryptor, const std::vector<std::vector<uint64_t>> &ciphertext);

    /*
    Helper function: Symmetric decryption of a vector
    */
    std::vector<uint64_t> symmetric_decrypt_vec(const pasta::PASTA &encryptor,
                                                const std::vector<uint64_t> &ciph_vec);
    /*
    Helper function: Encrypt the symmetric key using HE
    This function is adapted from https://github.com/IAIK/hybrid-HE-framework/blob/master/ciphers/pasta_3/seal/pasta_3_seal.cpp
    */
    std::vector<seal::Ciphertext> encrypt_symmetric_key(const std::vector<uint64_t> &ssk,
                                                        bool batch_encoder,
                                                        const seal::BatchEncoder &benc,
                                                        const seal::Encryptor &enc);
    /*
    Helper function: Encrypt the symmetric key using HE
    This function is adapted from https://github.com/IAIK/hybrid-HE-framework/blob/master/ciphers/pasta_3/seal/pasta_3_seal.cpp
    */
    void decomposition(pasta::PASTA_SEAL &HHE,
                       const std::vector<seal::Ciphertext> &c_k,        // the client's HE encrypted symmetric key
                       const std::vector<std::vector<uint64_t>> &c_ims, // the client's symmetric encrypted images
                       std::vector<seal::Ciphertext> &c_prime,          // the HE encrypted images
                       bool use_batch);

    /*
    Calculate the size of the symmetric encrypted data (in MB)
    */
    float sym_enc_data_size(std::vector<std::vector<uint64_t>> cs, bool verbose = false);

} // namespace pastahelper