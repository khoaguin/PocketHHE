#pragma once

#include <cstddef>
#include <iostream>
#include <seal/seal.h>
#include <pocketnn/pktnn.h>

namespace sealhelper {
    /*
    Helper function: get a SEALContext from parameters.
    */
    std::shared_ptr<seal::SEALContext> get_seal_context(uint64_t plain_mod = 65537, uint64_t mod_degree = 16384, int seclevel =128);

    /*
    Helper function: Prints the parameters in a SEALContext.
    */
    void print_parameters(const seal::SEALContext &context);

    /*
        Helper function: Encrypts a matrix using SEAL.
    */
    seal::Ciphertext encrypting(const std::vector<int64_t> &input, 
                                const seal::PublicKey &he_pk, 
                                const seal::BatchEncoder &benc, 
                                const seal::Encryptor &enc);
}
