#pragma once

#include <cstddef>
#include <iostream>
#include <typeinfo>
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
    Helper function: Encrypts the weight matrix into a vector of ciphertexts of its column vectors.
    */
    std::vector<seal::Ciphertext> encrypt_weight(pktnn::pktmat &mat, 
                                    const seal::PublicKey &he_pk, 
                                    const seal::BatchEncoder &benc, 
                                    const seal::Encryptor &enc);
    /* 
    Helper function: Decrypt the encrypted weight into a matrix of plaintexts.
    */
    pktnn::pktmat decrypt_weight(std::vector<seal::Ciphertext> &enc_weight,
                                    const seal::SecretKey &he_sk,
                                    const seal::BatchEncoder &benc,
                                    seal::Decryptor &dec,
                                    int size = 784);
}
