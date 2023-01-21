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
    Helper function: Encrypts the weight matrix (each row of the matrix into a ciphertext)
    */
    std::vector<seal::Ciphertext> encrypt_weight(pktnn::pktmat &weight,
                                    const seal::PublicKey &he_pk,
                                    const seal::BatchEncoder &benc,
                                    const seal::Encryptor &enc);
    /* 
    Helper function: Decrypt the encrypted weight.
    */
    pktnn::pktmat decrypt_weight(std::vector<seal::Ciphertext> &enc_weight,
                                 const seal::SecretKey &he_sk,
                                 const seal::BatchEncoder &benc,
                                 seal::Decryptor &dec,
                                 int vec_size = 784);

    /*
    Helper function: Encrypt each number in the bias vector into a ciphertext (no batch encoder needed).
    */
    std::vector<seal::Ciphertext> encrypt_bias(pktnn::pktmat &bias,
                                    const seal::PublicKey &he_pk,
                                    const seal::Encryptor &enc);

    /*
    Helper function: Decrypt the encrypted bias.
    */
    pktnn::pktmat decrypt_bias(std::vector<seal::Ciphertext> &enc_bias, 
                               const seal::SecretKey &he_sk,  
                               seal::Decryptor &dec);


}  // end of sealhelper namespace
