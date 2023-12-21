#pragma once

#include <cstddef>
#include <iostream>
#include <typeinfo>
#include <seal/seal.h>
#include <pocketnn/pktnn.h>
#include "../../configs/config.h"
#include "matrix.h"

namespace sealhelper
{
    /*
    Helper function: get a SEALContext from parameters.
    */
    std::shared_ptr<seal::SEALContext> get_seal_context(uint64_t plain_mod, uint64_t mod_degree, int seclevel = 128);

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

    std::vector<seal::Ciphertext> encrypt_weight_mat(const matrix::matrix &weight,
                                                     const seal::PublicKey &he_pk,
                                                     const seal::BatchEncoder &benc,
                                                     const seal::Encryptor &enc);

    std::vector<std::vector<seal::Ciphertext>> encrypt_weight_mat_no_batch(const matrix::matrix &weight,
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

    matrix::matrix decrypt_weight_mat(const std::vector<seal::Ciphertext> &enc_weight,
                                      const seal::BatchEncoder &benc,
                                      seal::Decryptor &dec,
                                      const int vec_size);

    matrix::matrix decrypt_weight_mat_no_batch(const std::vector<std::vector<seal::Ciphertext>> &enc_weight,
                                               const seal::BatchEncoder &benc,
                                               seal::Decryptor &dec,
                                               const int vec_size);

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

    /*
    Helper function: Decrypt a SEAL Ciphertext.
    */
    std::vector<int64_t> decrypting(const seal::Ciphertext &enc_input,
                                    const seal::SecretKey &he_sk,
                                    const seal::BatchEncoder &benc,
                                    const seal::SEALContext &context,
                                    size_t size);
    /*
    Helper function: multiply 2 SEAL Ciphertexts.
    */
    void packed_enc_multiply(const seal::Ciphertext &encrypted1,
                             const seal::Ciphertext &encrypted2,
                             seal::Ciphertext &destination,
                             const seal::Evaluator &evaluator);

    // void packed_enc_addition(const seal::Ciphertext &encrypted1,
    //                          const seal::Ciphertext &encrypted2,
    //                          seal::Ciphertext &destination,
    //                          const seal::Evaluator &evaluator);

    /*
    Calculate the public HE key size in MB
    */
    float he_pk_key_size(seal::PublicKey he_pk,
                         bool verbose = false);
    /*
    Calculate the total HE keys size in MB
    */
    float he_key_size(seal::PublicKey he_pk,
                      seal::RelinKeys he_rk,
                      seal::GaloisKeys he_gk,
                      bool verbose = false);

    /*
    Calculate the size of a vector of seal ciphertext in MB
    */
    float enc_weight_bias_size(const std::vector<seal::Ciphertext> &enc_weight,
                               const std::vector<seal::Ciphertext> &enc_bias,
                               bool ignore_bias = false,
                               bool verbose = false);

    /*
    Calculate the size of the client's HE encrypted symmetric key in MB
    */
    float he_vec_size(const std::vector<seal::Ciphertext> &enc_sym_key,
                      bool verbose = false,
                      std::string name = "HE vector");

    /*
    Calculate the encrypted sum of the encrypted ciphertext
    */
    void encrypted_vec_sum(const seal::Ciphertext &,
                           seal::Ciphertext &destination,
                           const seal::Evaluator &evaluator,
                           const seal::GaloisKeys &gal_keys,
                           const size_t vec_size);

} // end of sealhelper namespace
