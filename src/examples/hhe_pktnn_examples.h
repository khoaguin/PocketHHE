#pragma once

#include <iostream>
#include <vector>

#include <pocketnn/pktnn.h>
#include <seal/seal.h>

#include "../../configs/config.h"
#include "../util/sealhelper.h"
#include "../util/pastahelper.h"
#include "../util/utils.h"
#include "../pasta/pasta_3_plain.h"
#include "../pasta/pasta_3_seal.h"
#include "../../tests/ecg_tests.h"

namespace hhe_pktnn_examples
{
    struct Analyst
    {
        // the model weights and bias in plaintext
        pktnn::pktmat weight; // plaintext weight
        pktnn::pktmat bias;   // plaintext bias
        // the model weights and bias in ciphertext
        std::vector<seal::Ciphertext> enc_weight; // the encrypted weight
        std::vector<seal::Ciphertext> enc_bias;   // the encrypted bias
        // the HE keys
        seal::PublicKey he_pk;
        seal::SecretKey he_sk;
        seal::RelinKeys he_rk;
        seal::GaloisKeys he_gk;
        // the HE encrypted results from the csp
        std::vector<seal::Ciphertext> enc_results;
        // the HE decrypted results
        std::vector<std::vector<int64_t>> dec_results;
        // the final predictions
        std::vector<int64_t> predictions;
    };

    struct Client
    {
        // the symmetric keys
        std::vector<uint64_t> k;           // the secret symmetric keys
        std::vector<seal::Ciphertext> c_k; // the HE encrypted symmetric keys
        // the plaintext data
        pktnn::pktmat testData;   // the plaintext test images
        pktnn::pktmat testLabels; // the plaintext test labels
        // the encrypted data
        std::vector<std::vector<uint64_t>> cs; // the symmetric encrypted data
    };

    struct CSP
    {
        // things received from the analyst (data scientist)
        seal::PublicKey he_pk;
        seal::RelinKeys he_rk;
        seal::GaloisKeys he_gk;
        std::vector<seal::Ciphertext> enc_weight; // the encrypted weight (received from the analyst)
        std::vector<seal::Ciphertext> enc_bias;   // the encrypted bias (received from the analyst)
        // things received from the client
        std::vector<seal::Ciphertext> c_k;     // the HE encrypted symmetric keys (received from the client)
        std::vector<std::vector<uint64_t>> cs; // the symmetric encrypted data (received from the client)
        // the HE secret key needed to construct the HHE object
        seal::SecretKey he_sk;
        // the HE encrypted data after decomposition (and post-process if needed) of the
        // user's symmetric encrypted test data
        std::vector<seal::Ciphertext> c_primes;
        // the HE encrypted results that will be sent to the Analyst
        std::vector<seal::Ciphertext> enc_results;
    };

    /*
        Work in Progess
    */
    int hhe_pktnn_ecg_inference();

    /*
        Work in Progress
    */
    int hhe_pktnn_mnist_inference();

} // end of hhe_pktnn_examples namespace
