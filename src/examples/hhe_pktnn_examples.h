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
        pktnn::pktmat weight;                     // plaintext weight
        pktnn::pktmat bias;                       // plaintext bias
        std::vector<seal::Ciphertext> enc_weight; // the encrypted weight
        std::vector<seal::Ciphertext> enc_bias;   // the encrypted bias
        // the HE keys
        seal::PublicKey he_pk;
        seal::SecretKey he_sk;
        seal::RelinKeys he_rk;
        seal::GaloisKeys he_gk;
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
        std::vector<std::vector<uint64_t>> c_ims; // the symmetric encrypted images
    };

    struct CSP
    {
        // the HE secret key needed to construct the HHE object
        seal::SecretKey he_sk;
        // the HE encrypted data of user's test images (after decomposition of client.c_ims)
        std::vector<seal::Ciphertext> c_prime;
        // the HE encrypted results that will be sent to the Analyst
        seal::Ciphertext c_res;
    };

    /*
        Work in Progress
    */
    int hhe_pktnn_mnist_inference();

    /*
        Work in Progess
    */
    int hhe_pktnn_ecg_inference();

} // end of hhe_pktnn_examples namespace
