#include <iostream>
#include <seal/seal.h>
#include <pocketnn/pktnn.h>
#include "../src/util/sealhelper.h"
#include "../src/util/utils.h"

namespace mnist_test
{
    int test_encrypted_weight(std::vector<seal::Ciphertext> &enc_weight,
                              pktnn::pktmat &original_weight,
                              const seal::SecretKey &he_sk,
                              const seal::BatchEncoder &benc,
                              seal::Decryptor &dec,
                              int vec_size);

    int test_encrypted_bias(std::vector<seal::Ciphertext> &enc_bias,
                            pktnn::pktmat &original_bias,
                            const seal::SecretKey &he_sk,
                            seal::Decryptor &dec);
} // end of mnist_test namespace
