// #include <stdexcept>
#include <iostream>
#include <seal/seal.h>
#include <pocketnn/pktnn.h>
#include "../src/util/sealhelper.h"

namespace ecg_test
{
    int test_encrypted_weight(std::vector<seal::Ciphertext> &enc_weight,
                              pktnn::pktmat &original_weight,
                              const seal::SecretKey &he_sk,
                              const seal::BatchEncoder &benc,
                              seal::Decryptor &dec,
                              int vec_size);
} // end of ecg_test namespace
