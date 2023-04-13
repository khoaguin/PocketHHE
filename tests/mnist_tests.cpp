#include "mnist_tests.h"

namespace mnist_test
{
    int test_encrypted_weight(std::vector<seal::Ciphertext> &enc_weight,
                              pktnn::pktmat &original_weight,
                              const seal::SecretKey &he_sk,
                              const seal::BatchEncoder &benc,
                              seal::Decryptor &dec,
                              int vec_size)
    {
        pktnn::pktmat dec_weight_t = sealhelper::decrypt_weight(enc_weight, he_sk, benc, dec, vec_size);
        std::cout << "Test encrypted weight: NOT DONE\n";
        return 0;
    }

    int test_encrypted_bias(std::vector<seal::Ciphertext> &enc_bias,
                            pktnn::pktmat &original_bias,
                            const seal::SecretKey &he_sk,
                            seal::Decryptor &dec)
    {
        return 0;
    }
} // end of mnist_test namespace
