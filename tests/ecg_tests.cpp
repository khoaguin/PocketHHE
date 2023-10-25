#include "ecg_tests.h"

namespace ecg_test
{
    int test_encrypted_weight(std::vector<seal::Ciphertext> &enc_weight,
                              pktnn::pktmat &original_weight,
                              const seal::SecretKey &he_sk,
                              const seal::BatchEncoder &benc,
                              seal::Decryptor &dec,
                              int vec_size)
    {
        pktnn::pktmat dec_weight_t = sealhelper::decrypt_weight(enc_weight, he_sk, benc, dec, vec_size);
        std::vector<int64_t> decrypted_row;
        decrypted_row = dec_weight_t.getRow(0);
        // utils::print_vec(decrypted_row, 128, "decrypted row", ",\n");
        std::vector<int64_t> org_row = original_weight.getRow(0);
        assert(decrypted_row == org_row);
        std::cout << "Test pass: Decrypted weights are equal to plaintext weights" << std::endl;
        return 0;
    }

    int test_encrypted_bias(std::vector<seal::Ciphertext> &enc_bias,
                            pktnn::pktmat &original_bias,
                            const seal::SecretKey &he_sk,
                            seal::Decryptor &dec)
    {
        pktnn::pktmat dec_bias = sealhelper::decrypt_bias(enc_bias, he_sk, dec);
        assert(dec_bias.getRow(0) == original_bias.getRow(0));
        return 0;
    }
} // end of ecg_test namespace
