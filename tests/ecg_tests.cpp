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
        std::vector<int64_t> org_row = original_weight.getRow(0);
        assert(decrypted_row == org_row);
        return 0;
    }
} // end of ecg_test namespace
