#include "pastahelper.h"

namespace pastahelper
{

    /*
    Helper function: create galois keys indices to create HE galois keys
    Modified from https://github.com/IAIK/hybrid-HE-framework/blob/master/ciphers/pasta_3/seal/pasta_3_seal.cpp
    */
    std::vector<int> add_gk_indices(bool use_bsgs, const seal::BatchEncoder &benc)
    {
        size_t PASTA_T_COPPIED = 128; // get this constant from the file 'pasta_3_plain.h
        uint64_t BSGS_N1 = 16;        // coppied from 'pasta_seal_3.h'
        uint64_t BSGS_N2 = 8;
        std::vector<int> gk_indices;
        gk_indices.push_back(0);
        gk_indices.push_back(-1);
        if (PASTA_T_COPPIED * 2 != benc.slot_count())
            gk_indices.push_back(((int)PASTA_T_COPPIED));
        if (use_bsgs)
        {
            for (uint64_t k = 1; k < BSGS_N2; k++)
                gk_indices.push_back(-k * BSGS_N1);
        }
        return gk_indices;
    }

    std::vector<int> add_some_gk_indices(std::vector<int> gk_indices, std::vector<int> &gk_ind)
    {
        for (auto &it : gk_ind)
        {
            gk_indices.push_back(it);
        }
        return gk_indices;
    }

    std::vector<uint64_t> get_symmetric_key()
    {
        return {
            0x07a30,
            0x0cfe2,
            0x03bbb,
            0x06ab7,
            0x0de0b,
            0x0c36c,
            0x01c39,
            0x019e0,
            0x0e09c,
            0x04441,
            0x0c560,
            0x00fd4,
            0x0c611,
            0x0a3fd,
            0x0d408,
            0x01b17,
            0x0fa02,
            0x054ea,
            0x0afeb,
            0x0193b,
            0x0b6fa,
            0x09e80,
            0x0e253,
            0x03f49,
            0x0c8a5,
            0x0c6a4,
            0x0badf,
            0x0bcfc,
            0x0ecbd,
            0x06ccd,
            0x04f10,
            0x0f1d6,
            0x07da9,
            0x079bd,
            0x08e84,
            0x0b774,
            0x07435,
            0x09206,
            0x086d4,
            0x070d4,
            0x04383,
            0x05d65,
            0x0b015,
            0x058fe,
            0x0f0d1,
            0x0c700,
            0x0dc40,
            0x02cea,
            0x096db,
            0x06c84,
            0x008ef,
            0x02abc,
            0x03fdf,
            0x0ddaf,
            0x028c7,
            0x0ded4,
            0x0bb88,
            0x020cd,
            0x075c3,
            0x0caf7,
            0x0a8ff,
            0x0eadd,
            0x01c02,
            0x083b1,
            0x0a439,
            0x0e2db,
            0x09baa,
            0x02c09,
            0x0b5ba,
            0x0c7f5,
            0x0161c,
            0x0e94d,
            0x0bf6f,
            0x070f1,
            0x0f574,
            0x0784b,
            0x08cdb,
            0x08529,
            0x027c9,
            0x010bc,
            0x079ca,
            0x01ff1,
            0x0219a,
            0x00130,
            0x0ff77,
            0x012fb,
            0x03ca6,
            0x0d27d,
            0x05747,
            0x0fa91,
            0x00766,
            0x04f27,
            0x00254,
            0x06e8d,
            0x0e071,
            0x0804e,
            0x08b0e,
            0x08e59,
            0x04cd8,
            0x0485f,
            0x0bde0,
            0x03082,
            0x01225,
            0x01b5f,
            0x0a83e,
            0x0794a,
            0x05104,
            0x09c19,
            0x0fdcf,
            0x036fe,
            0x01e41,
            0x00038,
            0x086e8,
            0x07046,
            0x02c07,
            0x04953,
            0x07869,
            0x0e9c1,
            0x0af86,
            0x0503a,
            0x00f31,
            0x0535c,
            0x0c2cb,
            0x073b9,
            0x028e3,
            0x03c2b,
            0x0cb90,
            0x00c33,
            0x08fe7,
            0x068d3,
            0x09a8c,
            0x008e0,
            0x09fe8,
            0x0f107,
            0x038ec,
            0x0b014,
            0x007eb,
            0x06335,
            0x0afcc,
            0x0d55c,
            0x0a816,
            0x0fa07,
            0x05864,
            0x0dc8f,
            0x07720,
            0x0deef,
            0x095db,
            0x07cbe,
            0x0834e,
            0x09adc,
            0x0bab8,
            0x0f8f7,
            0x0b21a,
            0x0ca98,
            0x01a6c,
            0x07e4a,
            0x04545,
            0x078a7,
            0x0ba53,
            0x00040,
            0x09bc5,
            0x0bc7a,
            0x0401c,
            0x00c30,
            0x00000,
            0x0318d,
            0x02e95,
            0x065ed,
            0x03749,
            0x090b3,
            0x01e23,
            0x0be04,
            0x0b612,
            0x08c0c,
            0x06ea3,
            0x08489,
            0x0a52c,
            0x0aded,
            0x0fd13,
            0x0bd31,
            0x0c225,
            0x032f5,
            0x06aac,
            0x0a504,
            0x0d07e,
            0x0bb32,
            0x08174,
            0x0bd8b,
            0x03454,
            0x04075,
            0x06803,
            0x03df5,
            0x091a0,
            0x0d481,
            0x09f04,
            0x05c54,
            0x0d54f,
            0x00344,
            0x09ffc,
            0x00262,
            0x01fbf,
            0x0461c,
            0x01985,
            0x05896,
            0x0fedf,
            0x097ce,
            0x0b38d,
            0x0492f,
            0x03764,
            0x041ad,
            0x02849,
            0x0f927,
            0x09268,
            0x0bafd,
            0x05727,
            0x033bc,
            0x03249,
            0x08921,
            0x022da,
            0x0b2dc,
            0x0e42d,
            0x055fa,
            0x0a654,
            0x073f0,
            0x08df1,
            0x08149,
            0x00d1b,
            0x0ac47,
            0x0f304,
            0x03634,
            0x0168b,
            0x00c59,
            0x09f7d,
            0x0596c,
            0x0d164,
            0x0dc49,
            0x038ff,
            0x0a495,
            0x07d5a,
            0x02d4,
            0x06c6c,
            0x0ea76,
            0x09af5,
            0x0bea6,
            0x08eea,
            0x0fbb6,
            0x09e45,
            0x0e9db,
            0x0d106,
            0x0e7fd,
            0x04ddf,
            0x08bb8,
            0x0a3a4,
            0x03bcd,
            0x036d9,
            0x05acf,
        };
    };

    /*
    Helper function: Symmetric encryption. Loop through the input matrix and encrypt each row.
    Note that the input needs to be a matrix of positive integers.
    */
    std::vector<std::vector<uint64_t>> symmetric_encrypt(const pasta::PASTA &encryptor,
                                                         const pktnn::pktmat &plaintext)
    {
        std::vector<std::vector<uint64_t>> ciphertexts;
        for (int i = 0; i < plaintext.rows(); i++)
        {
            std::vector<int64_t> row = plaintext.getRow(i);
            std::vector<uint64_t> uint_row = utils::int64_to_uint64(row);
            std::vector<uint64_t> encrypted_row = encryptor.encrypt(uint_row);
            // utils::print_vec(encrypted_row, encrypted_row.size(), "encrypted_row");
            ciphertexts.push_back(encrypted_row);
        }
        return ciphertexts;
    };

    /*
    Helper function: Symmetric encryption of a vector of positive integers
    */
    std::vector<uint64_t> symmetric_encrypt_vec(const pasta::PASTA &encryptor,
                                                const matrix::vector &plaintext_vec)
    {
        std::vector<uint64_t> uint_row = utils::int64_to_uint64(plaintext_vec);
        std::vector<uint64_t> ciph = encryptor.encrypt(uint_row);
        return ciph;
    };

    /*
    Helper function: Symmetric decryption of a vector
    */
    std::vector<uint64_t> symmetric_decrypt_vec(const pasta::PASTA &encryptor,
                                                const std::vector<uint64_t> &ciph_vec)
    {
        std::vector<uint64_t> decrypted_row = encryptor.decrypt(ciph_vec);
        return decrypted_row;
    }

    /*
    Helper function: Symmetric decryption
    */
    std::vector<std::vector<uint64_t>> symmetric_decrypt(const pasta::PASTA &encryptor,
                                                         const std::vector<std::vector<uint64_t>> &ciphertext)
    {
        std::vector<std::vector<uint64_t>> plaintexts;
        for (int i = 0; i < ciphertext.size(); i++)
        {
            std::vector<uint64_t> decrypted_row = encryptor.decrypt(ciphertext[i]);
            plaintexts.push_back(decrypted_row);
            // utils::print_vec(decrypted_row, decrypted_row.size(), "decrypted_row");
        }
        return plaintexts;
    }

    std::vector<seal::Ciphertext> encrypt_symmetric_key(const std::vector<uint64_t> &ssk,
                                                        bool batch_encoder,
                                                        const seal::BatchEncoder &benc,
                                                        const seal::Encryptor &enc)
    {
        size_t PASTA_T_COPPIED = 128; // get this constant from the file 'pasta_3_plain.h
        size_t slots = benc.slot_count();
        size_t halfslots = slots >> 1;

        (void)batch_encoder; // patched implementation: ignore param
        std::vector<seal::Ciphertext> enc_sk;
        enc_sk.resize(1);
        seal::Plaintext k;
        std::vector<uint64_t> key_tmp(halfslots + PASTA_T_COPPIED, 0);
        for (size_t i = 0; i < PASTA_T_COPPIED; i++)
        {
            key_tmp[i] = ssk[i];
            key_tmp[i + halfslots] = ssk[i + PASTA_T_COPPIED];
        }
        benc.encode(key_tmp, k);
        enc.encrypt(k, enc_sk[0]);
        return enc_sk;
    }

    /*
    Helper function: Encrypt the symmetric key using HE
    This function is adapted from https://github.com/IAIK/hybrid-HE-framework/blob/master/ciphers/pasta_3/seal/pasta_3_seal.cpp
    */
    void decomposition(pasta::PASTA_SEAL &HHE,
                       const std::vector<seal::Ciphertext> &c_k,        // the client's HE encrypted symmetric key
                       const std::vector<std::vector<uint64_t>> &c_ims, // the client's symmetric encrypted images
                       std::vector<seal::Ciphertext> &c_prime,          //
                       bool use_batch)
    {
        for (int i = 0; i < c_ims.size(); ++i)
        {
            std::vector<uint64_t> c_im = c_ims[i];
            std::cout << "c_im.size() = " << c_im.size() << std::endl;
            std::vector<seal::Ciphertext> c_im_prime = HHE.decomposition(c_im, c_k, use_batch);
            std::cout << "c_im_prime.size() = " << c_im_prime.size() << std::endl;
            // seal::Ciphertext c_prime_i;
        }
    }

    float sym_enc_data_size(std::vector<std::vector<uint64_t>> cs, bool verbose)
    {
        size_t result = 0;
        for (std::vector<uint64_t> c : cs)
        {
            auto one_vec_size = sizeof(uint64_t) * c.size();
            result += one_vec_size;
        }
        if (verbose)
            std::cout << "The size of the symmetric encrypted data is " << result * 1e-6 << " Mb" << std::endl;

        return (float)result * 1e-6;
    }

} // end of the pastahelper namespace
