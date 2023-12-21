#include "sealhelper.h"

namespace sealhelper
{
    /*
    Helper function: get a SEALContext from parameters.
    */
    std::shared_ptr<seal::SEALContext> get_seal_context(uint64_t plain_mod, uint64_t mod_degree, int seclevel)
    {
        if (seclevel != 128)
            throw std::runtime_error("Security Level not supported");
        seal::sec_level_type sec = seal::sec_level_type::tc128;

        seal::EncryptionParameters parms(seal::scheme_type::bfv);
        parms.set_poly_modulus_degree(mod_degree);

        if (mod_degree == 65536)
        {
            sec = seal::sec_level_type::none;
            parms.set_coeff_modulus(
                {0xffffffffffc0001, 0xfffffffff840001, 0xfffffffff6a0001,
                 0xfffffffff5a0001, 0xfffffffff2a0001, 0xfffffffff240001,
                 0xffffffffefe0001, 0xffffffffeca0001, 0xffffffffe9e0001,
                 0xffffffffe7c0001, 0xffffffffe740001, 0xffffffffe520001,
                 0xffffffffe4c0001, 0xffffffffe440001, 0xffffffffe400001,
                 0xffffffffdda0001, 0xffffffffdd20001, 0xffffffffdbc0001,
                 0xffffffffdb60001, 0xffffffffd8a0001, 0xffffffffd840001,
                 0xffffffffd6e0001, 0xffffffffd680001, 0xffffffffd2a0001,
                 0xffffffffd000001, 0xffffffffcf00001, 0xffffffffcea0001,
                 0xffffffffcdc0001, 0xffffffffcc40001}); // 1740 bits
        }
        else
        {
            parms.set_coeff_modulus(seal::CoeffModulus::BFVDefault(mod_degree));
        }

        parms.set_plain_modulus(plain_mod);
        std::shared_ptr<seal::SEALContext> context = std::make_shared<seal::SEALContext>(parms, true, sec);

        return context;
    }

    /*
    Helper function: Prints the parameters in a SEALContext.
    */
    void print_parameters(const seal::SEALContext &context)
    {
        auto &context_data = *context.key_context_data();

        /*
        Which scheme are we using?
        */
        std::string scheme_name;
        switch (context_data.parms().scheme())
        {
        case seal::scheme_type::bfv:
            scheme_name = "BFV";
            break;
        case seal::scheme_type::ckks:
            scheme_name = "CKKS";
            break;
        case seal::scheme_type::bgv:
            scheme_name = "BGV";
            break;
        default:
            throw std::invalid_argument("unsupported scheme");
        }
        std::cout << "/" << std::endl;
        std::cout << "| Encryption parameters :" << std::endl;
        std::cout << "|   scheme: " << scheme_name << std::endl;
        std::cout << "|   poly_modulus_degree: " << context_data.parms().poly_modulus_degree() << std::endl;

        /*
        Print the size of the true (product) coefficient modulus.
        */
        std::cout << "|   coeff_modulus size: ";
        std::cout << context_data.total_coeff_modulus_bit_count() << " (";
        auto coeff_modulus = context_data.parms().coeff_modulus();
        std::size_t coeff_modulus_size = coeff_modulus.size();
        for (std::size_t i = 0; i < coeff_modulus_size - 1; i++)
        {
            std::cout << coeff_modulus[i].bit_count() << " + ";
        }
        std::cout << coeff_modulus.back().bit_count();
        std::cout << ") bits" << std::endl;

        /*
        For the BFV scheme print the plain_modulus parameter.
        */
        if (context_data.parms().scheme() == seal::scheme_type::bfv)
        {
            std::cout << "|   plain_modulus: " << context_data.parms().plain_modulus().value() << std::endl;
        }

        std::cout << "\\" << std::endl;
    }

    /*
    Helper function: Encrypts the weight matrix (each row into a ciphertext)
    */
    std::vector<seal::Ciphertext> encrypt_weight(pktnn::pktmat &weight,
                                                 const seal::PublicKey &he_pk,
                                                 const seal::BatchEncoder &benc,
                                                 const seal::Encryptor &enc)
    {
        // the result
        std::vector<seal::Ciphertext> encrypted_weight;
        // encrypt the rows in the weight matrix
        for (int r = 0; r < weight.rows(); ++r)
        {
            std::vector<int64_t> row = weight.getRow(r);
            // encode and encrypt the row
            seal::Plaintext plain_input;
            benc.encode(row, plain_input);
            seal::Ciphertext encrypted_row;
            enc.encrypt(plain_input, encrypted_row);
            encrypted_weight.push_back(encrypted_row);
        }

        return encrypted_weight;
    }

    std::vector<seal::Ciphertext> encrypt_weight_mat(const matrix::matrix &weight,
                                                     const seal::PublicKey &he_pk,
                                                     const seal::BatchEncoder &benc,
                                                     const seal::Encryptor &enc)
    {
        size_t rows = weight.size();
        std::vector<seal::Ciphertext> encrypted_weights;
        // encrypt the rows in the weight matrix
        for (size_t r = 0; r < rows; r++)
        {
            std::vector<int64_t> row = weight[r];
            seal::Plaintext plain_input;
            benc.encode(row, plain_input);
            seal::Ciphertext encrypted_row;
            enc.encrypt(plain_input, encrypted_row);
            encrypted_weights.push_back(encrypted_row);
        }

        return encrypted_weights;
    }

    std::vector<std::vector<seal::Ciphertext>> encrypt_weight_mat_no_batch(const matrix::matrix &weight,
                                                                           const seal::PublicKey &he_pk,
                                                                           const seal::BatchEncoder &benc,
                                                                           const seal::Encryptor &enc)
    {
        std::vector<std::vector<seal::Ciphertext>> encrypted_weights;
        return encrypted_weights;
    };

    matrix::matrix decrypt_weight_mat(const std::vector<seal::Ciphertext> &enc_weight,
                                      const seal::BatchEncoder &benc,
                                      seal::Decryptor &dec,
                                      const int vec_size)
    {
        size_t num_rows = enc_weight.size();
        matrix::matrix decrypted_weights(num_rows);
        // decrypt each row in the vector of ciphertexts
        for (auto r = 0; r < num_rows; ++r)
        {
            decrypted_weights[r].reserve(vec_size);
            std::vector<int64_t> plain;
            seal::Ciphertext encrypted_row = enc_weight[r];
            seal::Plaintext plain_result;
            dec.decrypt(encrypted_row, plain_result);
            benc.decode(plain_result, plain);
            for (int c = 0; c < vec_size; ++c)
            {
                decrypted_weights[r].push_back(plain[c]);
            }
        }
        return decrypted_weights;
    };

    /*
    Helper function: Decrypt the encrypted weight into a matrix of plaintexts.
    */
    pktnn::pktmat decrypt_weight(std::vector<seal::Ciphertext> &enc_weight,
                                 const seal::SecretKey &he_sk,
                                 const seal::BatchEncoder &benc,
                                 seal::Decryptor &dec,
                                 int vec_size)
    {
        // the decypted result
        int num_rows = enc_weight.size();
        pktnn::pktmat weight(num_rows, vec_size);
        // decrypt each row in the vector of ciphertexts
        for (int r = 0; r < enc_weight.size(); ++r)
        {
            std::vector<int64_t> plain;
            seal::Ciphertext encrypted_row = enc_weight[r];
            seal::Plaintext plain_result;
            dec.decrypt(encrypted_row, plain_result);
            benc.decode(plain_result, plain);
            for (int c = 0; c < vec_size; ++c)
            {
                weight.setElem(r, c, plain[c]);
            }
        }

        return weight;
    }

    /*
    Helper function: Encrypt each number in the bias vector into a ciphertext (no batch encoder needed).
    */
    std::vector<seal::Ciphertext> encrypt_bias(pktnn::pktmat &bias,
                                               const seal::PublicKey &he_pk,
                                               const seal::Encryptor &enc)
    {
        std::vector<seal::Ciphertext> encrypted_bias;
        for (int i = 0; i < bias.cols(); ++i)
        {
            int bi = bias.getElem(0, i);
            std::stringstream stream;
            stream << std::hex << bi;
            std::string plain_bi(stream.str());
            seal::Ciphertext encrypted_bi;
            enc.encrypt(plain_bi, encrypted_bi);
            encrypted_bias.push_back(encrypted_bi);
        }
        return encrypted_bias;
    }

    /*
    Helper function: Decrypt the encrypted bias.
    */
    pktnn::pktmat decrypt_bias(std::vector<seal::Ciphertext> &enc_bias,
                               const seal::SecretKey &he_sk,
                               seal::Decryptor &dec)
    {
        pktnn::pktmat dec_bias(1, enc_bias.size());
        for (int i = 0; i < enc_bias.size(); ++i)
        {
            seal::Ciphertext enc_bi = enc_bias[i];
            seal::Plaintext dec_bi;
            dec.decrypt(enc_bi, dec_bi);
            std::string plain_bi = dec_bi.to_string();
            std::stringstream stream(plain_bi);
            int bi;
            stream >> std::hex >> bi;
            dec_bias.setElem(0, i, bi);
        }
        return dec_bias;
    }

    /*
    Helper function: Decrypt a SEAL Ciphertext.
    */
    std::vector<int64_t> decrypting(const seal::Ciphertext &enc_input,
                                    const seal::SecretKey &he_sk,
                                    const seal::BatchEncoder &benc,
                                    const seal::SEALContext &context,
                                    size_t size)
    {
        // decrypt and decode the encrypted input
        seal::Decryptor decryptor(context, he_sk);
        seal::Plaintext plain_input;
        decryptor.decrypt(enc_input, plain_input);
        std::vector<int64_t> decrypted_vec;
        benc.decode(plain_input, decrypted_vec);

        return {decrypted_vec.begin(), decrypted_vec.begin() + size};
    }

    void packed_enc_multiply(const seal::Ciphertext &encrypted1,
                             const seal::Ciphertext &encrypted2,
                             seal::Ciphertext &destination,
                             const seal::Evaluator &evaluator)
    {
        evaluator.multiply(encrypted1, encrypted2, destination);
    }

    /*
    Calculate the public HE key size in MB
    */
    float he_pk_key_size(seal::PublicKey he_pk,
                         bool verbose)
    {
        std::stringstream pks;
        size_t pk_size = he_pk.save(pks);
        if (verbose)
        {
            std::cout << "The size of the HE public key is " << pk_size * 1e-6 << " Mb" << std::endl;
        }
        return (float)(pk_size * 1e-6);
    }

    /*
    Calculate the HE keys size in MB
    */
    float he_key_size(seal::PublicKey he_pk,
                      seal::RelinKeys he_rk,
                      seal::GaloisKeys he_gk,
                      bool verbose)
    {
        std::stringstream pks, rks, gks;
        size_t pk_size = he_pk.save(pks);
        size_t rk_size = he_rk.save(pks);
        size_t gk_size = he_gk.save(pks);
        float total_keys_size = (float)((pk_size + rk_size + gk_size) * 1e-6);

        if (verbose)
        {
            std::cout << "The size of the HE public key is " << pk_size * 1e-6 << " Mb" << std::endl;
            std::cout << "The size of the HE relin key is " << rk_size * 1e-6 << " Mb" << std::endl;
            std::cout << "The size of the HE galois key is " << gk_size * 1e-6 << " Mb" << std::endl;
            std::cout << "The total size of the HE keys is " << total_keys_size << " Mb" << std::endl;
        }

        return total_keys_size;
    };

    /*
    Calculate the HE encrypted weights and biases size in MB
    */
    float enc_weight_bias_size(const std::vector<seal::Ciphertext> &enc_weight,
                               const std::vector<seal::Ciphertext> &enc_bias,
                               bool ignore_bias,
                               bool verbose)
    {
        size_t enc_weight_size = 0;
        size_t enc_bias_size = 0;
        for (seal::Ciphertext cw : enc_weight)
        {
            std::stringstream ss;
            enc_weight_size += cw.save(ss);
        }
        if (verbose)
        {
            std::cout << "The size of the encrypted weight is " << enc_weight_size * 1e-6 << " Mb" << std::endl;
        }

        if (ignore_bias == false)
        {
            for (seal::Ciphertext cb : enc_bias)
            {
                std::stringstream ss;
                enc_bias_size += cb.save(ss);
            }
            if (verbose)
            {
                std::cout << "The size of the encrypted bias is " << enc_bias_size * 1e-6 << " Mb" << std::endl;
            }
        }

        return (float)((enc_weight_size + enc_bias_size) * 1e-6);
    }

    /*
    Calculate the size of a vector of seal ciphertext in MB
    */
    float he_vec_size(const std::vector<seal::Ciphertext> &enc_vec,
                      bool verbose,
                      std::string name)
    {
        size_t total_size = 0;
        for (seal::Ciphertext c : enc_vec)
        {
            std::stringstream ss;
            size_t c_size = c.save(ss);
            total_size += c_size;
        }
        if (verbose)
        {
            std::cout << "The size of " << name << " is " << total_size * 1e-6 << " Mb" << std::endl;
        }
        return (float)total_size * 1e-6;
    };

    /*
    Calculate the encrypted sum of the encrypted ciphertext
    Sum the elements of the resulting vector: This involves using the Evaluator to
    rotate and sum the elements in the encrypted state.
    The encrypted sum is the last value of the decrypted destination vector
    */
    void encrypted_vec_sum(const seal::Ciphertext &encrypted_inp,
                           seal::Ciphertext &destination,
                           const seal::Evaluator &evaluator,
                           const seal::GaloisKeys &gal_keys,
                           const size_t vec_size)
    {
        destination = encrypted_inp;
        for (auto i = -1; i > -vec_size; i -= 1)
        {
            seal::Ciphertext rotated;
            evaluator.rotate_rows(encrypted_inp, i, gal_keys, rotated);
            evaluator.add_inplace(destination, rotated);
        }
    };

} // end of sealhelper namespace
