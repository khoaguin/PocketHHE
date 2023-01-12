#include "sealhelper.h"

namespace sealhelper {
/*
Helper function: get a SEALContext from parameters.
*/
std::shared_ptr<seal::SEALContext> get_seal_context(uint64_t plain_mod, uint64_t mod_degree, int seclevel) {
    if (seclevel != 128) throw std::runtime_error("Security Level not supported");
    seal::sec_level_type sec = seal::sec_level_type::tc128;

    seal::EncryptionParameters parms(seal::scheme_type::bfv);
    parms.set_poly_modulus_degree(mod_degree);

    if (mod_degree == 65536) {
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
            0xffffffffcdc0001, 0xffffffffcc40001});  // 1740 bits
    } else {
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
    for (int r = 0; r < weight.rows(); ++r) {
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
    for (int r = 0; r < enc_weight.size(); ++r) {
        std::vector<int64_t> plain;
        seal::Ciphertext encrypted_row = enc_weight[r];
        seal::Plaintext plain_result;
        dec.decrypt(encrypted_row, plain_result);
        benc.decode(plain_result, plain);
        // std::vector<int64_t> dec_row(size);
        // for (int i = 0; i < size; ++i) {
        //     dec_row[i] = plain[i];
        // }
        for (int c = 0; c < vec_size; ++c) {
            // std::cout << plain[c] << " ";
            weight.setElem(r, c, plain[c]);
        }
        // std::cout << "\n";
        // break;
    }
    
    return weight;
}


}  // end of sealhelper namespace
