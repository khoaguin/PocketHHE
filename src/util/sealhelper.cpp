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

    seal::Ciphertext encrypting(const std::vector<int64_t> &input, 
                                const seal::PublicKey &he_pk, 
                                const seal::BatchEncoder &benc, 
                                const seal::Encryptor &enc)
    {
        // encode and encrypt the input
        seal::Plaintext plain_input;
        benc.encode(input, plain_input);
        seal::Ciphertext encrypted_input;
        // enc.encrypt(plain_input, encrypted_input);
        return encrypted_input;
    }
}
