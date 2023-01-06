#include "hhe_pktnn_examples.h"

struct Analyst {  
    std::vector<int64_t> w;  // plaintext weights
    std::vector<int64_t> b; // plaintext biases
    seal::Ciphertext w_c;  // the encrypted weights
    seal::Ciphertext b_c;  // the encrypted biases
    seal::PublicKey he_pk;
    seal::SecretKey he_sk;
    seal::RelinKeys he_rk;
    seal::GaloisKeys he_gk;
};


int hhe_pktnn_mnist_inference() {
    utils::print_example_banner("HHE with PocketNN");

    Analyst analyst;

    utils::print_line(__LINE__);
    std::cout << "----- Analyst -----" << "\n";
    std::cout << "Analyst constructs the HE context" << "\n";
    std::shared_ptr<seal::SEALContext> context = sealhelper::get_seal_context();
    sealhelper::print_parameters(*context);
    utils::print_line(__LINE__);
    std::cout << "Analyst creates the HE keys, batch encoder, encryptor and evaluator from the context" << "\n";
    seal::KeyGenerator keygen(*context);
    analyst.he_sk = keygen.secret_key();  // HE Decryption Secret Key
    keygen.create_public_key(analyst.he_pk);
    keygen.create_relin_keys(analyst.he_rk);
    seal::BatchEncoder analyst_he_benc(*context);
    seal::Encryptor analyst_he_enc(*context, analyst.he_pk);
    seal::Evaluator analyst_he_eval(*context);
    bool use_bsgs = false;
    std::vector<int> gk_indices = pastahelper::add_gk_indices(use_bsgs, analyst_he_benc);
    keygen.create_galois_keys(gk_indices, analyst.he_gk);
    utils::print_line(__LINE__);
    std::cout << "Analyst loads the weights and biases" << "\n";
    pktnn::pktfc fc1(config::dim_input, config::num_classes);
    fc1.loadWeight("weights/1_layer/fc1_weight.csv");
    fc1.loadBias("weights/1_layer/fc1_bias.csv");
    fc1.printWeight();

    return 0;
}