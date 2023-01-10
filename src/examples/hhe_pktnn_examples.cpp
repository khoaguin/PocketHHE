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
    utils::print_example_banner("Privacy-preserving Inference with a 1-layer Neural Network");

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
    std::cout << "Analyst loads the weights and biases from csv files" << "\n";
    pktnn::pktfc fc(config::dim_input, config::num_classes);
    fc.loadWeight("weights/1_layer/fc1_weight_50epochs.csv");
    fc.loadBias("weights/1_layer/fc1_bias_50epochs.csv");
    fc.printWeightShape();
    fc.printBiasShape();
    pktnn::pktmat fc_weight;
    pktnn::pktmat fc_bias;
    fc_weight = fc.getWeight();
    fc_bias = fc.getBias();
    fc_weight.printMat();

    utils::print_line(__LINE__);
    std::cout << "Analyst encrypts the weights and biases" << "\n";
    auto qualifiers = context->first_context_data()->qualifiers();
    std::cout << "Batching enabled: " << std::boolalpha << qualifiers.using_batching << std::endl;
    // fc.encryptWeight(analyst.he_pk, analyst_he_benc, analyst_he_enc);
    // fc.encryptBias(analyst.he_pk, analyst_he_benc, analyst_he_enc);

    // utils::print_line(__LINE__); 
    // std::cout << "---- User ----" << std::endl;
    // std::cout << "User creates the symmetric key" << std::endl;
    
    // std::cout << "User load his data" << std::endl;
    
    // std::cout << "User encrypts his data using the symmetric key" << std::endl;
    
    // std::cout << "User encrypts his symmetric key using HE" << std::endl;


    // utils::print_line(__LINE__); 
    // std::cout << "---- CSP ----" << std::endl;

    // std::cout << "CSP runs the decomposition algorithm to turn the symmetric encrypted data into HE encrypted data" << std::endl;

    // std::cout << "CSP evaluates the HE encrypted neural network on the HE encrypted data" << std::endl;
    
    // utils::print_line(__LINE__); 
    // std::cout << "---- Analyst ----" << std::endl;

    // std::cout << "Analyst decrypts the HE encrypted neural network output using his HE decryption secret key" << std::endl; 

    return 0;
}