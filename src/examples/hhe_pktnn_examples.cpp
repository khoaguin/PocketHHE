#include "hhe_pktnn_examples.h"

namespace hhe_pktnn_examples
{
    int hhe_pktnn_mnist_inference()
    {
        utils::print_example_banner("Privacy-preserving Inference with a 1-layer Neural Network");

        utils::print_line(__LINE__);
        std::cout << "---------------------- Configs ----------------------"
                  << "\n";
        std::cout << "Debugging = " << config::debugging << "\n";

        // the actors in the protocol
        Analyst analyst;
        Client client;
        CSP csp;

        // The Analyst
        std::cout << "\n";
        utils::print_line(__LINE__);
        std::cout << "---------------------- Analyst ----------------------"
                  << "\n";
        std::cout << "Analyst constructs the HE context"
                  << "\n";
        std::shared_ptr<seal::SEALContext> context = sealhelper::get_seal_context();
        sealhelper::print_parameters(*context);

        utils::print_line(__LINE__);
        std::cout << "Analyst creates the HE keys, batch encoder, encryptor and evaluator from the context"
                  << "\n";
        seal::KeyGenerator keygen(*context);
        analyst.he_sk = keygen.secret_key();     // HE secret key for decryption
        keygen.create_public_key(analyst.he_pk); // HE public key for encryption
        keygen.create_relin_keys(analyst.he_rk); // HE relinearization key to reduce noise in ciphertexts
        seal::BatchEncoder analyst_he_benc(*context);
        bool use_bsgs = false;
        std::vector<int> gk_indices = pastahelper::add_gk_indices(use_bsgs, analyst_he_benc);
        keygen.create_galois_keys(gk_indices, analyst.he_gk); // the HE Galois keys for batch computation
        seal::Encryptor analyst_he_enc(*context, analyst.he_pk);
        seal::Evaluator analyst_he_eval(*context);
        seal::Decryptor analyst_he_dec(*context, analyst.he_sk);

        utils::print_line(__LINE__);
        std::cout << "Analyst loads the weights and biases from csv files"
                  << "\n";
        pktnn::pktfc fc(config::dim_input, config::num_classes);
        if (config::debugging)
        {
            fc.loadWeight("../weights/1_layer/fc1_weight_50epochs.csv");
            fc.loadBias("../weights/1_layer/fc1_bias_50epochs.csv");
        }
        else
        {
            fc.loadWeight("weights/1_layer/fc1_weight_50epochs.csv");
            fc.loadBias("weights/1_layer/fc1_bias_50epochs.csv");
        }
        fc.printWeightShape();
        fc.printBiasShape();
        pktnn::pktmat fc_weight;
        pktnn::pktmat fc_bias;
        fc_weight = fc.getWeight();
        fc_bias = fc.getBias();
        // fc_weight.printMat();

        utils::print_line(__LINE__);
        std::cout << "Analyst encrypts the weights and biases using HE"
                  << "\n";
        auto qualifiers = context->first_context_data()->qualifiers();
        std::cout << "Batching enabled: " << std::boolalpha << qualifiers.using_batching << std::endl;
        std::cout << "Encrypt the transposed weight..."
                  << "\n";
        pktnn::pktmat weight_t;
        weight_t.transposeOf(fc_weight);
        std::vector<seal::Ciphertext> enc_weight = sealhelper::encrypt_weight(weight_t, analyst.he_pk, analyst_he_benc, analyst_he_enc);
        std::cout << "The encrypted weight vector has " << enc_weight.size() << " ciphertexts\n";
        if (config::verbose)
        {
            std::cout << "Decrypt and print the transposed weight to check (open the csv file to compare)..."
                      << "\n";
            pktnn::pktmat dec_weight_t = sealhelper::decrypt_weight(enc_weight, analyst.he_sk, analyst_he_benc, analyst_he_dec, 784);
            dec_weight_t.printMat();
        }

        std::cout << "Encrypt the bias..."
                  << "\n";
        std::vector<seal::Ciphertext> enc_bias = sealhelper::encrypt_bias(fc_bias, analyst.he_pk, analyst_he_enc);
        std::cout << "The encrypted bias vector has " << enc_bias.size() << " ciphertexts\n";
        if (config::verbose)
        {
            std::cout << "Decrypt the bias to check..."
                      << "\n";
            pktnn::pktmat dec_bias = sealhelper::decrypt_bias(enc_bias, analyst.he_sk, analyst_he_dec);
            dec_bias.printMat();
        }
        std::cout << "Analyst sends the encrypted weights and bias to the CSP..."
                  << "\n";

        // Client (Data Owner)
        std::cout << "\n";
        utils::print_line(__LINE__);
        std::cout << "---------------------- Client (Data Owner) ----------------------" << std::endl;

        utils::print_line(__LINE__);
        std::cout << "Client loads his MNIST data" << std::endl;
        pktnn::pktmat mnistTestLabels;
        pktnn::pktmat mnistTestImages;
        pktnn::pktloader::loadMnistImages(mnistTestImages, config::num_test_samples, false, config::debugging); // numTestSamples x (28*28)
        pktnn::pktloader::loadMnistLabels(mnistTestLabels, config::num_test_samples, false, config::debugging); // numTestSamples x 1
        std::cout << "Number of loaded test images = " << mnistTestImages.rows() << "\n";
        std::cout << "Each image is flattened into a vector of size: " << mnistTestImages.cols() << " (=28x28)"
                  << "\n";
        std::cout << ""
                  << "\n";
        mnistTestImages.printShape();
        client.mnistTestData = mnistTestImages;
        client.mnistTestLabels = mnistTestLabels;

        utils::print_line(__LINE__);
        std::cout << "Client creates the symmetric key" << std::endl;
        client.k = pastahelper::get_symmetric_key();
        std::cout << "Symmetric key size: " << client.k.size() << "\n";
        if (config::verbose)
        {
            utils::print_vec(client.k, client.k.size(), "Symmetric key: ");
        }

        utils::print_line(__LINE__);
        std::cout << "Client encrypts his symmetric key using HE" << std::endl;
        client.c_k = pastahelper::encrypt_symmetric_key(client.k, config::USE_BATCH, analyst_he_benc, analyst_he_enc);
        if (config::verbose)
        {
            pasta::PASTA_SEAL SymmetricDecryptor(context, analyst.he_pk, analyst.he_sk, analyst.he_rk, analyst.he_gk);
            std::vector<uint64_t> dec_ck = SymmetricDecryptor.decrypt_result(client.c_k, config::USE_BATCH);
            utils::print_vec(dec_ck, dec_ck.size(), "Decrypted symmetric key: ");
            std::cout << "Decrypted symmetric key size: " << dec_ck.size() << "\n";
            std::cout << "It is ok if the decrypted key size and the key size are different (128 vs 256). It's a part of the pasta library"
                      << "\n";
        }

        utils::print_line(__LINE__);
        std::cout << "Client encrypts his MNIST images using the symmetric key" << std::endl;
        pasta::PASTA SymmetricEncryptor(client.k, config::plain_mod);
        client.c_ims = pastahelper::symmetric_encrypt(SymmetricEncryptor, client.mnistTestData); // the symmetric encrypted images
        std::cout << "Number of encrypted images = " << client.c_ims.size() << "\n";
        if (config::verbose)
        {
            client.mnistTestData.printMat();
            for (auto i : client.c_ims)
            {
                utils::print_vec(i, i.size(), "Encrypted image ");
            }
            auto dec_ims = pastahelper::symmetric_decrypt(SymmetricEncryptor, client.c_ims);
            for (auto i : dec_ims)
            {
                utils::print_vec(i, i.size(), "Decrypted image ");
            }
        }
        std::cout << "The client sends the encrypted images and the encrypted symmetric key to the CSP..."
                  << "\n";

        // The Cloud Service Provider (CSP)
        std::cout << "\n";
        utils::print_line(__LINE__);
        std::cout << "-------------------------- CSP ----------------------" << std::endl;
        // CSP creates a new HE secret key from the context (this is needed to construct the PASTA object for decomposition)
        seal::KeyGenerator csp_keygen(*context);
        csp.he_sk = csp_keygen.secret_key();

        std::cout << "Testing max and min values on plaintext of the linear layer"
                  << "\n";
        pktnn::pktmat result;
        pktnn::pktmat weight_t_row;
        weight_t_row.sliceOf(weight_t, 0, 0, 0, 783);
        weight_t_row.printShape();
        result.matElemMulMat(weight_t_row, client.mnistTestData);
        std::cout << "Max value = " << result.getRowMax(0) << std::endl;
        std::cout << "Min value = " << result.getRowMin(0) << std::endl;

        // inspect if the csp 's HE secret key is different than the analyst' s HE secret
        // very long outputs => comment each line at a time to compare the output
        // csp.he_sk.save(std::cout);
        // std::cout << "\n";
        // analyst.he_sk.save(std::cout);
        // std::cout << "\n";

        // utils::print_line(__LINE__);
        // std::cout << "CSP runs the decomposition algorithm to turn the symmetric encrypted data into HE encrypted data" << std::endl;
        // pasta::PASTA_SEAL HHE(context, analyst.he_pk, csp.he_sk, analyst.he_rk, analyst.he_gk);
        // int N = 784; // 28x28
        // size_t rem = N % HHE.get_plain_size();
        // size_t num_block = N / HHE.get_plain_size();
        // if (rem)
        //     num_block++;
        // std::cout << "Remainder = " << rem << "\n";
        // std::cout << "Number of blocks = " << num_block << "\n";
        // std::vector<int> flatten_gks;
        // for (int i = 1; i < num_block; i++)
        //     flatten_gks.push_back(-(int)(i * HHE.get_plain_size()));
        // utils::print_vec(flatten_gks, flatten_gks.size(), "Flatten gks");

        // cipher.activate_bsgs(use_bsgs);
        // cipher.add_gk_indices();
        // cipher.add_some_gk_indices(flatten_gks);
        // if (use_bsgs)
        //     cipher.add_bsgs_indices(bsgs_n1, bsgs_n2);
        // else
        //     cipher.add_diagonal_indices(N);
        // cipher.create_gk();

        /*
        decomposing only 1 image
        c_ims_prime will be the HE encrypted images of client.mnistTestLabels
        */
        // std::vector<seal::Ciphertext> c_ims_prime = HHE.decomposition(client.c_ims[0], client.c_k, config::USE_BATCH);
        // std::cout << "One MNIST image is decomposed into = " << c_ims_prime.size() << " ciphertexts"
        //           << "\n";

        // utils::print_line(__LINE__);
        // std::cout << "Decomposition post processing (masking and flatenning)" << std::endl;
        // if (rem != 0)
        // {
        //     std::vector<uint64_t> mask(rem, 1);
        //     HHE.mask(c_ims_prime.back(), mask);
        //     utils::print_vec(mask, mask.size(), "Mask");
        // }

        // std::cout << "Flattening 7 ciphertexts into only 1" << std::endl;
        // Flatten(manually)
        // auto c_im_he = c_ims_prime[0];
        // seal::Ciphertext tmp;
        // for (size_t i = 1; i < c_ims_prime.size(); i++)
        // {
        //     analyst_he_eval.rotate_rows(c_ims_prime[i], -(int)(i * 128), analyst.he_gk, tmp);
        //     analyst_he_eval.add_inplace(c_im_he, tmp);
        // }
        // seal::Ciphertext c_im_he;
        // HHE.flatten(c_ims_prime, c_im_he, analyst.he_gk);

        // debugging: decrypt and check if the flattened result is correct
        // std::cout << "MNIST test image = ";
        // client.mnistTestData.printMat();
        // auto decrypted_im = sealhelper::decrypting(c_im_he, analyst.he_sk, analyst_he_benc,
        //                                            *context, 784);
        // utils::print_vec(decrypted_im, decrypted_im.size(), "Decrypted image");

        // int debug = 1;

        // for (auto c_im_he : c_ims_prime)
        // {
        //     auto decrypted_c_im = sealhelper::decrypting(c_im_he, analyst.he_sk, analyst_he_benc, *context, 128); // 112 = 784 / 7
        //     std::cout << "decrypted image size " << decrypted_c_im.size() << "\n";
        //     utils::print_vec(decrypted_c_im, decrypted_c_im.size(), "Decrypted image");
        // }
        // function to decompose many images
        // pastahelper::decomposition(HHE, client.c_k, client.c_ims, csp.c_prime, config::USE_BATCH);

        // std::cout << "CSP evaluates the HE encrypted neural network on the HE encrypted data" << std::endl;

        // utils::print_line(__LINE__);
        // std::cout << "---- Analyst ----" << std::endl;

        // std::cout << "Analyst decrypts the HE encrypted neural network output using his HE decryption secret key" << std::endl;

        return 0;
    }

    int hhe_pktnn_ecg_inference()
    {
        utils::print_example_banner("PocketHHE: Privacy-preserving Inference with a 1-layer Neural Network on Integer ECG dataset");

        // the actors in the protocol
        utils::print_line(__LINE__);
        std::cout << "---------------------- Create the parties ----------------------"
                  << "\n";
        std::cout << "Create analyst, data owner / client, server / csp \n";
        Analyst analyst;
        Client client;
        CSP csp;

        // The Analyst
        std::cout << "\n";
        utils::print_line(__LINE__);
        std::cout << "---------------------- Analyst ----------------------"
                  << "\n";
        std::cout << "Analyst constructs the HE context"
                  << "\n";
        std::shared_ptr<seal::SEALContext> context = sealhelper::get_seal_context();
        sealhelper::print_parameters(*context);
        utils::print_line(__LINE__);
        std::cout << "Analyst creates the HE keys, batch encoder, encryptor and evaluator from the context"
                  << "\n";
        seal::KeyGenerator keygen(*context);
        analyst.he_sk = keygen.secret_key();     // HE secret key for decryption
        keygen.create_public_key(analyst.he_pk); // HE public key for encryption
        keygen.create_relin_keys(analyst.he_rk); // HE relinearization key to reduce noise in ciphertexts
        seal::BatchEncoder analyst_he_benc(*context);
        bool use_bsgs = false;
        std::vector<int> gk_indices = pastahelper::add_gk_indices(use_bsgs, analyst_he_benc);
        keygen.create_galois_keys(gk_indices, analyst.he_gk); // the HE Galois keys for batch computation
        seal::Encryptor analyst_he_enc(*context, analyst.he_pk);
        seal::Evaluator analyst_he_eval(*context);
        seal::Decryptor analyst_he_dec(*context, analyst.he_sk);

        utils::print_line(__LINE__);
        std::cout << "Analyst creates the neural network, loads the pretrained weights and biases"
                  << "\n";
        pktnn::pktfc fc(128, 1);
        if (config::debugging)
        {
            fc.loadWeight("../" + config::save_weight_path);
            fc.loadBias("../" + config::save_bias_path);
        }
        else
        {
            fc.loadWeight(config::save_weight_path);
            fc.loadBias(config::save_bias_path);
        }
        fc.printWeightShape();
        fc.printBiasShape();
        // get the weights and biases to encrypt later
        pktnn::pktmat fc_weight;
        pktnn::pktmat fc_bias;
        fc_weight = fc.getWeight();
        fc_bias = fc.getBias();
        // fc_weight.printMat();
        // fc_bias.printMat();

        return 0;
    }

} // end of hhe_pktnn_examples namespace