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
        csp.enc_weight = analyst.enc_weight;
        csp.enc_bias = analyst.enc_bias;

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
        client.testData = mnistTestImages;
        client.testLabels = mnistTestLabels;

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
        client.cs = pastahelper::symmetric_encrypt(SymmetricEncryptor, client.testData); // the symmetric encrypted images
        std::cout << "Number of encrypted images = " << client.cs.size() << "\n";
        if (config::verbose)
        {
            client.testData.printMat();
            for (auto i : client.cs)
            {
                utils::print_vec(i, i.size(), "Encrypted image ");
            }
            auto dec_ims = pastahelper::symmetric_decrypt(SymmetricEncryptor, client.cs);
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
        result.matElemMulMat(weight_t_row, client.testData);
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
        c_ims_prime will be the HE encrypted images of client.testLabels
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
        // client.testData.printMat();
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
        Analyst analyst;
        Client client;
        CSP csp;

        // ---------------------- Analyst ----------------------
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
        analyst.weight = fc.getWeight();
        analyst.bias = fc.getBias();
        // divide bias by 128
        int old_bias = analyst.bias.getElem(0, 0);
        analyst.bias.setElem(0, 0, (int)old_bias / 128);
        // analyst.weight.printMat();
        // analyst.bias.printMat();

        utils::print_line(__LINE__);
        std::cout << "Analyst encrypts the weights and biases using HE"
                  << "\n";
        pktnn::pktmat fc_weight_t;
        fc_weight_t.transposeOf(analyst.weight);
        analyst.enc_weight = sealhelper::encrypt_weight(fc_weight_t,
                                                        analyst.he_pk,
                                                        analyst_he_benc,
                                                        analyst_he_enc);
        std::cout << "Encrypt the weight...";
        std::cout << "The encrypted weight vector has " << analyst.enc_weight.size() << " ciphertexts\n";
        ecg_test::test_encrypted_weight(analyst.enc_weight,
                                        fc_weight_t,
                                        analyst.he_sk,
                                        analyst_he_benc,
                                        analyst_he_dec,
                                        128);
        std::cout << "Encrypt the bias...";
        analyst.enc_bias = sealhelper::encrypt_bias(analyst.bias,
                                                    analyst.he_pk,
                                                    analyst_he_enc);
        std::cout << "The encrypted bias vector has " << analyst.enc_bias.size() << " ciphertexts\n";
        ecg_test::test_encrypted_bias(analyst.enc_bias,
                                      analyst.bias,
                                      analyst.he_sk,
                                      analyst_he_dec);

        utils::print_line(__LINE__);
        std::cout << "Analyst sends the HE keys (except the secret key) to the CSP..."
                  << "\n";
        csp.he_gk = analyst.he_gk;
        csp.he_pk = analyst.he_pk;
        csp.he_rk = analyst.he_rk;
        // calculate the commnication overhead (in MB)
        float he_pk_size = sealhelper::he_pk_key_size(analyst.he_pk, false);
        float he_keys_size = sealhelper::he_key_size(analyst.he_pk, analyst.he_rk, analyst.he_gk, true);

        std::cout << "Analyst sends the encrypted weight and bias to the CSP..."
                  << "\n";
        csp.enc_weight = analyst.enc_weight;
        csp.enc_bias = analyst.enc_bias;
        // calculate the size of encrypted weights and biases (in MB)
        float enc_weight_bias_size = sealhelper::enc_weight_bias_size(analyst.enc_weight, analyst.enc_bias, true, true);

        // ---------------------- Client (Data Owner) ----------------------
        std::cout << "\n";
        utils::print_line(__LINE__);
        std::cout << "---------------------- Client (Data Owner) ----------------------"
                  << "\n";

        utils::print_line(__LINE__);
        std::cout << "Client loads his ECG test data" << std::endl;
        int numTestSamples = 13245;
        pktnn::pktmat ecgTestLabels(numTestSamples, 1);
        pktnn::pktmat ecgTestInput(numTestSamples, 128);
        pktnn::pktloader::loadEcgData(ecgTestInput, "data/mit-bih/csv/mitbih_x_test_int.csv",
                                      numTestSamples, config::debugging);

        pktnn::pktloader::loadEcgLabels(ecgTestLabels, "data/mit-bih/csv/mitbih_bin_y_test.csv",
                                        numTestSamples, config::debugging);
        ecgTestLabels.selfMulConst(128); // scale the output from 0-1 to 0-128

        if (config::dry_run) // get a slice of dry_run data samples
        {
            int dryRunNumSamples = config::dry_run_num_samples;
            std::cout << "Dry run: get a slice of " << dryRunNumSamples << " data samples"
                      << "\n";
            client.testData.sliceOf(ecgTestInput, 0, dryRunNumSamples - 1, 0, 127);
            client.testLabels.sliceOf(ecgTestLabels, 0, dryRunNumSamples - 1, 0, 0);
        }
        else
        {
            client.testData = ecgTestInput;
            client.testLabels = ecgTestLabels;
        }
        std::cout << "Test data shape: ";
        client.testData.printShape();
        // client.testData.printMat();
        std::cout << "Test labels shape: ";
        client.testLabels.printShape();

        utils::print_line(__LINE__);
        std::cout << "Client creates the symmetric key" << std::endl;
        client.k = pastahelper::get_symmetric_key();
        std::cout << "Symmetric key size: " << client.k.size() << "\n";
        // utils::print_vec(client.k, client.k.size(), "Symmetric key: ");

        utils::print_line(__LINE__);
        std::cout << "Client encrypts his symmetric key using HE" << std::endl;
        client.c_k = pastahelper::encrypt_symmetric_key(client.k, config::USE_BATCH, analyst_he_benc, analyst_he_enc);

        utils::print_line(__LINE__);
        std::cout << "Client symmetrically encrypts his ECG data" << std::endl;
        pasta::PASTA SymmetricEncryptor(client.k, config::plain_mod);
        client.cs = pastahelper::symmetric_encrypt(SymmetricEncryptor, client.testData); // the symmetric encrypted images
        std::cout << "The symmetric encrypted data has " << client.cs.size() << " ciphertexts\n";

        utils::print_line(__LINE__);
        std::cout << "The client sends the symmetric encrypted data and the HE encrypted symmetric key to the CSP..."
                  << "\n";
        csp.c_k = client.c_k;
        csp.cs = client.cs;
        // calculate the size of the symmetric encrypted data and HE encrypted symmetric key (in MB)
        float sym_enc_data_size = pastahelper::sym_enc_data_size(client.cs, true);
        float he_enc_sym_key_size = sealhelper::he_vec_size(client.c_k, true, "HE encrypted symmetric key");

        // -------------------------- CSP (server) ----------------------
        std::cout << "\n";
        utils::print_line(__LINE__);
        std::cout << "-------------------------- CSP ----------------------" << std::endl;

        utils::print_line(__LINE__);
        std::cout << "CSP runs the decomposition algorithm to turn the symmetric encrypted data into HE encrypted data" << std::endl;
        seal::KeyGenerator csp_keygen(*context);
        csp.he_sk = csp_keygen.secret_key();
        // Below is to check if the csp key is different to the analyst key (they must be different)
        // csp.he_sk.save(std::cout);
        // std::cout << "\n";
        // analyst.he_sk.save(std::cout);
        // std::cout << "\n";
        pasta::PASTA_SEAL HHE(context, csp.he_pk, csp.he_sk, csp.he_rk, csp.he_gk);
        for (std::vector<uint64_t> c : csp.cs)
        {
            std::vector<seal::Ciphertext> c_prime = HHE.decomposition(c, csp.c_k, config::USE_BATCH);
            if (c_prime.size() == 1)
            {
                csp.c_primes.push_back(c_prime[0]);

                // --- for debugging: we decrypt the decomposed ciphertexts with the analyst's secret key
                // to check if the decryption is same as the plaintext data of the client
                // std::vector<int64_t> dec_c_prime = sealhelper::decrypting(c_prime[0], analyst.he_sk, analyst_he_benc, *context, 128);
                // utils::print_vec(dec_c_prime, dec_c_prime.size(), "decrypted c_prime ", "\n");
            }
            else
            {
                std::cout << "there are more than 1 seal ciphertexts in the each decomposed ciphertext\n";
                std::cout << "we need to do some post-processing\n";
            }
        }
        std::cout << "There are " << csp.c_primes.size() << " decomposed HE ciphertexts\n";

        utils::print_line(__LINE__);
        std::cout << "CSP evaluates the HE encrypted weights (& biases) on the HE encrypted data" << std::endl;
        for (seal::Ciphertext c_prime : csp.c_primes)
        {
            seal::Ciphertext enc_result;
            sealhelper::packed_enc_multiply(c_prime, csp.enc_weight[0], enc_result, analyst_he_eval);
            // we only do element-wise multiplication for now and ignore the
            // bias for simplication as it does not affect the result
            csp.enc_results.push_back(enc_result);
        }

        utils::print_line(__LINE__);
        std::cout << "CSP sends the HE encrypted result to the analyst" << std::endl;
        analyst.enc_results = csp.enc_results;
        float enc_results_size = sealhelper::he_vec_size(analyst.enc_results, true, "HE encrypted results");

        // ---------------------- Analyst (again) ----------------------
        std::cout << "\n";
        utils::print_line(__LINE__);
        std::cout << "---------------------- Analyst ----------------------"
                  << "\n";
        utils::print_line(__LINE__);
        std::cout << "The analyst decrypts the HE encrypted results received from the CSP" << std::endl;
        for (seal::Ciphertext enc_result : analyst.enc_results)
        {
            std::vector<int64_t> dec_result = sealhelper::decrypting(enc_result,
                                                                     analyst.he_sk,
                                                                     analyst_he_benc,
                                                                     *context,
                                                                     128);
            analyst.dec_results.push_back(dec_result);
            // utils::print_vec(dec_result, dec_result.size(), "decrypted result ", ", ");
        }

        utils::print_line(__LINE__);
        std::cout << "The analyst applies the non-linear operations on the decrypted results and get the final predictions" << std::endl;
        for (std::vector<int64_t> dec_result : analyst.dec_results)
        {
            // first find the sum of the decrypted results
            int sum = 0;
            for (auto i : dec_result)
            {
                sum += i;
            }
            // std::cout << "sum = " << sum << "\n";
            // apply the pocket sigmoid function
            int out = utils::simple_pocket_sigmoid(sum);
            // the final prediction
            int final_pred = 0;
            out > 64 ? final_pred = 128 : final_pred = 0;
            // add the prediction to the analyst's predictions
            analyst.predictions.push_back(final_pred);
        }

        // find the accuracy
        int testNumCorrect = 0;
        for (int i = 0; i < analyst.predictions.size(); ++i)
        {
            if (config::verbose)
                std::cout << "Prediction = " << analyst.predictions[i]
                          << "| Actual = " << client.testLabels.getElem(i, 0) << "\n";
            if (client.testLabels.getElem(i, 0) == analyst.predictions[i])
            {
                ++testNumCorrect;
            }
        }

        // ---------------------- Experiment results ----------------------
        std::cout << "\n";
        utils::print_line(__LINE__);
        std::cout << "---------------------- Experiment Results ----------------------"
                  << "\n";
        std::cout << "Final correct predions = " << testNumCorrect << " (out of "
                  << analyst.predictions.size() << " total examples)"
                  << "\n";
        std::cout << "Encrypted accuracy = "
                  << (double)testNumCorrect / analyst.predictions.size() * 100 << "% \n";
        // print out communication and computation costs here
        utils::print_line(__LINE__);
        std::cout << "Communication cost: " << std::endl;
        std::cout << "Client - CSP: " << sym_enc_data_size + he_enc_sym_key_size << " (Mb)" << std::endl;
        std::cout << "Client - Analyst: " << he_pk_size << " (Mb)" << std::endl;
        std::cout << "Analyst - CSP: " << he_keys_size + enc_weight_bias_size + enc_results_size << " (Mb)" << std::endl;
        float total_comm = sym_enc_data_size + he_enc_sym_key_size + he_pk_size +
                           he_keys_size + enc_weight_bias_size + enc_results_size;
        std::cout << "Total communication cost: " << total_comm << " (Mb)" << std::endl;

        utils::print_line(__LINE__);
        std::cout << "Computation cost: " << std::endl;
        std::cout << "Client: " << 0 << " (ms)" << std::endl;
        std::cout << "Analyst: " << 0 << " (ms)" << std::endl;
        std::cout << "CSP: " << 0 << " (ms)" << std::endl;

        return 0;
    }

} // end of hhe_pktnn_examples namespace