#include "pktnn_examples.h"

int fc_int_bp_simple()
{
    utils::print_example_banner("PocketNN: Simple training with dummy data using backpropagation");

    // constructing the neural net
    const int dim1 = 3;
    const int dim2 = 5;
    const int numEpochs = 3;

    pktnn::pktmat x(1, dim1);
    pktnn::pktfc fc1(dim1, dim2);
    pktnn::pktfc fc2(dim2, 1);

    std::cout << "--- Weights (first layer) before initialization --- \n";
    fc1.printWeight(std::cout);
    // fc2.printWeight(std::cout);

    // initialize the weights and biases
    fc1.useDfa(false).initHeWeightBias().setActv(pktnn::pktactv::Actv::pocket_tanh).setNextLayer(fc2);
    fc2.useDfa(false).initHeWeightBias().setActv(pktnn::pktactv::Actv::as_is).setPrevLayer(fc1);

    std::cout << "--- Weights after initialization --- \n";
    fc1.printWeight(std::cout);
    // fc2.printWeight(std::cout);

    // dummy data
    std::cout << "--- Data --- \n";
    x.setElem(0, 0, 10);
    x.setElem(0, 1, 20);
    x.setElem(0, 2, 30);
    std::cout << "x = ";
    x.printMat(std::cout);
    int y = 551; // random number
    std::cout << "y = " << y << "\n";

    std::cout << "--- Training --- \n";
    for (int i = 0; i < numEpochs; ++i)
    {
        fc1.forward(x);
        fc2.forward(fc1);

        int y_hat = fc2.mOutput.getElem(0, 0);
        int loss = pktnn::pktloss::scalarL2Loss(y, y_hat);
        int lossDelta = pktnn::pktloss::scalarL2LossDelta(y, y_hat);
        std::cout << "y: " << y << ", y_hat: " << y_hat << ", l2 loss: " << loss << ", l2 loss delta: " << lossDelta << "\n";

        pktnn::pktmat lossDeltaMat;
        lossDeltaMat.resetZero(1, 1).setElem(0, 0, lossDelta);

        fc2.backward(lossDeltaMat, 1e5);
    }

    std::cout << "--- Weights after training --- \n";
    fc1.printWeight(std::cout);
    // fc2.printWeight(std::cout);

    return 0;
};

int fc_int_dfa_mnist()
{
    utils::print_example_banner("PocketNN: Training on MNIST using direct feedback alignment with a 3-layer FC network");

    // Loading the MNIST dataset
    std::cout << "----- Loading MNIST data ----- \n";
    int numTrainSamples = 60000;
    int numTestSamples = 10000;

    pktnn::pktmat mnistTrainLabels;
    pktnn::pktmat mnistTrainImages;
    pktnn::pktmat mnistTestLabels;
    pktnn::pktmat mnistTestImages;

    pktnn::pktloader::loadMnistImages(mnistTrainImages, numTrainSamples, true); // numTrainSamples x (28*28)
    pktnn::pktloader::loadMnistLabels(mnistTrainLabels, numTrainSamples, true); // numTrainSamples x 1
    pktnn::pktloader::loadMnistImages(mnistTestImages, numTestSamples, false);  // numTestSamples x (28*28)
    pktnn::pktloader::loadMnistLabels(mnistTestLabels, numTestSamples, false);  // numTestSamples x 1
    std::cout << "Loaded train images: " << mnistTrainImages.rows() << ".\nLoaded test images: " << mnistTestImages.rows() << "\n";

    // Defining the network
    std::cout << "----- Defining the neural net ----- \n";
    pktnn::pktactv::Actv a = pktnn::pktactv::Actv::pocket_tanh;
    pktnn::pktfc fc1(config::dim_input, config::dim_layer1);
    pktnn::pktfc fc2(config::dim_layer1, config::dim_layer2);
    pktnn::pktfc fcLast(config::dim_layer2, config::num_classes);
    fc1.useDfa(true).setActv(a).setNextLayer(fc2);
    fc2.useDfa(true).setActv(a).setNextLayer(fcLast);
    fcLast.useDfa(true).setActv(a);

    // Initial stats before training
    pktnn::pktmat trainTargetMat(numTrainSamples, config::num_classes);
    pktnn::pktmat testTargetMat(numTestSamples, config::num_classes);

    int numCorrect = 0;
    fc1.forward(mnistTrainImages);
    for (int r = 0; r < numTrainSamples; ++r)
    {
        trainTargetMat.setElem(r, mnistTrainLabels.getElem(r, 0), pktnn::UNSIGNED_4BIT_MAX);
        if (trainTargetMat.getMaxIndexInRow(r) == fcLast.mOutput.getMaxIndexInRow((r)))
        {
            ++numCorrect;
        }
    }
    std::cout << "Initial training numCorrect: " << numCorrect << " / 60000"
              << "\n";

    numCorrect = 0;
    fc1.forward(mnistTestImages);
    for (int r = 0; r < numTestSamples; ++r)
    {
        testTargetMat.setElem(r, mnistTestLabels.getElem(r, 0), pktnn::UNSIGNED_4BIT_MAX);
        if (testTargetMat.getMaxIndexInRow(r) == fcLast.mOutput.getMaxIndexInRow((r)))
        {
            ++numCorrect;
        }
    }
    std::cout << "Initial test numCorrect: " << numCorrect << " / 10000"
              << "\n";

    // Training
    std::cout << "----- Start training -----\n";
    pktnn::pktmat lossMat;
    pktnn::pktmat lossDeltaMat;
    pktnn::pktmat batchLossDeltaMat;
    pktnn::pktmat miniBatchImages;
    pktnn::pktmat miniBatchTrainTargets;

    int epoch = 3;
    int miniBatchSize = 20; // CAUTION: Too big minibatch size can cause overflow
    int lrInv = 1000;
    std::cout << "Learning Rate Inverse = " << lrInv << ", numTrainSamples = " << numTrainSamples << ", miniBatchSize = " << miniBatchSize << ", numEpochs = " << epoch << "\n";

    // random indices template
    int *indices = new int[numTrainSamples];
    for (int i = 0; i < numTrainSamples; ++i)
    {
        indices[i] = i;
    }

    std::string testCorrect = "";
    std::cout << "Epoch | SumLoss | NumCorrect | Accuracy\n";
    for (int e = 1; e <= epoch; ++e)
    {
        // Shuffle the indices
        for (int i = numTrainSamples - 1; i > 0; --i)
        {
            int j = rand() % (i + 1); // Pick a random index from 0 to r
            int temp = indices[j];
            indices[j] = indices[i];
            indices[i] = temp;
        }

        if ((e % 10 == 0) && (lrInv < 2 * lrInv))
        {
            // reducing the learning rate by a half every 5 epochs
            // avoid overflow
            lrInv *= 2;
        }

        int sumLoss = 0;
        int sumLossDelta = 0;
        int epochNumCorrect = 0;
        int numIter = numTrainSamples / miniBatchSize;

        for (int i = 0; i < numIter; ++i)
        {
            int batchNumCorrect = 0;
            const int idxStart = i * miniBatchSize;
            const int idxEnd = idxStart + miniBatchSize;
            miniBatchImages.indexedSlicedSamplesOf(mnistTrainImages, indices, idxStart, idxEnd);
            miniBatchTrainTargets.indexedSlicedSamplesOf(trainTargetMat, indices, idxStart, idxEnd);

            // miniBatchImages.printMat(std::cout); // print out the input data

            fc1.forward(miniBatchImages);
            sumLoss += pktnn::pktloss::batchL2Loss(lossMat, miniBatchTrainTargets, fcLast.mOutput);
            sumLossDelta = pktnn::pktloss::batchL2LossDelta(lossDeltaMat, miniBatchTrainTargets, fcLast.mOutput);

            for (int r = 0; r < miniBatchSize; ++r)
            {
                if (miniBatchTrainTargets.getMaxIndexInRow(r) == fcLast.mOutput.getMaxIndexInRow((r)))
                {
                    ++batchNumCorrect;
                }
            }
            fcLast.backward(lossDeltaMat, lrInv);
            epochNumCorrect += batchNumCorrect;
        }
        std::cout << e << " | " << sumLoss << " | " << epochNumCorrect << " | " << (epochNumCorrect * 1.0 / numTrainSamples) << "\n";

        // check the test set accuracy
        fc1.forward(mnistTestImages);
        int testNumCorrect = 0;
        for (int r = 0; r < numTestSamples; ++r)
        {
            if (testTargetMat.getMaxIndexInRow(r) == fcLast.mOutput.getMaxIndexInRow((r)))
            {
                ++testNumCorrect;
            }
        }
        testCorrect += (std::to_string(e) + "," + std::to_string(testNumCorrect) + "\n");
    }

    fc1.forward(mnistTrainImages);
    numCorrect = 0;
    for (int r = 0; r < numTrainSamples; ++r)
    {
        if (trainTargetMat.getMaxIndexInRow(r) == fcLast.mOutput.getMaxIndexInRow((r)))
        {
            ++numCorrect;
        }
    }
    std::cout << "Final training numCorrect = " << numCorrect << "\n";
    // fc1.printBias(std::cout);
    // fcLast.printWeight(std::cout);
    // fcLast.printBias(std::cout);

    std::cout << "----- Test -----\n";
    fc1.forward(mnistTestImages);
    numCorrect = 0;
    for (int r = 0; r < numTestSamples; ++r)
    {
        if (testTargetMat.getMaxIndexInRow(r) == fcLast.mOutput.getMaxIndexInRow((r)))
        {
            ++numCorrect;
        }
    }
    std::cout << "Epoch | NumCorrect\n";
    std::cout << testCorrect;
    std::cout << "Final test numCorrect = " << numCorrect << "\n";
    std::cout << "Final test accuracy = " << (numCorrect * 1.0 / numTestSamples) << "\n";
    std::cout << "Final learning rate inverse = " << lrInv << "\n";

    std::cout << "----- Save weights and biases after training -----\n";
    // fc1.saveWeight("weights/3_layer/fc1_weight.csv");
    // fc1.saveBias("weights/3_layer/fc1_bias.csv");
    // fc2.saveWeight("weights/3_layer/fc2_weight.csv");
    // fc2.saveBias("weights/3_layer/fc2_bias.csv");
    // fcLast.saveWeight("weights/3_layer/fcLast_weight.csv");
    // fcLast.saveBias("weights/3_layer/fcLast_bias.csv");

    delete[] indices;

    return 0;
};

int fc_int_dfa_mnist_inference()
{
    utils::print_example_banner("PocketNN: Inference on MNIST using pretrained weights");
    std::cout << "----- Constructing the network -----\n";
    int numClasses = 10;
    int mnistRows = 28;
    int mnistCols = 28;

    const int dimInput = mnistRows * mnistCols;
    const int dim1 = 100;
    const int dim2 = 50;
    pktnn::pktactv::Actv a = pktnn::pktactv::Actv::pocket_tanh;

    pktnn::pktfc fc1(dimInput, dim1);
    pktnn::pktfc fc2(dim1, dim2);
    pktnn::pktfc fcLast(dim2, numClasses);
    fc1.useDfa(true).setActv(a).setNextLayer(fc2);
    fc2.useDfa(true).setActv(a).setNextLayer(fcLast);
    fcLast.useDfa(true).setActv(a);
    std::cout << "Constructing a 3-layer fully connected neural network done!\n";

    std::cout << "----- Loading the MNIST test data -----\n";
    int numTestSamples = 10000;
    pktnn::pktmat mnistTestLabels;
    pktnn::pktmat mnistTestImages;
    pktnn::pktloader::loadMnistImages(mnistTestImages, numTestSamples, false); // numTestSamples x (28*28)
    pktnn::pktloader::loadMnistLabels(mnistTestLabels, numTestSamples, false); // numTestSamples x 1
    std::cout << "Loaded test images: " << mnistTestImages.rows() << "\n";

    std::cout << "----- Initial Test Before Loading Weights -----\n";
    pktnn::pktmat testTargetMat(numTestSamples, numClasses);
    int numCorrect = 0;
    fc1.forward(mnistTestImages);
    for (int r = 0; r < numTestSamples; ++r)
    {
        testTargetMat.setElem(r, mnistTestLabels.getElem(r, 0), pktnn::UNSIGNED_4BIT_MAX);
        if (testTargetMat.getMaxIndexInRow(r) == fcLast.mOutput.getMaxIndexInRow((r)))
        {
            ++numCorrect;
        }
    }
    std::cout << "Initial test numCorrect: " << numCorrect << " / 10000"
              << "\n";
    std::cout << "Initial test accuracy = " << (numCorrect * 1.0 / numTestSamples) << "\n";
    // testTargetMat.printMat(std::cout);

    std::cout << "----- Loading weights and biases -----\n";
    fc1.loadWeight("weights/3_layer/fc1_weight.csv");
    fc1.loadBias("weights/3_layer/fc1_bias.csv");
    fc2.loadWeight("weights/3_layer/fc2_weight.csv");
    fc2.loadBias("weights/3_layer/fc2_bias.csv");
    fcLast.loadWeight("weights/3_layer/fcLast_weight.csv");
    fcLast.loadBias("weights/3_layer/fcLast_bias.csv");
    std::cout << "fc1: \n";
    // fc1.printBias(std::cout);
    // fc1.printWeight(std::cout);
    fc1.printWeightShape(std::cout);
    fc1.printBiasShape(std::cout);
    std::cout << "fc2: \n";
    fc2.printWeightShape(std::cout);
    fc2.printBiasShape(std::cout);
    std::cout << "fcLast: \n";
    fcLast.printWeightShape(std::cout);
    fcLast.printBiasShape(std::cout);
    // fcLast.printWeight(std::cout);

    std::cout << "----- Test -----\n";
    // pktnn::pktmat testTargetMat(numTestSamples, numClasses);
    fc1.forward(mnistTestImages);
    int testCorrect = 0;
    for (int r = 0; r < numTestSamples; ++r)
    {
        if (testTargetMat.getMaxIndexInRow(r) == fcLast.mOutput.getMaxIndexInRow((r)))
        {
            ++testCorrect;
        }
    }
    std::cout << "Final test numCorrect = " << testCorrect << "\n";
    std::cout << "Final test accuracy = " << (testCorrect * 1.0 / numTestSamples) << "\n";

    return 0;
}

int fc_int_dfa_mnist_one_layer()
{
    utils::print_example_banner("PocketNN: Training on MNIST using direct feedback alignment with a 1-layer FC network");

    // Loading the MNIST dataset
    std::cout << "----- Loading MNIST data ----- \n";
    int numTrainSamples = 60000;
    int numTestSamples = 10000;

    pktnn::pktmat mnistTrainLabels;
    pktnn::pktmat mnistTrainImages;
    pktnn::pktmat mnistTestLabels;
    pktnn::pktmat mnistTestImages;

    pktnn::pktloader::loadMnistImages(mnistTrainImages, numTrainSamples, true); // numTrainSamples x (28*28)
    pktnn::pktloader::loadMnistLabels(mnistTrainLabels, numTrainSamples, true); // numTrainSamples x 1
    pktnn::pktloader::loadMnistImages(mnistTestImages, numTestSamples, false);  // numTestSamples x (28*28)
    pktnn::pktloader::loadMnistLabels(mnistTestLabels, numTestSamples, false);  // numTestSamples x 1
    std::cout << "Loaded train images: " << mnistTrainImages.rows() << ".\nLoaded test images: " << mnistTestImages.rows() << "\n";

    // Defining the network
    std::cout << "----- Defining the neural net ----- \n";
    pktnn::pktactv::Actv a = pktnn::pktactv::Actv::pocket_tanh;
    pktnn::pktfc fc1(config::dim_input, config::num_classes);
    fc1.useDfa(true).setActv(a);

    // Initial stats before training
    pktnn::pktmat trainTargetMat(numTrainSamples, config::num_classes);
    pktnn::pktmat testTargetMat(numTestSamples, config::num_classes);

    int numCorrect = 0;
    fc1.forward(mnistTrainImages);
    for (int r = 0; r < numTrainSamples; ++r)
    {
        trainTargetMat.setElem(r, mnistTrainLabels.getElem(r, 0), pktnn::UNSIGNED_4BIT_MAX);
        if (trainTargetMat.getMaxIndexInRow(r) == fc1.mOutput.getMaxIndexInRow((r)))
        {
            ++numCorrect;
        }
    }
    std::cout << "Initial training numCorrect: " << numCorrect << " / 60000"
              << "\n";

    numCorrect = 0;
    fc1.forward(mnistTestImages);
    for (int r = 0; r < numTestSamples; ++r)
    {
        testTargetMat.setElem(r, mnistTestLabels.getElem(r, 0), pktnn::UNSIGNED_4BIT_MAX);
        if (testTargetMat.getMaxIndexInRow(r) == fc1.mOutput.getMaxIndexInRow((r)))
        {
            ++numCorrect;
        }
    }
    std::cout << "Initial test numCorrect: " << numCorrect << " / 10000"
              << "\n";

    // Training
    std::cout << "----- Start training -----\n";
    pktnn::pktmat lossMat;
    pktnn::pktmat lossDeltaMat;
    pktnn::pktmat batchLossDeltaMat;
    pktnn::pktmat miniBatchImages;
    pktnn::pktmat miniBatchTrainTargets;

    std::cout << "Learning Rate Inverse = " << config::lr_inv << ", numTrainSamples = " << numTrainSamples << ", miniBatchSize = " << config::mini_batch_size << ", numEpochs = " << config::epoch << "\n";

    // random indices template
    int *indices = new int[numTrainSamples];
    for (int i = 0; i < numTrainSamples; ++i)
    {
        indices[i] = i;
    }

    std::string testCorrect = "";
    std::cout << "Epoch | SumLoss | NumCorrect | Accuracy\n";
    for (int e = 1; e <= config::epoch; ++e)
    {
        // Shuffle the indices
        for (int i = numTrainSamples - 1; i > 0; --i)
        {
            int j = rand() % (i + 1); // Pick a random index from 0 to r
            int temp = indices[j];
            indices[j] = indices[i];
            indices[i] = temp;
        }

        if ((e % 10 == 0) && (config::lr_inv < 2 * config::lr_inv))
        {
            // reducing the learning rate by a half every 5 epochs
            // avoid overflow
            config::lr_inv *= 2;
        }

        int sumLoss = 0;
        int sumLossDelta = 0;
        int epochNumCorrect = 0;
        int numIter = numTrainSamples / config::mini_batch_size;

        for (int i = 0; i < numIter; ++i)
        {
            int batchNumCorrect = 0;
            const int idxStart = i * config::mini_batch_size;
            const int idxEnd = idxStart + config::mini_batch_size;
            miniBatchImages.indexedSlicedSamplesOf(mnistTrainImages, indices, idxStart, idxEnd);
            miniBatchTrainTargets.indexedSlicedSamplesOf(trainTargetMat, indices, idxStart, idxEnd);

            // miniBatchImages.printMat(std::cout); // print out the input data

            fc1.forward(miniBatchImages);
            sumLoss += pktnn::pktloss::batchL2Loss(lossMat, miniBatchTrainTargets, fc1.mOutput);
            sumLossDelta = pktnn::pktloss::batchL2LossDelta(lossDeltaMat, miniBatchTrainTargets, fc1.mOutput);

            for (int r = 0; r < config::mini_batch_size; ++r)
            {
                if (miniBatchTrainTargets.getMaxIndexInRow(r) == fc1.mOutput.getMaxIndexInRow((r)))
                {
                    ++batchNumCorrect;
                }
            }
            fc1.backward(lossDeltaMat, config::lr_inv);
            epochNumCorrect += batchNumCorrect;
        }
        std::cout << e << " | " << sumLoss << " | " << epochNumCorrect << " | " << (epochNumCorrect * 1.0 / numTrainSamples) << "\n";

        // check the test set accuracy
        fc1.forward(mnistTestImages);
        int testNumCorrect = 0;
        for (int r = 0; r < numTestSamples; ++r)
        {
            if (testTargetMat.getMaxIndexInRow(r) == fc1.mOutput.getMaxIndexInRow((r)))
            {
                ++testNumCorrect;
            }
        }
        testCorrect += (std::to_string(e) + "," + std::to_string(testNumCorrect) + "\n");
    }

    fc1.forward(mnistTrainImages);
    numCorrect = 0;
    for (int r = 0; r < numTrainSamples; ++r)
    {
        if (trainTargetMat.getMaxIndexInRow(r) == fc1.mOutput.getMaxIndexInRow((r)))
        {
            ++numCorrect;
        }
    }
    std::cout << "Final training numCorrect = " << numCorrect << "\n";

    std::cout << "----- Test -----\n";
    fc1.forward(mnistTestImages);
    numCorrect = 0;
    for (int r = 0; r < numTestSamples; ++r)
    {
        if (testTargetMat.getMaxIndexInRow(r) == fc1.mOutput.getMaxIndexInRow((r)))
        {
            ++numCorrect;
        }
    }
    std::cout << "Epoch | NumCorrect\n";
    std::cout << testCorrect;
    std::cout << "Final test numCorrect = " << numCorrect << "\n";
    std::cout << "Final test accuracy = " << (numCorrect * 1.0 / numTestSamples) << "\n";
    std::cout << "Final learning rate inverse = " << config::lr_inv << "\n";

    std::cout << "----- Save weights and biases after training -----\n";
    fc1.saveWeight("weights/1_layer/fc1_weight_50epochs.csv");
    fc1.saveBias("weights/1_layer/fc1_bias_50epochs.csv");

    delete[] indices;

    return 0;
}

int fc_int_dfa_mnist_one_layer_inference()
{
    utils::print_example_banner("PocketNN: Inference on MNIST using pretrained weights for the 1-layer network");
    std::cout << "----- Constructing the network -----\n";
    pktnn::pktactv::Actv a = pktnn::pktactv::Actv::pocket_tanh;
    pktnn::pktfc fc1(config::dim_input, config::dim_layer1);
    fc1.useDfa(true).setActv(a);
    std::cout << "Constructing the 1-layer fully connected neural network done!\n";

    std::cout << "----- Loading the MNIST test data -----\n";
    int numTestSamples = 10000;
    pktnn::pktmat mnistTestLabels;
    pktnn::pktmat mnistTestImages;
    pktnn::pktloader::loadMnistImages(mnistTestImages, numTestSamples, false); // numTestSamples x (28*28)
    pktnn::pktloader::loadMnistLabels(mnistTestLabels, numTestSamples, false); // numTestSamples x 1
    std::cout << "Loaded test images: " << mnistTestImages.rows() << "\n";
    // mnistTestImages.printMat();

    std::cout << "----- Initial Test Before Loading Weights -----\n";
    pktnn::pktmat testTargetMat(numTestSamples, config::num_classes);
    int numCorrect = 0;
    fc1.forward(mnistTestImages);
    for (int r = 0; r < numTestSamples; ++r)
    {
        testTargetMat.setElem(r, mnistTestLabels.getElem(r, 0), pktnn::UNSIGNED_4BIT_MAX);
        if (testTargetMat.getMaxIndexInRow(r) == fc1.mOutput.getMaxIndexInRow((r)))
        {
            ++numCorrect;
        }
    }
    std::cout << "Initial test numCorrect: " << numCorrect << " / 10000"
              << "\n";
    std::cout << "Initial test accuracy = " << (numCorrect * 1.0 / numTestSamples) << "\n";
    // testTargetMat.printMat(std::cout);

    std::cout << "----- Loading weights and biases -----\n";
    fc1.loadWeight("weights/1_layer/fc1_weight_50epochs.csv");
    fc1.loadBias("weights/1_layer/fc1_bias_50epochs.csv");
    fc1.printWeightShape(std::cout);
    fc1.printBiasShape(std::cout);

    std::cout << "----- Test -----\n";
    fc1.forward(mnistTestImages);
    int testCorrect = 0;
    for (int r = 0; r < numTestSamples; ++r)
    {
        if (testTargetMat.getMaxIndexInRow(r) == fc1.mOutput.getMaxIndexInRow((r)))
        {
            ++testCorrect;
        }
    }
    std::cout << "Final test numCorrect = " << testCorrect << "\n";
    std::cout << "Final test accuracy = " << (testCorrect * 1.0 / numTestSamples) << "\n";

    return 0;
}

int fc_int_dfa_ecg_one_layer()
{
    utils::print_example_banner("PocketNN: Training on MIT-BIH using direct feedback alignment with a 1-layer FC network");

    // Loading the ECG dataset
    std::cout << "----- Loading MIT-BIH ECG data ----- \n";
    int numTrainSamples = 13245;
    int numTestSamples = 13245;

    pktnn::pktmat ecgTrainLabels(numTrainSamples, 1);
    pktnn::pktmat ecgTrainInput(numTrainSamples, 128);
    pktnn::pktmat ecgTestLabels(numTestSamples, 1);
    pktnn::pktmat ecgTestInput(numTestSamples, 128);

    pktnn::pktloader::loadEcgData(ecgTrainInput, "data/mit-bih/csv/mitbih_x_train_int.csv",
                                  numTrainSamples, config::debugging);
    ecgTrainInput.printShape();
    pktnn::pktloader::loadEcgLabels(ecgTrainLabels, "data/mit-bih/csv/mitbih_bin_y_train.csv",
                                    numTrainSamples, config::debugging);
    ecgTrainLabels.selfMulConst(128); // scale the output from 0-1 to 0-128 due to PocketNN's sigmoid function
    ecgTrainLabels.printShape();

    pktnn::pktloader::loadEcgData(ecgTestInput, "data/mit-bih/csv/mitbih_x_test_int.csv",
                                  numTestSamples, config::debugging);
    ecgTestInput.printShape();
    pktnn::pktloader::loadEcgLabels(ecgTestLabels, "data/mit-bih/csv/mitbih_bin_y_test.csv",
                                    numTestSamples, config::debugging);
    ecgTestLabels.selfMulConst(128); // scale the output from 0-1 to 0-128
    ecgTestLabels.printShape();

    // Defining the network
    std::cout << "----- Defining the neural net ----- \n";
    pktnn::pktactv::Actv a = pktnn::pktactv::Actv::pocket_sigmoid;
    pktnn::pktfc fc1(128, 1);
    fc1.useDfa(true).setActv(a);
    std::cout << "Neural network with 1 fc layer and pocket_sigmoid activation\n";

    // Initialy weights and biases are 0s
    // fc1.printWeight(std::cout);
    // fc1.printBias(std::cout);

    // Initial stats before training
    std::cout << "----- Initial stats before training ----- \n";
    int numCorrect = 0;
    fc1.forward(ecgTrainInput);
    // fc1.printOutput();
    for (int r = 0; r < numTrainSamples; ++r)
    {
        int output_row_r = 0;
        fc1.mOutput.getElem(r, 0) > 64 ? output_row_r = 128 : output_row_r = 0;
        if (ecgTrainLabels.getElem(r, 0) == output_row_r)
        {
            ++numCorrect;
        }
    }
    std::cout << "Initial training numCorrect: " << numCorrect << " / " << numTrainSamples
              << "\n";
    std::cout << "Initial training accuracy: " << (numCorrect * 1.0 / numTrainSamples) << "\n";

    numCorrect = 0;
    fc1.forward(ecgTestInput);
    for (int r = 0; r < numTestSamples; ++r)
    {
        int output_row_r = 0;
        fc1.mOutput.getElem(r, 0) > 64 ? output_row_r = 128 : output_row_r = 0;
        if (ecgTestLabels.getElem(r, 0) == output_row_r)
        {
            ++numCorrect;
        }
    }
    std::cout << "Initial test numCorrect: " << numCorrect << " / " << numTestSamples
              << "\n";
    std::cout << "Initial test accuracy: " << (numCorrect * 1.0 / numTestSamples) << "\n";

    // Training
    std::cout << "----- Start training -----\n";
    pktnn::pktmat lossMat;
    pktnn::pktmat lossDeltaMat;
    pktnn::pktmat batchLossDeltaMat;
    pktnn::pktmat miniBatchInput;
    pktnn::pktmat miniBatchTrainTargets;

    std::cout << "Learning Rate Inverse = " << config::lr_inv
              << ", numTrainSamples = " << numTrainSamples << ", miniBatchSize = "
              << config::mini_batch_size << ", numEpochs = " << config::epoch
              << ", weight lower bound = " << config::weight_lower_bound
              << ", weight upper bound = " << config::weight_upper_bound << "\n";

    // random indices template
    int *indices = new int[numTrainSamples];
    for (int i = 0; i < numTrainSamples; ++i)
    {
        indices[i] = i;
    }

    float best_train_acc = 0.0;
    float best_test_acc = 0.0;
    int best_train_epoch = 0;
    int best_test_epoch = 0;
    std::string testCorrect = "";
    std::cout << "Epoch | SumLoss | NumCorrect | Accuracy\n";
    for (int e = 1; e <= config::epoch; ++e)
    {
        // Shuffle the indices
        for (int i = numTrainSamples - 1; i > 0; --i)
        {
            int j = rand() % (i + 1); // Pick a random index from 0 to r
            int temp = indices[j];
            indices[j] = indices[i];
            indices[i] = temp;
        }

        if ((e % 10 == 0) && (config::lr_inv < 2 * config::lr_inv))
        {
            // reducing the learning rate by a half every 5 epochs to avoid overflow
            config::lr_inv *= 2;
        }

        int sumLoss = 0;
        int sumLossDelta = 0;
        int epochNumCorrect = 0;
        int numIter = numTrainSamples / config::mini_batch_size;

        // The training loop
        for (int i = 0; i < numIter; ++i)
        {
            int batchNumCorrect = 0;
            const int idxStart = i * config::mini_batch_size;
            const int idxEnd = idxStart + config::mini_batch_size;
            miniBatchInput.indexedSlicedSamplesOf(ecgTrainInput, indices, idxStart, idxEnd);
            miniBatchTrainTargets.indexedSlicedSamplesOf(ecgTrainLabels, indices, idxStart, idxEnd);

            fc1.forward(miniBatchInput);
            sumLoss += pktnn::pktloss::batchL2Loss(lossMat, miniBatchTrainTargets, fc1.mOutput);
            sumLossDelta = pktnn::pktloss::batchL2LossDelta(lossDeltaMat, miniBatchTrainTargets, fc1.mOutput);

            // calculate the number of correct predictions in the batch
            for (int r = 0; r < config::mini_batch_size; ++r)
            {
                int output_row_r = 0;
                fc1.mOutput.getElem(r, 0) > 64 ? output_row_r = 128 : output_row_r = 0;
                // std::cout << fc1.mOutput.getElem(r, 0) << "----" << output_row_r << " ";
                if (miniBatchTrainTargets.getElem(r, 0) == output_row_r)
                {
                    ++batchNumCorrect;
                }
            }
            epochNumCorrect += batchNumCorrect;
            // the backward pass to calculate the gradients
            fc1.backward(lossDeltaMat, config::lr_inv, config::weight_lower_bound, config::weight_upper_bound);
        }
        float train_acc = epochNumCorrect * 1.0 / numTrainSamples;
        if (train_acc > best_train_acc)
        {
            best_train_acc = train_acc;
            best_train_epoch = e;
            // std::cout << "found best train accuracy = " << best_train_acc << " at epoch " << e << "\n";
        }
        std::cout << e << " | " << sumLoss << " | " << epochNumCorrect << " | " << train_acc << "\n";

        // after training through the whole dataset, check the test set accuracy
        fc1.forward(ecgTestInput);
        int testNumCorrect = 0;
        for (int r = 0; r < numTestSamples; ++r)
        {
            int output_row_r = 0;
            fc1.mOutput.getElem(r, 0) > 64 ? output_row_r = 128 : output_row_r = 0;

            if (ecgTestLabels.getElem(r, 0) == output_row_r)
            {
                ++testNumCorrect;
            }
        }
        float test_acc = testNumCorrect * 1.0 / numTestSamples;
        testCorrect += (std::to_string(e) + " | " + std::to_string(testNumCorrect) + " | " + std::to_string(test_acc)) + "\n";
        if (test_acc > best_test_acc)
        {
            best_test_acc = test_acc;
            best_test_epoch = e;
            testCorrect += "found best test accuracy = " + std::to_string(best_test_acc) + " at epoch " + std::to_string(e) + ". ";
            testCorrect += "save weights to " + config::save_weight_path + "\n";
            fc1.saveWeight(config::save_weight_path);
            fc1.saveBias(config::save_bias_path);
        }
    }
    std::cout << "Epoch | NumCorrect | TestAccuracy \n";
    std::cout << testCorrect;

    std::cout << "----- Results -----\n";
    std::cout << "best train accuracy = " << best_train_acc << " at epoch " << best_train_epoch << "\n";
    std::cout << "best test accuracy = " << best_test_acc << " at epoch " << best_test_epoch << "\n";

    // fc1.mOutput.printMat(std::cout);
    std::cout << "trained weight shape: ";
    fc1.getWeight().printShape();
    std::cout << "trained weight average = " << fc1.getWeight().average()
              << "; trained weight max value = " << fc1.getWeight().getColMax(0)
              << "; trained weight min value = " << fc1.getWeight().getColMin(0) << "\n";
    std::cout << "trained bias shape: ";
    fc1.getBias().printShape();
    std::cout << "trained bias = ";
    fc1.getBias().printMat();

    return 0;
}

int fc_int_dfa_ecg_one_layer_inference()
{
    utils::print_example_banner("PocketNN: Inference on MITBIH ECG data using pretrained weights for the 1-layer network");
    std::cout << "----- Constructing the network -----\n";
    pktnn::pktactv::Actv a = pktnn::pktactv::Actv::pocket_sigmoid;
    pktnn::pktfc fc1(128, 1);
    fc1.useDfa(true).setActv(a);
    std::cout << "Constructing the 1-layer fully connected neural network with pocket sigmoid activation done!\n";

    std::cout << "----- Loading pretrained weights and biases -----\n";
    fc1.loadWeight(config::save_weight_path);
    fc1.loadBias(config::save_bias_path);
    fc1.printWeightShape(std::cout);
    fc1.printBiasShape(std::cout);
    // fc1.getWeight().printMat(std::cout);
    // fc1.getBias().printMat(std::cout);

    std::cout << "----- Loading the ECG test data -----\n";
    int numTestSamples = 13245;
    pktnn::pktmat ecgTestLabels(numTestSamples, 1);
    pktnn::pktmat ecgTestInput(numTestSamples, 128);

    pktnn::pktloader::loadEcgData(ecgTestInput, "data/mit-bih/csv/mitbih_x_test_int.csv",
                                  numTestSamples, config::debugging);
    ecgTestInput.printShape();
    pktnn::pktloader::loadEcgLabels(ecgTestLabels, "data/mit-bih/csv/mitbih_bin_y_test.csv",
                                    numTestSamples, config::debugging);
    ecgTestLabels.selfMulConst(128); // scale the output from 0-1 to 0-128
    ecgTestLabels.printShape();

    std::cout << "----- Test -----\n";
    fc1.forward(ecgTestInput);
    int testNumCorrect = 0;
    for (int r = 0; r < numTestSamples; ++r)
    {
        int output_row_r = 0;
        fc1.mOutput.getElem(r, 0) > 64 ? output_row_r = 128 : output_row_r = 0;

        if (ecgTestLabels.getElem(r, 0) == output_row_r)
        {
            ++testNumCorrect;
        }
    }
    float test_acc = testNumCorrect * 1.0 / numTestSamples;
    std::cout << "Final test numCorrect = " << testNumCorrect << " (out of "
              << numTestSamples << " total examples)"
              << "\n";
    std::cout << "Final test accuracy = " << test_acc * 100 << "%"
              << "\n";

    return 0;
}