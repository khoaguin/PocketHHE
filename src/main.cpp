#include "examples/pktnn_examples.h"
#include "examples/hhe_pktnn_examples.h"

int main()
{
    // --- Plaintext examples ---
    // - MNIST -
    // pktnn_examples::fc_int_bp_simple(); // simple training with dummy data using backpropagation
    // pktnn_examples::fc_int_dfa_mnist();  // training on MNIST using direct feedback alignment (a 3-layer network with pocket-tanh activation)
    // pktnn_examples::fc_int_dfa_mnist_inference();  // inference on MNIST using model trained with direct feedback alignment
    // pktnn_examples::fc_int_dfa_mnist_one_layer(); // training on MNIST using direct feedback alignment (only 1 layer)
    // pktnn_examples::fc_int_dfa_mnist_one_layer_inference(); // inference on MNIST using model trained with direct feedback alignment (only 1 layer)

    // - ECG data -
    // pktnn_examples::fc_int_dfa_ecg_one_layer(); // training on MIT-BIH using direct feedback alignment for the 1-layer nn
    // pktnn_examples::fc_int_dfa_ecg_one_layer_inference(); // inference on MIT-BIH for the 1-layer nn

    // - SpO2 (hypnogram) data -
    // pktnn_examples::fc_int_dfa_spo2_one_layer(); // training on hypnogram data using direct feedback alignment for the 1-layer nn
    // pktnn_examples::fc_int_dfa_spo2_square(); // training on SpO2 data with a network with 2 linear layers and a square activation function

    // --- HHE examples ---
    // hhe_pktnn_examples::hhe_pktnn_ecg_inference();       // encrypted inference protocol on ECG for the 1-layer nn
    // hhe_pktnn_examples::hhe_pktnn_mnist_inference();     // encrypted inference protocol on MNIST for the 1-layer nn
    hhe_pktnn_examples::hhe_pktnn_spo2_inference(); // encrypted inference protocol on SpO2 data for the 1-layer nn
    // hhe_pktnn_examples::hhe_pktnn_mnist_square_inference(); // encrypted inference protocol on MNIST / FMNIST data for the 2fc + square activation nn

    return 0;
}