#include "examples/pktnn_examples.h"
#include "examples/hhe_pktnn_examples.h"

int main()
{
    // Plaintext examples
    // fc_int_bp_simple(); // simple training with dummy data using backpropagation
    // fc_int_dfa_mnist();  // training on MNIST using direct feedback alignment (a 3-layer network with pocket-tanh activation)
    // fc_int_dfa_mnist_inference();  // inference on MNIST using model trained with direct feedback alignment

    // fc_int_dfa_mnist_one_layer(); // training on MNIST using direct feedback alignment (only 1 layer)
    // fc_int_dfa_mnist_one_layer_inference(); // inference on MNIST using model trained with direct feedback alignment (only 1 layer)

    // fc_int_dfa_ecg_one_layer(); // training on MIT-BIH using direct feedback alignment (only 1 layer)
    // fc_int_dfa_ecg_one_layer_inference();

    // HHE examples
    // hhe_pktnn_examples::hhe_pktnn_mnist_inference();
    // hhe_pktnn_examples::hhe_pktnn_ecg_inference();

    return 0;
}