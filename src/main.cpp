#include <seal/seal.h>

#include "examples/pktnn_examples.h"
#include "examples/hhe_pktnn_examples.h"

int main() {
    
    // fc_int_bp_simple();  // simple training with dummy data using backpropagation
    // fc_int_dfa_mnist();  // training on MNIST using direct feedback alignment
    // fc_int_dfa_mnist_inference();  // inference on MNIST using model trained with direct feedback alignment
    
    hhe_pktnn();
    
    return 0;
}

