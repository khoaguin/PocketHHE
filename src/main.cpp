#include "pktnn_examples.h"
#include <seal/seal.h>

int main() {
    // fc_int_bp_simple();  // simple training with dummy data using backpropagation
    // fc_int_dfa_mnist();  // training on MNIST using direct feedback alignment
    // fc_int_dfa_mnist_inference();  // inference on MNIST using model trained with direct feedback alignment

    seal::EncryptionParameters parms(seal::scheme_type::bfv);
    size_t poly_modulus_degree = 4096;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_plain_modulus(1024);
    seal::SEALContext context(parms);

    return 0;
}