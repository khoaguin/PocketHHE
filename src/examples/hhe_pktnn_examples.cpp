#include "hhe_pktnn_examples.h"

int hhe_pktnn() {
    print_example_banner("HHE with PocketNN");
    print_line(__LINE__);
    std::cout << "Analyst creates the HE parameters and HE context" << std::endl;
    std::shared_ptr<seal::SEALContext> context = get_seal_context(config::plain_mod, config::mod_degree, config::seclevel);
    print_parameters(*context);
    
    return 0;
}