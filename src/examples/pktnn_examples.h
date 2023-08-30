#pragma once

#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>

#include <pocketnn/pktnn.h>

#include "../../configs/config.h"
#include "../util/sealhelper.h"
#include "../util/utils.h"

namespace pktnn_examples
{
    int fc_int_bp_simple();
    int fc_int_dfa_mnist();
    int fc_int_dfa_mnist_inference();
    int fc_int_dfa_mnist_one_layer();
    int fc_int_dfa_mnist_one_layer_inference();
    int fc_int_dfa_ecg_one_layer();
    int fc_int_dfa_ecg_one_layer_inference();
    int fc_int_dfa_spo2_one_layer();
    int fc_int_dfa_spo2_square();
}
