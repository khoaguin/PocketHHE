#pragma once

#include <iostream>
#include <vector>
#include <pocketnn/pktnn.h>
#include <seal/seal.h>

#include "../../configs/config.h"
#include "../util/sealhelper.h"
#include "../util/pastahelper.h"
#include "../util/utils.h"


int hhe_pktnn_mnist_inference();