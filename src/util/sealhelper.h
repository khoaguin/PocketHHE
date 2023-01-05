#pragma once

#include <cstddef>
#include <iostream>
#include <iomanip>
#include <seal/seal.h>

/*
Helper function: Prints the name of the example in a fancy banner.
*/
void print_example_banner(std::string title);

/*
Helper function: Print line number.
*/
void print_line(int line_number);

/*
Helper function: get a SEALContext from parameters.
*/
std::shared_ptr<seal::SEALContext> get_seal_context(uint64_t plain_mod, uint64_t mod_degree, int seclevel);

/*
Helper function: Prints the parameters in a SEALContext.
*/
void print_parameters(const seal::SEALContext &context);