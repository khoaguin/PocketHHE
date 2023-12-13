#include "examples/pktnn_examples.h"
#include "examples/hhe_pktnn_examples.h"

#include "seal/seal.h"
#include <iostream>
#include <vector>

using namespace std;
using namespace seal;

namespace experimentals
{

     inline int encrypted_sum()
     {
          utils::print_example_banner("Example: Encrypted Vector Sum in BFV");

          // Set up SEAL with batching enabled
          seal::EncryptionParameters parms(seal::scheme_type::bfv);
          // size_t poly_modulus_degree = config::mod_degree;
          size_t poly_modulus_degree = 8192;
          parms.set_poly_modulus_degree(poly_modulus_degree);
          parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
          parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
          seal::SEALContext context(parms);
          sealhelper::print_parameters(context);

          // Key generation
          seal::PublicKey public_key;
          seal::GaloisKeys gal_keys;
          seal::KeyGenerator keygen(context);
          seal::RelinKeys relin_keys;
          keygen.create_public_key(public_key);
          keygen.create_galois_keys(gal_keys);
          keygen.create_relin_keys(relin_keys);
          seal::SecretKey secret_key = keygen.secret_key();

          // Encryptor, decryptor, evaluator, and batch encoder
          seal::Encryptor encryptor(context, public_key);
          seal::Decryptor decryptor(context, secret_key);
          seal::Evaluator evaluator(context);
          seal::BatchEncoder encoder(context);

          // Example vectors
          vector<int64_t> vec1 = {-1, 0, 3, 4, 5, 6, -3, -1};
          vector<int64_t> vec2 = {4, 5, 6, 7, 8, 9, 3, 4};

          // Plaintext Comutation
          int32_t sum = 0;
          for (size_t i = 0; i < vec1.size(); ++i)
          {
               sum += vec1[i] * vec2[i];
          }

          // Resize vectors to match the batch size
          // Essentially, we pad them with zeros
          size_t slot_count = encoder.slot_count();
          std::cout << "slot_count = " << slot_count << std::endl;
          // vec1.resize(slot_count, 0);
          // vec2.resize(slot_count, 0);
          // utils::print_vec(vec1, vec1.size(), "vec1");
          int32_t len = (int32_t)vec1.size();
          std::cout << "vec1.size = " << vec1.size() << std::endl;
          std::cout << "vec2.size = " << vec2.size() << std::endl;

          // Encode and encrypt the vectors
          seal::Plaintext plain_vec1, plain_vec2;
          encoder.encode(vec1, plain_vec1);
          encoder.encode(vec2, plain_vec2);
          seal::Ciphertext encrypted_vec1, encrypted_vec2;
          encryptor.encrypt(plain_vec1, encrypted_vec1);
          encryptor.encrypt(plain_vec2, encrypted_vec2);

          // Multiply the encrypted vectors
          seal::Ciphertext encrypted_product;
          evaluator.multiply(encrypted_vec1, encrypted_vec2, encrypted_product);
          std::cout << "encrypted_product size before Relinearization = " << encrypted_product.size() << std::endl;
          evaluator.relinearize_inplace(encrypted_product, relin_keys);
          std::cout << "encrypted_product size after Relinearization = " << encrypted_product.size() << std::endl;

          // Decrypt and decode the multiplication
          seal::Plaintext plain_product;
          decryptor.decrypt(encrypted_product, plain_product);
          vector<int64_t> decoded_product;
          encoder.decode(plain_product, decoded_product);
          utils::print_vec(decoded_product, len, "decoded_product");

          // Sum the elements of the resulting vector using encrypted rotation
          seal::Ciphertext sum_cipher;
          sealhelper::encrypted_vec_sum(encrypted_product, sum_cipher, evaluator, gal_keys, len);

          // Decrypt and decode the result
          seal::Plaintext plain_sum;
          decryptor.decrypt(sum_cipher, plain_sum);
          vector<int64_t> decoded_sum;
          encoder.decode(plain_sum, decoded_sum);
          utils::print_vec(decoded_sum, len, "decoded_sum");

          // Output the sum of the product of the vectors
          cout << "Plain sum of the product of the vectors = " << sum << endl;
          cout << "Decrypted sum of the product of the vectors: " << decoded_sum[len - 1] << endl;

          return 0;
     }

     inline void example_rotation_bfv()
     {
          // Source: https://github.com/microsoft/SEAL/blob/4.0.0/native/examples/6_rotation.cpp
          utils::print_example_banner("Example: Rotation / Rotation in BFV");

          EncryptionParameters parms(scheme_type::bfv);

          size_t poly_modulus_degree = 8192;
          parms.set_poly_modulus_degree(poly_modulus_degree);
          parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
          parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));

          SEALContext context(parms);
          sealhelper::print_parameters(context);
          cout << endl;

          KeyGenerator keygen(context);
          SecretKey secret_key = keygen.secret_key();
          PublicKey public_key;
          keygen.create_public_key(public_key);
          RelinKeys relin_keys;
          keygen.create_relin_keys(relin_keys);
          Encryptor encryptor(context, public_key);
          Evaluator evaluator(context);
          Decryptor decryptor(context, secret_key);

          BatchEncoder batch_encoder(context);
          size_t slot_count = batch_encoder.slot_count();
          size_t row_size = slot_count / 2;
          cout << "Plaintext matrix row size: " << row_size << endl;

          vector<uint64_t> pod_matrix(slot_count, 0ULL);
          pod_matrix[0] = 0ULL;
          pod_matrix[1] = 1ULL;
          pod_matrix[2] = 2ULL;
          pod_matrix[3] = 3ULL;
          pod_matrix[row_size] = 4ULL;
          pod_matrix[row_size + 1] = 5ULL;
          pod_matrix[row_size + 2] = 6ULL;
          pod_matrix[row_size + 3] = 7ULL;

          cout << "Input plaintext matrix:" << endl;
          utils::print_matrix(pod_matrix, row_size);

          /*
          First we use BatchEncoder to encode the matrix into a plaintext. We encrypt
          the plaintext as usual.
          */
          Plaintext plain_matrix;
          utils::print_line(__LINE__);
          cout << "Encode and encrypt." << endl;
          batch_encoder.encode(pod_matrix, plain_matrix);
          Ciphertext encrypted_matrix;
          encryptor.encrypt(plain_matrix, encrypted_matrix);
          cout << "    + Noise budget in fresh encryption: " << decryptor.invariant_noise_budget(encrypted_matrix) << " bits"
               << endl;
          cout << endl;

          /*
          Rotations require yet another type of special key called `Galois keys'. These
          are easily obtained from the KeyGenerator.
          */
          GaloisKeys galois_keys;
          keygen.create_galois_keys(galois_keys);

          /*
          Now rotate both matrix rows 3 steps to the left, decrypt, decode, and print.
          */
          utils::print_line(__LINE__);
          cout << "Rotate rows 3 steps left." << endl;
          evaluator.rotate_rows_inplace(encrypted_matrix, 3, galois_keys);
          Plaintext plain_result;
          cout << "    + Noise budget after rotation: " << decryptor.invariant_noise_budget(encrypted_matrix) << " bits"
               << endl;
          cout << "    + Decrypt and decode ...... Correct." << endl;
          decryptor.decrypt(encrypted_matrix, plain_result);
          batch_encoder.decode(plain_result, pod_matrix);
          utils::print_matrix(pod_matrix, row_size);

          /*
          We can also rotate the columns, i.e., swap the rows.
          */
          utils::print_line(__LINE__);
          cout << "Rotate columns." << endl;
          evaluator.rotate_columns_inplace(encrypted_matrix, galois_keys);
          cout << "    + Noise budget after rotation: " << decryptor.invariant_noise_budget(encrypted_matrix) << " bits"
               << endl;
          cout << "    + Decrypt and decode ...... Correct." << endl;
          decryptor.decrypt(encrypted_matrix, plain_result);
          batch_encoder.decode(plain_result, pod_matrix);
          utils::print_matrix(pod_matrix, row_size);

          /*
          Finally, we rotate the rows 4 steps to the right, decrypt, decode, and print.
          */
          utils::print_line(__LINE__);
          cout << "Rotate rows 4 steps right." << endl;
          evaluator.rotate_rows_inplace(encrypted_matrix, -4, galois_keys);
          cout << "    + Noise budget after rotation: " << decryptor.invariant_noise_budget(encrypted_matrix) << " bits"
               << endl;
          cout << "    + Decrypt and decode ...... Correct." << endl;
          decryptor.decrypt(encrypted_matrix, plain_result);
          batch_encoder.decode(plain_result, pod_matrix);
          utils::print_matrix(pod_matrix, row_size);

          /*
          Note that rotations do not consume any noise budget. However, this is only
          the case when the special prime is at least as large as the other primes. The
          same holds for relinearization. Microsoft SEAL does not require that the
          special prime is of any particular size, so ensuring this is the case is left
          for the user to do.
          */
     }

}
