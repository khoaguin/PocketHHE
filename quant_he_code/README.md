Source code from the paper [Neural Network Quantisation for Faster Homomorphic Encryption](https://eprint.iacr.org/2023/503)

- `src/`: The source code from the original paper
- `notebooks/`: Modified code
  - `mnist_conv.ipynb`: QTA of a conv model (2 conv layers, 2 square activation functions, 1 linear layer)
  - `mnist_fc.ipynb`: QTA of a fc model (2 linear layers, 1 square activation function)
  - `mnist_fc_2.ipynb`: same model with `mnist_fc.ipynb`, but in brevitas style according to [concrete-ml's tutorial](https://docs.zama.ai/concrete-ml/deep-learning/fhe_friendly_models)
