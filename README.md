# PocketHHE 
Privacy-preserving Machine Learning (PocketNN on MNIST) with hybrid homomorphic encryption.
## Datasets
Two datasets used in this project are copied from their original website and are stored in `data/`
- MNIST dataset: MNIST dataset is from [the MNIST website](http://yann.lecun.com/exdb/mnist/)
- Fashion-MNIST dataset: Fashion-MNIST dataset is from [its github repository](https://github.com/zalandoresearch/fashion-mnist).

## Repo structure
```
├── data              
├── images      # hold the images in `README.md`
├── libs        # hold the libraries needed
├── src         # hold the source code for training and inferencing
├── tests           # hold the unit tests
└── weights         # hold the trained weights and biases
 ```

## Requirements
`cpp==9.4.0`   
`CMAKE>=3.13`

## How to run
- `cmake -S . -B build -DCMAKE_PREFIX_PATH=libs/seal`
- `cmake --build build`
- Run the compiled binary, for example `./build/MyPocketNN`