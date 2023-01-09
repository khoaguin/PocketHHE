# PocketHHE 
An Integer-only, Lightweight Privacy-preserving Machine Learning Framework with Hybrid Homomorphic Encryption (HHE). 
Built with [SEAL](https://github.com/microsoft/SEAL), [PASTA](https://github.com/IAIK/hybrid-HE-framework) and [PocketNN](https://github.com/khoaguin/PocketNN).
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
`SEAL==4.0.0`  

The PASTA library for HHE is built upon Microsoft's SEAL library. [Here](https://github.com/microsoft/SEAL) are the instructions for installing the Microsoft's SEAL library. You can also follow the following instructions 
1. First, clone the project by `https://github.com/khoaguin/PocketHHE` and `cd` into it.
2. 


```bash
https://github.com/khoaguin/PocketHHE
cd PocketHHE/libs

```

## How to run
- `cmake -S . -B build -DCMAKE_PREFIX_PATH=libs/seal`  
Note that `-DCMAKE_PREFIX_PATH` specifies where you install the SEAL library in your work station. In our repo, we install SEAL locally into `libs/seal`.
- `cmake --build build`
- Run the compiled binary, for example `./build/MyPocketNN`