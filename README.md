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

The PASTA library for HHE is built upon Microsoft's SEAL library. In this repo, SEAL is already installed in `libs/seal`. If you want to install it somewhere else, please refer to the [SEAL's repo](https://github.com/microsoft/SEAL).

## How to run
- Before compiling the application, you can take a look at `src/config.cpp` and change the configurations to your own settings. Notes:
    - The `debugging` variable should be set to `false` if you are not debugging the application (I used VSCode to debug so it might cause problems if you are not using VSCode to debug).
- `cmake -S . -B build -DCMAKE_PREFIX_PATH=libs/seal`  
Here, `-DCMAKE_PREFIX_PATH` specifies the path to the installed SEAL library.
- `cmake --build build`
- Run the compiled binary, for example `./build/MyPocketNN`