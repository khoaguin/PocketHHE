# PocketHHE

An Integer-only, Lightweight Privacy-preserving Machine Learning Framework with Hybrid Homomorphic Encryption (HHE).
Built with [SEAL](https://github.com/microsoft/SEAL), [PASTA](https://github.com/IAIK/hybrid-HE-framework) and [PocketNN](https://github.com/khoaguin/PocketNN).

## Datasets

The image datasets used in this project are copied from their original website and are stored in `data/`

- MNIST dataset: MNIST dataset is from [the MNIST website](http://yann.lecun.com/exdb/mnist/)
- Fashion-MNIST dataset: Fashion-MNIST dataset is from [its github repository](https://github.com/zalandoresearch/fashion-mnist).

We use the processed ECG dataset from [this work](https://github.com/SharifAbuadbba/split-learning-1D) which is originally the [MIT-BIH 1.0 from Physionet](https://www.physionet.org/content/mitdb/1.0.0/).

## Repo structure

```
├── configs         # hold the configuration parameters needed to run experiments
├── data            # hold the datasets
├── images          # hold the images in `README.md`
├── libs            # hold the libraries needed
├── notebooks       # hold the notebooks to train float neural nets on plaintext data
├── src             # hold the source code
    └── main.cpp    # comment out the functions in `main` to run different protocols / experiments
├── tests           # hold some unit tests
└── weights         # hold the trained weights and biases
```

## Requirements

`cpp==11.3.0`  
`CMAKE>=3.25.1`  
`SEAL==4.0.0`

The PASTA library for HHE is built upon Microsoft's SEAL library. In this repo, SEAL is already installed in `libs/seal`. If you want to install it somewhere else, please refer to the [SEAL's repo](https://github.com/microsoft/SEAL).

## How to run

Before compiling the application, you can take a look at `src/config.cpp` and change the configurations to your own settings. Notes:

- The `debugging` variable should be set to `false` if you are not debugging the application and to `true` if you want to debug (I used VSCode to debug so it might cause problems if you are not using VSCode).
- When `dry_run` is `true`, we only run a few data examples set by `dry_run_num_samples`. Otherwise it runs the whole dataset.
- `save_weight_path` and `save_bias_path` defines the paths that the trained weights and bias will be saved. It also defines the paths that the trained weights and bias will be loaded from in the inference protocols.

To compile and run, run `bash start.sh`, which essentially does the following:

- Run `cmake -S . -B build -DCMAKE_PREFIX_PATH=libs/seal`.
  Here, `-DCMAKE_PREFIX_PATH` specifies the path to the installed SEAL library.
- Then build with `cmake --build build`
- Run the compiled binary, for example `./build/PocketHHE`
