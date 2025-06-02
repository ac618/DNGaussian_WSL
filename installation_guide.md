# Installation Guide

## Step 1: Downgrade GCC

For CUDA 11.3 to build gridencoder, you need to downgrade GCC to version 9 (<=10.0). You can run the following commands to install and setup GCC 9:

```bash
sudo apt install gcc-9 g++-9
```

Then, set the alternatives:

```bash
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90
```

If priority 90 is lower than current version, you can maually set the alternatives:

```bash
sudo update-alternatives --config gcc
sudo update-alternatives --config g++
```

To check the current version of GCC, you can run:

```bash
gcc --version
```

## Step 2: Create Conda Environment

You can create a conda environment using the provided `environment.yml` file.

```bash
conda env create -f environment.yml
conda activate dngaussian
```

## Step 3: Install timm without dependencies

The `timm` package has a dependency `puccinialin`, which requires Python 3.9 or higher. Since we are using Python 3.7, we need to install `timm` without dependencies.

```bash
pip install timm==0.9.10 --no-deps
```

To verify the installation, you can run:

```bash
python -c "import timm; print(timm.__version__)"
```

## Step 4: Install Submodules

You need to install the submodules of the project. You can do this by running the following commands:

```bash
cd submodules
git clone git@github.com:ashawkey/diff-gaussian-rasterization.git --recursive
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git
```

Before installing the submodules, you need to modify the `setup.py` files of `diff-gaussian-rasterization` and `simple-knn`. Specifically, you need to add `-D__int128=long long` to the `extra_compile_args` for `nvcc` in both setup files. If you don't do this, you may encounter an error related to `__int128` type not being defined like the error message below:

```
/usr/include/linux/types.h:12:27: error: expected initializer before ‘__s128’
    12 | typedef __signed__ __int128 __s128 __attribute__((aligned(16)));
```

You can modify the `setup.py` files manually or use the files provided in `setups` directory of this repository by runing the following commands:

```bash
cp ../setups/diff-gaussian-rasterization/setup.py diff-gaussian-rasterization/
cp ../setups/simple-knn/setup.py simple-knn/
```

```bash
pip install ./diff-gaussian-rasterization ./simple-knn
```

After the installation, you should be good to go.

## Notes

The `environment.yml` and `setup.py` of the gridencoder have been modified to support newer Ubuntu.
Changes regarding the `environment.yml` file:

- cudatoolkit=11.3 is replaced with cudatoolkit-dev=11.3 since nvcc is required to build the gridencoder and submodules. If your CUDA version is already 11.3, (which is not the case for most users though), you can use `cudatoolkit=11.3` instead.

Changes regarding the `setup.py` in gridencoder:

- `'-D__int128=long long'` is added to the nvcc flags to avoid the `__int128` type error.
