# Installing Python Dependencies for Running Transformer Models

## Step 1: Install Miniconda

See the [regular instructions for installing Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation) for install instructions per OS. We'll follow the one for Linux.

Execute the following commands. Once you run `bash install-miniconda.sh`, respond to the questions asked on screen. The defaults should be OK.

**Important.** If you can, ensure that `$PYTHONPATH` is empty (not set).

```sh
# Download the latest Miniconda installer.
cd ~/Downloads/
curl -X GET https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh --output install-miniconda.sh
chmod +x install-miniconda.sh

# Run the installer.
bash install-miniconda.sh
```

Then, quit and re-open your terminal. Changes made by Miniconda will be in effect now.

To verify that your installation works, call `conda list`; it should provide a small list of pre-provided packages present by default in Miniconda.

Finally and optionally, update your Conda installation:

```sh
conda update -n base -c defaults conda  # press 'y' if needed
```

## Step 2: Build PyTorch from source

### Step 2.1: Create a Conda environment

```sh
cd source/py-model/
conda create --prefix ./.venv  # press 'y' if needed
conda activate $PWD/.venv
```

### Step 2.2: Install PyTorch dependencies

Run the steps below, typing 'y' where asked. Check your CUDA version using `nvidia-smi`.

**Note.** If you have a version that does not precisely match, use the nearest CUDA version supported below your system's CUDA version. For example, if you have CUDA 11.4, choose `magma-cuda113`. All supported MAGMA CUDA versions can be found [here](https://anaconda.org/pytorch/repo).

```sh
# General dependencies
conda install astunparse numpy ninja pyyaml setuptools cmake cffi typing_extensions future six requests dataclasses

# Linux dependencies (see `https://github.com/pytorch/pytorch#from-source` for other OSes).
MAGMA_CUDA_VERSION="000"  # Use `nvidia-smi` to replace "000" by e.g. "113".
conda install mkl mkl-include
conda install -c pytorch magma-cuda${MAGMA_CUDA_VERSION}
```

### Step 2.3: Download the PyTorch source code

Step out of the `py-model/` subdirectory, back into the `source/` subdirectory, and perform some Git commands afterwards, like so:

```sh
cd ../  # change directory to `source/`
git clone --recursive https://github.com/pytorch/pytorch
```

### Step 2.4: Install PyTorch

```sh
cd pytorch/
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export CXXFLAGS="-std=c++14"
export MY_CUDA_HOME="/usr/local/cuda-11.4"  # change this line to your CUDA location
export LDFLAGS="${LDFLAGS} -Wl,-rpath-link,${MY_CUDA_HOME}/lib64 -L${MY_CUDA_HOME}/lib64"
python3 setup.py install
```

