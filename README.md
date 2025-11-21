# EditNovo

This is the official repo for the paper: 

We will release the future model update (user-interface, new model weight, optimized modules etc) here, please leave a **star** and **watching** if you want to get notified and follow up.

## Environment Setup

Create a new conda environment first:

```
conda create --name EditNovo python=3.10
```

This will create an anaconda environment

Activate this environment by running:

```
conda activate EditNovo
```

then install dependencies:

```
pip install -r ./requirements.txt
```

Then, compile C++/CUDA Extensions:

```
python setup.py build_ext --inplace
```

The installation automatically compiles:

- **editnovo.libnat**: CPU version edit distance calculation library
- **editnovo.libnat_cuda**: GPU accelerated version (if CUDA detected)

Lastly, install CuPy to use our CUDA-accelerated precise mass-control decoding:

**_Please install the following Cupy package in a GPU available env, If you are using a slurm server, this means you have to enter a interative session with sbatch to install Cupy, If you are using a machine with GPU already on it (checking by nvidia-smi), then there's no problem_**

**Check your CUDA version using command nvidia-smi, the CUDA version will be on the top-right corner**

| cuda version                    | command                  |
| ------------------------------- | ------------------------ |
| v10.2 (x86_64 / aarch64)        | pip install cupy-cuda102 |
| v11.0 (x86_64)                  | pip install cupy-cuda110 |
| v11.1 (x86_64)                  | pip install cupy-cuda111 |
| v11.2 ~ 11.8 (x86_64 / aarch64) | pip install cupy-cuda11x |
| v12.x (x86_64 / aarch64)        | pip install cupy-cuda12x |

## Model Settings

Some of the important settings in config.yaml under ./editnovo

**n_beam**: number of CTC-paths (beams) considered during inference. We recommend a value of 5.

**n_peaks**: Number of the most intense peaks to retain, any remaining peaks are discarded. We recommend a value of 150.

**min_mz**: Minimum peak m/z allowed, peaks with smaller m/z are discarded. We recommend a value of 50.

**max_mz**: Maximum peak m/z allowed, peaks with larger m/z are discarded. We recommend a value of 2500.

**min_intensity**: Min peak intensity allowed, less intense peaks are discarded. We recommend a value of 0.01.

## Run Instructions

### Step 1: Download Required Files

To evaluate the provided test MGF file (you can replace this MGF file with your own), download the following files:

1. **Model Checkpoint**: [editnovo-massive-kb.ckpt](https://drive.google.com/file/d/1SOc_XPBBgrgcxGDJApjK7hLs3ao3e2cs/view?usp=sharing)
2. **Test MGF File**: [Bacillus.10k.mgf](https://drive.google.com/file/d/1HqfCETZLV9ZB-byU0pqNNRXbaPbTAceT/view?usp=drive_link)

**Note:** If you are using a remote server, you can use the `gdown` package to easily download the content from Google Drive to your server disk.

### Step 2: Choose the Mode

The `mode` argument can be set to either:

- `evaluate`: Use this mode when evaluating data with a labeled dataset.
- `train`: Use this mode to train a new model.


**Important**: Select `evaluate` only if your data is labeled.

### Step 3: Run the Commands

Execute the following command in the terminal:

```bash
CUDA_VISIBLE_DEVICES=0 python -m editnovo.editnovo evaluate ./bacillus.10k.mgf -m ./model_massive.ckpt
```
This command forces the program to use only GPU 0.

If your server has multiple GPUs and you want to utilize all of them to speed up the evaluation:

```bash
python -m editnovo.editnovo evaluate ./bacillus.10k.mgf -m ./model_massive.ckpt
```

This automatically uses all GPUs available in the current machine.