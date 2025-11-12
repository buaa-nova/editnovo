# Getting Started

## Installation

We recommend to run editnovo in a dedicated [conda environment](https://docs.conda.io/en/latest/).
This helps keep your environment for editnovo and its dependencies separate from your other Python environments.

```{Note}
Don't know what conda is?
Conda is a package manager for Python packages and many others.
We recommend installing the Anaconda Python distribution which includes conda.
Check out the [Windows](https://docs.anaconda.com/anaconda/install/windows/#), [MacOS](https://docs.anaconda.com/anaconda/install/mac-os/), and [Linux](https://docs.anaconda.com/anaconda/install/linux/) installation instructions.
```

Once you have conda installed, you can use this helpful [cheat sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf) to see common commands and what they do.

### Create a conda environment

First, open the terminal (MacOS and Linux) or the Anaconda Prompt (Windows).
All of the commands that follow should be entered into this terminal or Anaconda Prompt window---that is, your *shell*.
To create a new conda environment for editnovo, run the following:

```sh
conda create --name editnovo_env python=3.10
```

This will create an anaconda environment called `editnovo_env` that has Python 3.10 installed.

Activate this environment by running:

```sh
conda activate editnovo_env
```

Your shell should now say **(editnovo_env)** instead of **(base)**.
If this is the case, then you have set up conda and the environment correctly.

```{note}
Be sure to retype in the activation command into your terminal when you reopen anaconda and want to use editnovo.
```

### *Optional:* Install PyTorch manually

editnovo employs the PyTorch machine learning framework, which by default will be installed automatically along with the other dependencies.
However, if you have a graphics processing unit (GPU) that you want editnovo to use, we recommend installing PyTorch manually.
This will ensure that the version of PyTorch used by editnovo will be compatible with your GPU.
For installation instructions, see the [PyTorch documentation](https://pytorch.org/get-started/locally/#start-locally)

### Install editnovo

You can now install the editnovo Python package (dependencies will be installed automatically as needed):

```sh
pip install editnovo
```

After installation, test that it was successful by viewing the editnovo command line interface help:
```sh
editnovo --help
```
![`editnovo --help`](images/help.svg)


All auxiliary data, model, and training-related parameters can be specified in a YAML configuration file. 
To generate a YAML file containing the current editnovo defaults, run:
```sh
editnovo configure
```
![`editnovo configure --help`](images/configure-help.svg)

When using editnovo to sequence peptides from mass spectra or evaluate a previous model's performance, you can change some of the parameters in the first section of this file.
Parameters in the second section will not have an effect unless you are training a new editnovo model.

### Download model weights

Using editnovo to sequence peptides from new mass spectra, editnovo needs compatible pretrained model weights to make its predictions.
By default, editnovo will try to download the latest compatible model weights from GitHub when it is run. 

However, our model weights are uploaded with new editnovo versions on the [Releases page](https://github.com/Noble-Lab/editnovo/releases) under the "Assets" for each release (file extension: `.ckpt`).
This model file or a custom one can then be specified using the `--model` command-line parameter when executing editnovo.

Not all releases will have a model file included on the [Releases page](https://github.com/Noble-Lab/editnovo/releases), in which case model weights for alternative releases with the same major version number can be used.

The most recent model weights for editnovo version 4.2 and above are currently provided under [editnovo v4.2.0](https://github.com/Noble-Lab/editnovo/releases/tag/v4.2.0):
- `editnovo_v4_2_0.ckpt`: Default editnovo weights to use as described in [Melendez et al.](https://pubs.acs.org/doi/full/10.1021/acs.jproteome.4c00422). These weights will be downloaded automatically if no weights are explicitly specified.

Alternatively, model weigths for editnovo version 4.x as described in [Yilmaz et al.](https://www.nature.com/articles/s41467-024-49731-x) are currently provided under [editnovo v4.0.0](https://github.com/Noble-Lab/editnovo/releases/tag/v4.0.0):
- `editnovo_massivekb.ckpt`: editnovo weights to use when analyzing tryptic data. These weights need to be downloaded manually.
- `editnovo_nontryptic.ckpt`: editnovo weights to use when analyzing non-tryptic data, obtained by fine-tuning the tryptic model on multi-enzyme data. These weights need to be downloaded manually.

## Running editnovo

```{note}
We recommend a Linux system with a dedicated GPU to achieve optimal runtime performance.
```

### Sequence new mass spectra

To sequence your own mass spectra with editnovo, use the `editnovo sequence` command:

```sh
editnovo sequence -o results.mztab spectra.mgf
```
![`editnovo sequence --help`](images/sequence-help.svg)

editnovo can predict peptide sequences for MS/MS spectra in mzML, mzXML, and MGF files.
This will write peptide predictions for the given MS/MS spectra to the specified output file in mzTab format.

### Evaluate *de novo* sequencing performance

To evaluate _de novo_ sequencing performance based on known mass spectrum annotations, use the `editnovo evaluate` command:

```sh
editnovo evaluate annotated_spectra.mgf
```
![`editnovo evaluate --help`](images/evaluate-help.svg)


To evaluate the peptide predictions, ground truth peptide labels must to be provided as an annotated MGF file where the peptide sequence is denoted in the `SEQ` field. 
Compatible MGF files are available from [MassIVE-KB](https://massive.ucsd.edu/ProteoSAFe/static/massive-kb-libraries.jsp).

### Train a new model

To train a model from scratch, run:

```sh
editnovo train --validation_peak_path validation_spectra.mgf training_spectra.mgf
```
![`editnovo train --help`](images/train-help.svg)

Training and validation MS/MS data need to be provided as annotated MGF files, where the peptide sequence is denoted in the `SEQ` field.

If a training is continued for a previously trained model, specify the starting model weights using `--model`.

## Try editnovo on a small example

Let's use editnovo to sequence peptides from a small collection of mass spectra in an MGF file (~100 MS/MS spectra).
The example MGF file is available at [`sample_data/sample_preprocessed_spectra.mgf`](https://github.com/Noble-Lab/editnovo/blob/main/sample_data/sample_preprocessed_spectra.mgf).

To obtain *de novo* sequencing predictions for these spectra:
1. Download the example MGF above.
2. [Install editnovo](#installation).
3. Ensure your editnovo conda environment is activated by typing `conda activate editnovo_env`. (If you named your environment differently, type in that name instead.)
4. Sequence the mass spectra with editnovo, replacing `[PATH_TO]` with the path to the example MGF file that you downloaded:
```sh
editnovo sequence [PATH_TO]/sample_preprocessed_spectra.mgf
```

```{note}
If you want to store the output mzTab file in a different location than the current working directory, specify an alternative output location using the `--output` parameter.
```

This job should complete in < 1 minute.

Congratulations! editnovo is installed and running.
