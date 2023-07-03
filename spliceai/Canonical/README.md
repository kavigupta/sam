
# Sparse Adjusted Motifs (SAM)

This repository contains the code used to produce the results in the paper.

## General Instructions

Run all the commands from this folder (`spliceai/Canonical`) unless otherwise specified.

We hope to clean up the codebase in the future and release it as a package on pypi, but
it should be adequate at the moment as documentation on the precise process used to compute
the results in the paper.

System requirements

- python 3.8 or higher
- Ubuntu machines (18.04, 20.04, 22.04)
- GPUs that are 12GB or larger
- Some (but not most) experiments may require up to 250GB of RAM,
- most models take several days to train.

It is possible that this codebase will run on other OSs or systems with lower system
requirements, but we have not tested this.

If you have any questions on running the code, contact kavig at mit dot edu.

## Explanation of files

The folder structure exists for legacy reasons as this project started off using the
spliceai codebase. The code does reference parent folders within the repository, especially
`../data` where a lot of the data is stored.

The folder `modular_splicing` contains the code for the project. The folder `experiments`
contains the entry points, one per discrete experiment. Most of these are not directly
relevant to the paper.

The folder `notebooks/biology-paper-1` contains the notebooks used to produce the results in the paper.

The folders `working` and `shelved` contain in-development code that should not impact
any of the results in the paper, it is included just so we don't have to modify the code
that imports it (which might introduce bugs).


## Prerequisites

First, install cuda on your machine. Check that this is working by running

    nvidia-smi

This command should complete without errors.

Then install the h5py dependencies. On ubuntu, this can be done via

    sudo apt install pkg-config libhdf5-dev
    sudo apt install python3-h5py

Then, install python, pip and virtualenv however is recommended on your system and run

    virtualenv venv --python=python3.8
    source venv/bin/activate
    pip install -r requirements.txt


## Getting the data

### Human Data (hg19, canonical)

To use human data, download http://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz and unzip to some path which we will call `$FA_LOCATION`. Then run

    PYTHONPATH=. python -m modular_splicing.data_pipeline.run canonical_dataset --ref-genome $FA_LOCATION

### Human data (alternative and evolutionary)

Use the same `$FA_LOCATION` as before. Then run

    PYTHONPATH=. python -m modular_splicing.data_pipeline.run evo_alt_dataset --ref-genome $FA_LOCATION

## Running Human Training Experiments

### LSSI models

Then, run the following commands to ensure the standard acceptor and donor LSSI models are in the appropriate locations

    mkdir -p model/splicepoint-model-acceptor-1/model/
    mkdir -p model/splicepoint-donor2-2.sh/model/

    cp splicepoint-models/donor2.m model/splicepoint-donor2-2.sh/model/1627500
    cp splicepoint-models/acceptor.m model/splicepoint-model-acceptor-1/model/1627200

You can train more of these using the instructions below using definitions

- acceptor: msp-262a6
- donor: msp-262da5

### Training Full models

> If you did not use `--data-dir ./` you will have to modify the commands below by prefixing them with `MSP_DATA_DIR=$YOUR_DATA_DIR`.

To run experiments, you can run

    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. python experiments/$EXPNAME.py $SEED

This will run the training for the given `EXPNAME` on seed `SEED` and gpu `GPU`. To test you can run

    CUDA_VISIBLE_DEVICES=$GPU  PYTHONPATH=. python experiments/$EXPNAME.py $SEED test

### NFM models

NFM models take a while to train and the RBNS data for them is quite large, so you can find them at
    [this drive folder](https://drive.google.com/drive/folders/1DQWGqYLNAzB8EwOHkCIlG6sef5iEMAKa?usp=sharing).

Place them in this directory as in `model/rbns-binary-model-TRA2A-21x2_4`.

### Reproducing the paper results

All the paper results can be found in notebooks under the folder `notebooks/biology-paper-1`.
Many of these experiments will error if run, complaining about a missing model as


    FileNotFoundError: [Errno 2] No such file or directory: 'model/$EXPNAME_$SEED/model'


This is because the models are not included in the repository. To run these experiments, you will need to train
the models yourself, as described above. A categorized list of relevant models can be found in
`modular_splicing/models_for_testing` under files

- `main_models.py` contains models used for the main results
- `lssi_models.py` contains models used for LSSI validation
- `eclip_models.py` contains models used for eCLIP validation
- `rnacompete_models.py` contains models used for RNACompete validation
- `reconstruction_models.py` contains models used for the reconstruction experiment
- `nfm_models.py` contains models used for the NFM experiment
- `spliceai_models.py` contains models used for the SpliceAI baseline. These should be trained as in the
    SpliceAI repository, and the resulting models should be placed in `model/standard-$WINDOW-1/model/`.
    for $WINDOW=400 and $WINDOW=10000.

Main results in notebooks:

- LSSI results and comparison to MaxEnt: lssi.ipynb
- Reconstruction of the original sequence: reconstruction-result.ipynb
- End-to-end accuracy results: e2e-results.ipynb
- Visualization of difference between AM and FM PSAMs: difference-between-psams.ipynb
- NFM results: nfm-results.ipynb
- Module Substitution Experiment: module-substitution-binarized.ipynb
- eCLIP validation results: eclip.ipynb
- AM-E vs equivalent FM: am-e.ipynb
- MPRA results: millions-of-random-sequences.ipynb
- Knockdown results: knockdown/knockdown-experiment.ipynb
- RNA Maps produced by our technqiue: rna-maps.ipynb
- Examples of Individual exons: individual-exon-examples.ipynb
- Results on evolutionary and alternative splice sites: evo-alt.ipynb

Supplementary Results in notebooks:

- Amount of information in motifs after sparsity layer: bits-in-postsparse.ipynb
- Using SpliceAI as an aggregator model: spliceai-as-aggregator.ipynb
- Robustness to reducing width in AMs: width-experiments/robustness.ipynb
- Various width experiments: width-experiments/topline.ipynb
- SpliceAI's interpretability: spliceai-interpretability.ipynb
- RNACompete results: appendix-rna-compete.ipynb
