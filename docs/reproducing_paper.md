# Reproducing the Neurocomputing (c-ss-ms-gVAE) experiments

This repository keeps the dataset loaders from the original research tree. Point `--data_path` (or `DATA_DIR`) at your local dataset root. The **active** malaria and BUI loaders read tensors from `--data_path`.

## Environment

```bash
conda env create -f environment.yml
conda activate c-ss-ms-gvae
```

Or from `src/`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt
```

Use a PyTorch build compatible with your CUDA driver if training on GPU. The pinned versions (PyTorch 1.1.0, Python 3.7) match the original research environment.

## Datasets

Point `--data_path` at a **local copy** of each dataset. Do not commit data into this repository.

### MVTec AD (`mvtec`, `net_name=mvtec_net`)

Download from <https://www.mvtec.com/company/research/datasets/mvtec-ad/>. Pass `--category` (`carpet`, `leather`, `tile`, `wood`, object names, …). The Stochastic U-Net reads `src/networks/configs/UNetConfig.json` from this repo.

### Malaria (`malaria_dataset`)

The wired loader is `src/datasets/malaria_dataset_patch_with_GGnoise_noveltest.py`. Place these tensors **in** `--data_path`:

```
<data_path>/train_data_1209onwards_10k.pt
<data_path>/train_label_1209onwards_10k.pt
<data_path>/test_data_2p5k.pt
<data_path>/test_label_2p5k.pt
```

Older malaria files under `src/datasets/malaria_dataset_*.py` are not used by `load_dataset`. They resolve tensors via `DATA_DIR` if you run them.

### Nanofibre (`nanofibre`)

ImageFolder layout with `train/` and `test/` under `--data_path` (see `src/datasets/nanofibre.py`).

### Breast ultrasound / BUI (`bui_dataset`)

Place these tensors in `--data_path`:

```
<data_path>/data_train_novel.pt
<data_path>/label_train_novel.pt
<data_path>/data_test_novel.pt
<data_path>/label_test_novel.pt
```

### MNIST / Fashion-MNIST / CIFAR-10

Downloaded by torchvision on first run if `--data_path` is writable.

### Corrosion and ODDS

Loaders are present. Corrosion tensors are resolved from `DATA_DIR` (for example `train_data_10k.pt`).

## Training (Deep SAD-style / c-ss-ms-gVAE)

Use the generalized runner (no machine-specific paths):

```bash
export DATA_DIR=$HOME/data/mvtec          # your local dataset root
export RESULTS_DIR=$HOME/results/c-ss-ms-gvae
./scripts/run_experiment.sh mvtec mvtec_net --category carpet
./scripts/run_experiment.sh --sweep malaria_dataset cifar10_LeNet \
    --ratios 0,0.01,0.05,0.1,0.2 --seeds 0,1,2 --ablations A
```

Or from `src/` (Click requires `<xp_dir>` to exist):

```bash
python main.py mvtec mvtec_net <xp_dir> <data_path> --category carpet [options...]
python main.py malaria_dataset cifar10_LeNet <xp_dir> <data_path> [options...]
python main.py nanofibre cifar10_LeNet <xp_dir> <data_path> [options...]
python main.py bui_dataset cifar10_LeNet <xp_dir> <data_path> [options...]
```

`--ablation_type {A,B,C,C_a,D,E,F,VAE,test,DeepSAD}` selects the loss configurations reported in the paper. `A` is the full proposed model.

## Hybrid SSAD baseline

```bash
python baseline_ssad.py malaria_dataset <xp_dir> <data_path> --hybrid True --load_ae /path/to/model.tar ...
```

`--load_ae` must match the architecture saved in `model.tar`.

## Binary classifier baseline

See `binary_classifier/` and `binary_classifier/baseline_binary_classifier.py`.

## t-SNE and visualization scripts

Plotters read `DATA_DIR` and `RESULTS_DIR` from the environment. Set those to your local folders before running `src/tsne_plotter_*.py`.
