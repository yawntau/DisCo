# DisCo: Disentanglement with Compensation for Remote Sensing Image Change Captioning

Official implementation of **DisCo** for remote sensing image change captioning.

<p align="center">
  <img src="https://img.shields.io/badge/Task-Change%20Captioning-1f6feb?style=for-the-badge" alt="Task">
  <img src="https://img.shields.io/badge/Framework-PyTorch-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white" alt="Framework">
  <img src="https://img.shields.io/badge/Backbone-SegFormer-0f766e?style=for-the-badge" alt="Backbone">
  <img src="https://img.shields.io/badge/Datasets-LEVIR--MCI%20%7C%20WHU--CDC-f59e0b?style=for-the-badge" alt="Datasets">
</p>

## 📖 About

- **Authors**: Tao Yang, Qing Zhou, and Qi Wang
- **Task**: Remote sensing image change captioning
- **Framework**: PyTorch

DisCo is a change captioning framework that combines purified feature disentanglement with latent relational compensation to improve semantic description of changes between bi-temporal remote sensing images.

## ✨ Features

- Supports training and evaluation on **LEVIR-MCI** and **WHU-CDC**
- Uses a **SegFormer**-based visual encoder by default
- Includes distributed training with `torchrun`
- Provides evaluation scripts with BLEU, METEOR, ROUGE-L, and CIDEr
- Includes single-pair caption generation for inference

## 📁 Repository Structure

```text
DisCo/
├── data/
│   ├── LEVIR_MCI/
│   ├── whu_CDC/
│   └── LEVIR_MCI.py
├── eval_func/
│   ├── bleu/
│   ├── cider/
│   ├── meteor/
│   └── rouge/
├── model/
│   ├── model_decoder.py
│   ├── model_encoder_att.py
│   ├── segformer.py
│   └── pretrained/              # prepare SegFormer MIT backbone weights here
├── utils_tool/
├── merge_metrics.py
├── predict.py
├── preprocess_data.py
├── requirements.txt
├── run.sh
├── run_test.sh
├── test.py
└── train.py
```

## 🛠 Environment

### Option 1: Install from `requirements.txt`

```bash
conda create -n change python=3.11 -y
conda activate change
pip install -r requirements.txt
```

### Option 2: Install PyTorch first with a matching CUDA build

If your machine uses a specific CUDA version, install the matching PyTorch build first, then install the remaining dependencies:

```bash
conda create -n change python=3.11 -y
conda activate change

# Example: choose the correct command for your CUDA version
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

## 📌 Prerequisites

### 1. Prepare Java for METEOR evaluation

`test.py` uses `eval_func/meteor/meteor-1.5.jar`, so Java must be available:

```bash
java -version
```

If Java is missing, install OpenJDK first.

### 2. Prepare SegFormer pretrained weights

The default backbone is `segformer-mit_b1`. The code loads pretrained weights from:

```text
model/pretrained/mit_b1.pth
```

If you want to use another MIT backbone, also place the matching file in `model/pretrained/`, for example:

```text
model/pretrained/mit_b0.pth
model/pretrained/mit_b2.pth
model/pretrained/mit_b3.pth
model/pretrained/mit_b4.pth
model/pretrained/mit_b5.pth
```

## 🗂 Dataset Preparation

This repository already includes dataset split files, token files, and vocabulary files under `data/LEVIR_MCI/` and `data/whu_CDC/`.

You still need to prepare the image directories yourself.

### Expected image directory layout

```text
your_dataset_root/
└── images/
    ├── train/
    │   ├── A/
    │   └── B/
    ├── val/
    │   ├── A/
    │   └── B/
    └── test/
        ├── A/
        └── B/
```

Each file name in `A/` should have a corresponding image with the same file name in `B/`.

### Included metadata files

- `data/LEVIR_MCI/train.txt`, `val.txt`, `test.txt`
- `data/LEVIR_MCI/tokens/`
- `data/LEVIR_MCI/vocab.json`
- `data/whu_CDC/train.txt`, `val.txt`, `test.txt`
- `data/whu_CDC/tokens/`
- `data/whu_CDC/vocab.json`

### Regenerating tokens and vocabulary

If you want to rebuild token files and vocabulary from raw annotations, use:

```bash
python preprocess_data.py --dataset LEVIR_MCI
```

Note that `preprocess_data.py` currently contains a hard-coded `DATA_PATH_ROOT`. You should update that path before using the script on your machine.

## 🚀 Training

### Train on WHU-CDC

The provided `run.sh` script is configured for WHU-CDC. Before running it, update:

- `DATA_FOLDER`
- `LIST_PATH`
- `TOKEN_FOLDER`
- `LD_LIBRARY_PATH` if needed
- `CUDA_VISIBLE_DEVICES`

Then run:

```bash
bash run.sh
```

### Manual training command

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train.py \
    --data_folder /path/to/whu_CDC_dataset/images \
    --list_path ./data/whu_CDC/ \
    --token_folder ./data/whu_CDC/tokens/ \
    --vocab_file vocab \
    --max_length 26 \
    --allow_unk 1 \
    --data_name whu_CDC \
    --savepath ./models_ckpt/ \
    --train_batchsize 64 \
    --workers 4 \
    --val_interval 5 \
    --num_epochs 50
```

### Main training arguments

- `--data_folder`: dataset image root
- `--list_path`: directory containing `train.txt`, `val.txt`, and `test.txt`
- `--token_folder`: token file directory
- `--vocab_file`: vocabulary json prefix
- `--data_name`: `LEVIR_CC`, `LEVIR_MCI`, or `whu_CDC` depending on your setup
- `--network`: backbone name, default `segformer-mit_b1`
- `--train_batchsize`: training batch size per GPU
- `--val_interval`: validation frequency in epochs
- `--checkpoint`: optional checkpoint for resuming

## 📊 Evaluation

### Evaluate a single checkpoint

```bash
python test.py \
    --data_folder /path/to/LEVIR-MCI-dataset/images \
    --list_path ./data/LEVIR_MCI/ \
    --token_folder ./data/LEVIR_MCI/tokens/ \
    --vocab_file vocab \
    --max_length 42 \
    --allow_unk 1 \
    --data_name LEVIR_MCI \
    --checkpoint ./models_ckpt/your_run/best_model.pth \
    --summary_json ./models_ckpt/your_run/all_metrics_summary.json \
    --gpu_id 0
```

### Evaluate all checkpoints in a directory

Edit `CKPT_DIR` and related paths in `test.sh`, then run:

```bash
bash test.sh
```

### Training + automatic evaluation pipeline

`run_test.sh` performs:

1. Training
2. Multi-checkpoint evaluation
3. Metric merging with `merge_metrics.py`

Before using it, update dataset paths, GPU IDs, and environment variables in the script.

## 🔎 Inference

Generate a caption for a single image pair:

```bash
python predict.py \
    --imgA_path /path/to/image_A.png \
    --imgB_path /path/to/image_B.png
```

Important notes:

- `predict.py` currently uses hard-coded default paths that are Windows-specific
- The checkpoint path inside `predict.py` should be changed to your actual trained model
- The vocabulary path and dataset-related defaults in `predict.py` should also be adjusted before use

## 📦 Output

Training checkpoints are saved under:

```text
./models_ckpt/
```

Evaluation results are typically saved under:

```text
./predict_result/
./models_ckpt/<run_name>/all_metrics_summary.json
```

## ⚠️ Notes

- `model/pretrained/` is required for SegFormer backbones but is ignored by the current `.gitignore`
- `models_ckpt/` is ignored and should not be committed
- `preprocess_data.py` contains machine-specific dataset paths and may need editing before use
- `predict.py` contains machine-specific default paths and should be adjusted before inference
- The repository includes evaluation code for METEOR, so Java is a runtime dependency for testing

## 📝 Citation

If you use this repository in your research, please cite the corresponding paper when available.

```bibtex
@article{disco_remote_sensing_change_captioning,
  title={DisCo: Purified Disentanglement with Latent Relational Compensation for Remote Sensing Image Change Captioning},
  author={Yang, Tao and Zhou, Qing and Wang, Qi},
  journal={To be updated},
  year={To be updated}
}
```

## 🙏 Acknowledgment

This repository uses standard captioning metrics including BLEU, METEOR, ROUGE-L, and CIDEr, and builds on PyTorch, MMEngine, MMCV, MMSegmentation, and SegFormer-related components.
