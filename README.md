# CMSR

This is an implementation for the framework [CMT-SR](https://github.com/ChenMetanoia/CATSR-KDD/).

## Overview

In this project, we reproduce the study [“Pre-Training with Transferable Attention for Addressing Market Shifts in Cross-Market Sequential Recommendation”](https://www.researchgate.net/profile/Chen_Wang423/publication/383117472_Pre-Training_with_Transferable_Attention_for_Addressing_Market_Shifts_in_Cross-Market_Sequential_Recommendation/links/66bd1e86311cbb0949391351/Pre-Training-with-Transferable-Attention-for-Addressing-Market-Shifts-in-Cross-Market-Sequential-Recommendation.pdf). The paper addresses a critical challenge in modern recommendation systems—namely, the need to adapt recommendation models to diverse market conditions. To overcome these issues, the authors propose a novel framework that leverages pre-training with a specially designed item re-weighting loss and selective self-attention transferring, ensuring better adaptation to market-specific dynamics.

Our objective is to compare the reproduced results with those reported in the original paper to verify the soundness of the proposed theoretical principles. By carefully aligning our implementation with the original framework, we aim to determine whether any discrepancies in performance emerge. In cases where results deviate, a thorough investigation will be conducted to identify potential issues in our reproduction process, thereby clarifying the sources of any observed differences.

## Framework

![framework](pic\framework.png)

## Requirements

```
pytorch==1.11.0
python==3.8
recbole==1.1.1
transformers==4.18.0
```

Operating System: Linux / Window

## Datasets

Download [Amazon meta dataset](https://nijianmo.github.io/amazon/index.html)

Category: Electronics

Data: metadata

Put dataset into `data/Amazon/metadata` directory, i.e. `data/Amazon/metadata/meta_Electronics.json.gz`

## Quick Start

### Pretrain

```
sh scripts/pretrain.sh
```

### Finetune

```
sh scripts/finetune.sh
```

## Source File Description

`config/CMSR.yaml`: Configuration file for the CMSR model, specifying model architecture, training parameters, evaluation metrics, and hyperparameters.

`config/DATA.yaml`: Configuration file for dataset-related settings, including data paths, preprocessing options, and splitting strategies.

`data/data_processing.py`: Script for processing raw data, generating item embeddings using a pre-trained transformer model, and preparing datasets for training and evaluation.

`log/loss.ipynb`: Jupyter Notebook for visualizing training loss and evaluation metrics (e.g., NDCG@10, Hit@10) from log files during pretraining and fine-tuning.

`model/adapter.py`: Implements MLP-based adapters for modifying query and key matrices in the Transformer architecture.

`model/cmsr.py`: Implementation of the CMSR model, including its architecture, loss functions, and forward pass logic for sequential recommendation tasks.

`model/encoder.py`: Implements the Transformer-based encoder with multiple layers for processing sequential data.

`model/layers.py`: Contains core building blocks like multi-head attention, feed-forward layers, and layer normalization for the Transformer model.

`model/wbpr.py`: Implements the Weighted Bayesian Personalized Ranking (WBPR) loss function for recommendation tasks.

`model/wce.py`: Provides functions for calculating Weighted Cross-Entropy (WCE) loss and item popularity-based weights.

`scripts/finetune.sh`: Shell script to automate the fine-tuning process by iterating over various hyperparameter and dataset combinations.

`scripts/pretrain.sh`: Shell script to automate the pretraining process with adjustable configurations and logging.

`finetune.py`: Script for fine-tuning the CMSR model with configurable parameters, dataset handling, and evaluation logging.

`pretrain.py`: Script for pretraining the CMSR model with specified configurations, saving checkpoints, and logging validation results.

### CMSR Model Architecture

- **Pretrain** (single encoder layer, without MLP adapter)

```
CMSR(
  (item_embedding): Embedding(31126, 768)
  (position_embedding): Embedding(50, 768)
  (cmsr_encoder): Encoder(
    (layers): ModuleList(
      (0): EncoderLayer(
        (attention): MultiHeadAttention(
          (w_q): Linear(in_features=768, out_features=768, bias=True)
          (w_k): Linear(in_features=768, out_features=768, bias=True)
          (w_v): Linear(in_features=768, out_features=768, bias=True)
          (softmax): Softmax(dim=-1)
          (attn_dropout): Dropout(p=0.5, inplace=False)
          (w_out): Linear(in_features=768, out_features=768, bias=True)
        )
        (dropout1): Dropout(p=0.5, inplace=False)
        (norm1): LayerNorm()
        (ffn): FeedForward(
          (linear1): Linear(in_features=768, out_features=256, bias=True)
          (linear2): Linear(in_features=256, out_features=768, bias=True)
        )
        (dropout2): Dropout(p=0.5, inplace=False)
        (norm2): LayerNorm()
      )
    )
  )
  (LayerNorm): LayerNorm()
  (dropout): Dropout(p=0.5, inplace=False)
)
Trainable parameters: 2799616
```

- **Finetune** (single encoder layer, with MLP adapter)

```python
CMSR(
  (item_embedding): Embedding(3252, 768)
  (position_embedding): Embedding(50, 768)
  (cmsr_encoder): Encoder(
    (layers): ModuleList(
      (0): EncoderLayer(
        (attention): MultiHeadAttention(
          (w_q): Linear(in_features=768, out_features=768, bias=True)
          (w_k): Linear(in_features=768, out_features=768, bias=True)
          (w_v): Linear(in_features=768, out_features=768, bias=True)
          (mlp_q): MLPAdapter(
            (adapter): Sequential(
              (0): Linear(in_features=768, out_features=256, bias=True)
              (1): GELU()
              (2): Dropout(p=0.1, inplace=False)
              (3): Linear(in_features=256, out_features=768, bias=True)
              (4): Dropout(p=0.1, inplace=False)
            )
          )
          (mlp_k): MLPAdapter(
            (adapter): Sequential(
              (0): Linear(in_features=768, out_features=256, bias=True)
              (1): GELU()
              (2): Dropout(p=0.1, inplace=False)
              (3): Linear(in_features=256, out_features=768, bias=True)
              (4): Dropout(p=0.1, inplace=False)
            )
          )
          (softmax): Softmax(dim=-1)
          (attn_dropout): Dropout(p=0.5, inplace=False)
          (w_out): Linear(in_features=768, out_features=768, bias=True)
        )
        (dropout1): Dropout(p=0.5, inplace=False)
        (norm1): LayerNorm()
        (ffn): FeedForward(
          (linear1): Linear(in_features=768, out_features=256, bias=True)
          (linear2): Linear(in_features=256, out_features=768, bias=True)
        )
        (dropout2): Dropout(p=0.5, inplace=False)
        (norm2): LayerNorm()
      )
    )
  )
  (LayerNorm): LayerNorm()
  (dropout): Dropout(p=0.5, inplace=False)
  (loss): CrossEntropyLoss()
)
Trainable parameters: 3588098
```

## Reference

https://github.com/ChenMetanoia/CATSR-KDD/tree/main

https://github.com/RUCAIBox/RecBole/tree/master

https://recbole.io/docs/user_guide/config_settings.html
