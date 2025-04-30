<p align="center">
  <img src="image.png" alt="Dynamic NdLinear Logo" width="300"/>
</p>


# DynamicNdLinear

> **Note:** This project extends and builds upon the original *NdLinear* implementation by the respective authors. All architectural credits for the base NdLinear layer go to them. This repository showcases our research experiment focused on the **DynamicNdLinear** variant and its comparative evaluation using the CIFAR-10 dataset.

DynamicNdLinear is a general-purpose alternative to `nn.Linear` for structured, multidimensional input data. Inspired by the limitations of traditional flatten-and-feed approaches in neural networks, this layer dynamically applies axis-specific transformations conditioned on input context.

## 🔍 Motivation

Standard `nn.Linear` layers flatten multidimensional data, discarding axis-specific structure (e.g., temporal, spatial, or channel-wise features). While this may suffice for some tasks, it often loses crucial inductive biases present in structured inputs such as:

- Spectrograms (Time × Frequency)
- Multivariate time series
- Bioinformatics (gene × cell, modality × token)
- Audio and NLP tensor representations

DynamicNdLinear is designed to **preserve and enhance axis-wise interactions** through learned, gated linear projections along each dimension—adaptively selected based on the input.

##  How It Works

Given an input tensor `X ∈ ℝ^(B × D1 × D2 × ... × Dn)`, DynamicNdLinear performs the following steps:

1. **Gating**:

   - Flatten input tensor and compute attention weights (gates) `g ∈ ℝ^(B × n)` indicating the importance of each axis for transformation.
   - The gate dynamically modulates the update strength for each axis.

2. **Axis-Wise Transformation**:

   - For each axis `i`, apply a learned linear transform `X @ Wi + bi`.
   - Perform appropriate `permute`, `reshape`, and reverse operations to isolate the axis.

3. **Blending**:

   - Combine transformed and original data using gate-weighted blending:
     ```
     X_i' = gate_i * transformed_i + (1 - gate_i) * X
     ```

The result is a representation that selectively projects and mixes axis features while preserving structure.

##  Mathematical View

Let `X ∈ ℝ^(B × D1 × D2 × ... × Dn)` be the input.
For each axis `i` (from 1 to `n`), we learn:

- A projection matrix `W_i ∈ ℝ^(D_i × H_i)`
- A bias `b_i ∈ ℝ^(H_i)`

The axis transformation is:

```
X' = softmax(g(X)) ⊙ (X @ W_i + b_i) + (1 - softmax(g(X))) ⊙ X
```

Here:

- `g(X)` is a gate learned via a feedforward network.
- `⊙` denotes element-wise multiplication broadcast across dimensions.

This enables **input-dependent modulation** of how strongly each axis is transformed.

##  Ideal Use Cases

DynamicNdLinear is especially suited for:

- **Image & Vision Data** (e.g., CIFAR-10, MNIST, segmentation tensors)
- **Audio Processing** (e.g., spectrogram classification)
- **Multimodal Fusion** (e.g., video + audio + text)
- **Biological Data** (e.g., genomics, scRNA-seq matrices)
- **Time-Series Tensors** (e.g., EEG, multivariate sensors)

##  CIFAR-10 Benchmark Results

### StaticNdLinear (Baseline)

```text
Epoch 10 — Loss: 1.6933, Acc: 0.4226
StaticNdLinear — Accuracy: 0.4174
```

### DynamicNdLinear (Ours)

```text
Epoch 10 — Loss: 0.7910, Acc: 0.7268
CIFAR-10 DynamicNdLinear — Accuracy: 0.6937
```

| Model           | Final Accuracy |
| --------------- | -------------- |
| StaticNdLinear  | 41.74%         |
| DynamicNdLinear | **69.37%**     |

##  Code Organization

This repository includes:

- `DynamicNdLinear`: The main module for dynamic axis-aware transformation.
- `CIFARDynamicClassifier`: CNN + DynamicNdLinear based CIFAR-10 model.
- `StaticNdLinear`: A baseline implementation of the fixed NdLinear layer.
- `run_dynamic_ndlinear()`: Script to train and evaluate CIFAR model.

All code is currently provided in `.ipynb` format. You may convert it to `.py` or module form as needed.

##  Research Contribution

This experiment introduces a **dynamic, gated** mechanism into axis-wise tensor projection. Unlike traditional layers, it adaptively learns which axis matters per input, improving both interpretability and performance.

Future directions may include:

- Extension to Transformer-like models.
- Hybrid versions with attention.
- Gated spatial-temporal modeling for video/audio.

---




