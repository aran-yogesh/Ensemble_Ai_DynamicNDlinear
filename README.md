# DynamicNdLinear

DynamicNdLinear is a general-purpose alternative to `nn.Linear` designed specifically for structured, multidimensional data. Unlike traditional fully connected layers that flatten input and treat every feature equally, DynamicNdLinear learns to project and blend features **axis-wise**, dynamically selecting which dimensions to emphasize based on the input.

---

## 🔍 Motivation

Standard `nn.Linear` layers ignore the structure of multidimensional inputs (e.g., images, spectrograms, gene matrices). Flattening removes spatial or semantic relationships between axes. This leads to information loss and reduced generalization.

DynamicNdLinear preserves structure and improves performance by applying **axis-specific linear projections gated by learned attention weights**.

---

##  How It Works (Mathematics)

Given input `X ∈ ℝ^(B × D1 × D2 × ... × Dn)`:

1. **Gating**:

   - Flatten X: `X_flat = X.view(B, -1)`
   - Gate vector: `gates = Softmax(Linear(ReLU(Linear(X_flat)))) ∈ ℝ^(B × n)`

2. **Axis-wise Projections**:
   For each axis `i`:

   - Permute to bring axis `i` last.
   - Apply projection: `X_i' = X_i @ W_i + b_i`
   - Reshape and reverse permute.

3. **Blending**:

   ```
   X_i_final = gate[i] * X_i' + (1 - gate[i]) * X
   ```

This process lets the model **adaptively select** which axes to transform per sample.

---

##  Benefits Over StaticNdLinear

| Feature     | StaticNdLinear   | DynamicNdLinear              |
| ----------- | ---------------- | ---------------------------- |
| Gating      | No               | Yes (sample-wise)            |
| Adaptivity  | Fixed transforms | Context-sensitive            |
| Performance | Medium           | Higher (for structured data) |

StaticNdLinear uses the same transformation across samples and relies on fixed-order axis projections. In contrast, DynamicNdLinear uses **soft gates** to modulate axis importance dynamically, making it **input-sensitive**.

---

## 📊 Performance (CIFAR-10)

Both models were trained for 10 epochs on CIFAR-10:

### DynamicNdLinear

- **Accuracy**: 69.37%
- **Highlights**: Excellent performance on automobiles (89.63%), ships (85.4%), and frogs (72.3%)

### StaticNdLinear

- **Accuracy**: 41.74%
- **Struggled** with class separability and generalization

---

## 👍 Best Use Cases

DynamicNdLinear is ideal for:

- Image and vision tasks (spatial axis interaction)
- Audio (time × frequency structures)
- Bioinformatics (gene × sample)
- Time-series (sensor × time)
- Any task with **semantic axes**

---

## 📖 Sample Code (CIFAR-10)

See [dynamic\_ndlinear.py](./dynamic_ndlinear.py) and [static\_ndlinear.py](./static_ndlinear.py) for complete training and evaluation pipelines.

You can run the dynamic version using:

```python
python dynamic_ndlinear.py
```

And static version using:

```python
python static_ndlinear.py
```

Both scripts include:

- Training loop
- Evaluation with classification report and confusion matrix
- Curve plotting for loss and accuracy

---

## 📚 Summary

DynamicNdLinear generalizes `nn.Linear` to structured tensors by learning to:

- Decide which axis to transform (via gates)
- Apply axis-specific linear projections
- Blend transformed and original data adaptively

This results in better generalization and interpretability for data with inherent multi-axis semantics.

---

