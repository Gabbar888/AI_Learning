# PyTorch Notes

Working notes on PyTorch behavior learned while verifying NumPy implementations against PyTorch. Focused on patterns that recur across the project, not function-by-function reference.

---

## Dtype contract for loss functions

`F.cross_entropy(logits, targets)` is strict about dtypes:

- **`logits`** must be floating-point (`float32` or `float64`)
- **`targets`** must be `torch.long` (int64)

Mismatches throw clear errors. Always cast explicitly:

```python
logits_t = torch.tensor(logits, dtype=torch.float32)
targets_t = torch.tensor(targets, dtype=torch.long)
```

## The integer-array gotcha

When you create a NumPy array with all-integer literals (e.g., `np.array([[1000, 999, 998]])`), the dtype is `int64`. Converting that to a PyTorch tensor gives a `Long` tensor. `cross_entropy` then refuses with:

```
NotImplementedError: "log_softmax_lastdim_kernel_impl" not implemented for 'Long'
```

Two fixes (use both for safety):

1. Write logits with decimal points: `np.array([[1000.0, 999.0, 998.0]])`
2. Cast explicitly at the boundary: `torch.tensor(logits, dtype=torch.float32)`

## Reading PyTorch errors

PyTorch errors name three things, in order:

1. **The operation that failed** — often a low-level kernel name like `log_softmax_lastdim_kernel_impl`
2. **The dtype or shape that was wrong** — `'Long'`, `expected scalar type Long but found Int`, etc.
3. **What was expected** — implied or stated

Parse those three and the fix is usually obvious. Don't be intimidated by the kernel names; treat them as breadcrumbs to *where* in the pipeline the problem is.

## `cross_entropy` takes logits, not probabilities

`F.cross_entropy(logits, targets)` expects raw logits as input. It internally runs `log_softmax + NLL loss` as a single fused, numerically stable kernel.

Never pass `softmax(logits)` to it — that would do an unnecessary exp→log roundtrip and lose numerical stability. If you have probabilities and need cross-entropy, take the log yourself and use `F.nll_loss`.

## Standard shape contract

For the non-segmentation case:

- **logits**: shape `(B, C)` — batch size, number of classes
- **targets**: shape `(B,)` — integer class indices in `[0, C-1]`, 0-indexed

The batch dimension is always required, even when B=1. For a single example, reshape to `(1, C)` rather than passing a 1-D `(C,)`.

## Extracting scalars for comparison

A 0-D PyTorch tensor (e.g., the output of `cross_entropy` with default `reduction='mean'`) needs `.item()` to convert to a Python float:

```python
loss_value = F.cross_entropy(logits, targets).item()
```

Required when comparing against NumPy scalars via `np.allclose` — without it, you're comparing a tensor object to a number.
