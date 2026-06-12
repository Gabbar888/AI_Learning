Shape, Size and ndim of np.array([1,2,3]) are (3,) , 3 , 1
Shape, Size and ndim of np.array([[1,2,3],[4,5,6]]) are (2,3) , 6 , 2

np.zeros() by default stores the zeros in float64

# NumPy Notes

Working notes on NumPy concepts learned while implementing softmax + cross-entropy from scratch. Only the durable patterns — not function-by-function reference.

---

## Axes and reductions

The axis you reduce *over* is the one that disappears. If `x.shape == (B, C)`, then `x.sum(axis=-1).shape == (B,)` — the C dimension collapsed.

`axis=-1` means "the last axis." Useful because the same code works whether your input is shape `(C,)`, `(B, C)`, or `(N, B, C)` — you always operate on the trailing dimension.

## `keepdims=True`

When you reduce along an axis and need to broadcast the result back against the original array, use `keepdims=True`. It preserves the reduced axis as size 1 instead of removing it.

```
x.shape == (B, C)
x.sum(axis=-1).shape == (B,)              # 1-D, may not broadcast back
x.sum(axis=-1, keepdims=True).shape == (B, 1)   # 2-D, broadcasts cleanly
```

Rule of thumb: if the very next thing you do is divide/subtract the original array by the reduction, use `keepdims=True`.

## Broadcasting

Two shapes are aligned **right-to-left**. For each pair of aligned dimensions, sizes must either be equal OR one must be 1. Missing leading dimensions count as 1.

```
(B, C) and (C,)     → align as (B, C) vs (_, C) → works if Cs match
(B, C) and (B,)     → align as (B, C) vs (_, B) → fails unless C == B
(B, C) and (B, 1)   → works, the 1 stretches to C
```

The classic failure: dividing `(B, C)` by `(B,)`. The trailing dims are `C` and `B`, which only match if they're equal. Either errors or silently produces wrong values when `B == C`. Fix with `keepdims=True` on the reduction so you get `(B, 1)`.

## Fancy indexing

The pattern for "pick one column per row":

```
log_probs[np.arange(B), targets]
```

Two index arrays get zipped position-by-position: row 0 picks column `targets[0]`, row 1 picks column `targets[1]`, etc. Result is 1-D of length B. Replaces a Python for-loop and runs in compiled C — much faster on large batches.

## Bracket nesting = dimension count

The number of nested brackets equals the number of dimensions of the resulting array:

```
np.array(5)        → shape ()       (0-D, scalar)
np.array([5])      → shape (1,)     (1-D)
np.array([[5]])    → shape (1, 1)   (2-D)
np.array([[[5]]])  → shape (1, 1, 1) (3-D)
```

`np.array([1, 3])` is 1-D shape `(2,)`. `np.array([[1, 3]])` is 2-D shape `(1, 2)`. Not interchangeable — functions expecting batched inputs will reject the 1-D version.

## Adding an axis to an array

Two equivalent ways to turn shape `(B,)` into `(B, 1)`:

```
x[:, None]         # None and np.newaxis are the same
x[:, np.newaxis]
```

## `np.mean` vs `np.average`

`np.mean(x)` is unweighted arithmetic mean. `np.average(x, weights=w)` is weighted. If you don't need weights, use `np.mean` — clearer intent and slightly faster.

## The shape-printing habit

When any non-trivial array operation is unexpected: print `.shape` (and sometimes `.dtype`) of every intermediate. 90% of NumPy bugs are shape bugs. Two seconds of printing saves an hour of confused debugging. Eventually this becomes mental, but write it explicitly while learning.