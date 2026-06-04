import numpy as np
import math
def mat_mul(a,b):
    if(a.shape[1] != b.shape[0]):
        print("Error: The Matrices do not conform")
    c = np.zeros((a.shape[0],b.shape[1]))
    for i in range(0,a.shape[0]):
        for j in range(0,b.shape[1]):
            for k in range(0,a.shape[1]):
                c[i,j] += a[i][k]*b[k][j]
    return c

def dot_prod(a,b):
    assert (a.shape == b.shape) and (a.ndim == 1)
    # c = np.zeros(a.shape)
    c = 0.0
    for i in range(0,a.shape[0]):
        c += a[i]*b[i]
    return c

def norm(a):
    c = dot_prod(a,a)
    return math.sqrt(c)
# a = np.array([1,2,3])
# print(a.shape)


# ---- Tests: verify against NumPy ----
if __name__ == "__main__":
    # Test 1: dot product
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    assert abs(dot_prod(a, b) - np.dot(a, b)) < 1e-9
    print(f"dot: my={dot_prod(a, b)}, numpy={np.dot(a, b)} ✓")

    # Test 2: norm
    v = np.array([3.0, 4.0])
    assert abs(norm(v) - np.linalg.norm(v)) < 1e-9
    print(f"norm: my={norm(v)}, numpy={np.linalg.norm(v)} ✓")

    # Test 3: matmul, small case
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])
    assert np.allclose(mat_mul(A, B), A @ B)
    print(f"matmul small ✓")
    
    # Test 4: matmul, non-square
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
    B = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # (3, 2)
    assert np.allclose(mat_mul(A, B), A @ B)
    print(f"matmul non-square ✓")

    # Test 5: matmul, larger random
    np.random.seed(42)
    A = np.random.randn(5, 7)
    B = np.random.randn(7, 3)
    assert np.allclose(mat_mul(A, B), A @ B)
    print(f"matmul random ✓")

    print("\nAll tests passed.")