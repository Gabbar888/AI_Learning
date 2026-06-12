import numpy as np
import matplotlib.pyplot as plt

def f(a,b):
    return a**2+3*a+6*(b**2)

def ana_grad(a,b):
    return 2*a+3, 12*b

def num_grad(a,b):
    h = 1e-5
    grad_x = (f(a+h,b)-f(a-h,b))/(2*h)
    grad_y = (f(a,b+h)-f(a,b-h))/(2*h)
    return grad_x,grad_y

fig = plt.figure(figsize=(14, 5))

# Plot 1: the function as a 3D surface
ax1 = fig.add_subplot(121, projection='3d')
X = np.linspace(-3, 3, 50)
Y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(X, Y)
Z = f(X, Y)
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax1.set_title('f(x, y) = x² + 3x + 6y²')
ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('f(x, y)')

# Plot 2: gradient field (arrows pointing in the direction of steepest ascent)
ax2 = fig.add_subplot(122)
x_coords = np.linspace(-3, 3, 15)
y_coords = np.linspace(-3, 3, 15)
X2, Y2 = np.meshgrid(x_coords, y_coords)
U_analytical, V_analytical = ana_grad(X2, Y2)
U_numerical, V_numerical = num_grad(X2, Y2)

# analytical gradient as blue arrows
ax2.quiver(X2, Y2, U_analytical, V_analytical, color='blue', alpha=0.6, label='Analytical')
# numerical gradient as red arrows — they should align
ax2.quiver(X2, Y2, U_numerical, V_numerical, color='red', alpha=0.4,scale=400, label='Numerical')
ax2.set_title('Gradient field — analytical (blue) vs numerical (red)')
ax2.set_xlabel('x'); ax2.set_ylabel('y')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig('gradient_comparison.png', dpi=100)
plt.show()

