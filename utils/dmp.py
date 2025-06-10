import numpy as np
import matplotlib.pyplot as plt

T = 1.0
alpha_z, beta_z, alpha_x = 25.0, 6.25, 1.0
y0, g = 0.0, 1.0
dt = 0.01
time = np.arange(0, T, dt)

# Step 1: minimum jerk trajectory
s = time / T
y_d = y0 + (g - y0)*(10*s**3 - 15*s**4 + 6*s**5)
dy_d = np.gradient(y_d, dt)
ddy_d = np.gradient(dy_d, dt)

# Step 2: phase variable x
x = np.exp(-alpha_x * time)

# Step 3: desired forcing term
fd = T**2 * ddy_d - alpha_z*(beta_z*(g - y_d) - T*dy_d)

# Step 4: learn weights
N = 10
c = np.exp(-alpha_x * np.linspace(0, 1, N))
h = 1.0 / (0.65*(c[1:] - c[:-1])**2).mean()
psi = np.exp(-h*(x.reshape(-1,1) - c.reshape(1,-1))**2)
psi_sum = np.sum(psi, axis=1, keepdims=True)
X = (psi / psi_sum) * x.reshape(-1,1)
w = np.linalg.lstsq(X, fd, rcond=None)[0]

# Step 5: simulate
y, z, x_sim = y0, 0.0, 1.0
Y = []
for t in time:
    psi = np.exp(-h*(x_sim - c)**2)
    f = (np.dot(psi, w) / np.sum(psi)) * x_sim
    dz = alpha_z*(beta_z*(g - y) - z) + f
    dy = z
    dx = -alpha_x * x_sim
    z += dz * dt / T
    y += dy * dt / T
    x_sim += dx * dt / T
    Y.append(y)

plt.plot(time, y_d, label="Demo")
plt.plot(time, Y, label="DMP")
plt.legend(); plt.xlabel("Time"); plt.ylabel("y")
plt.show()
