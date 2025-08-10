import math
import numpy as np
import matplotlib.pyplot as plt


# Objective: Plot various types of stability using equation of motion in state space

# System Properties

m = 2                   # Mass
k = -400                 # Stiffness
d = 4                   # Damping
T = 10                   # Total Time

# Initial Conditions 
x0_1 = 0
x0_2 = 0.1
x0 = np.array([x0_1, x0_2])

# Build A matrix from State space

A = np.array([[-d/m, -k/m], [1, 0]])

eig_val, eig_vec = np.linalg.eig(A)

# Eigen Value Decomposition

V = eig_vec 

b_vec = np.dot(np.linalg.inv(V), x0)

# print("Eigenvalues:", eig_val)
# print("b vector:", b_vec)

if (np.imag(eig_val[0]) !=0 and np.imag(eig_val[1]) !=0):
    if (np.real(eig_val[0])<0 and np.real(eig_val[1])<0):
        print("Fixed Point Type:  Stable Focus")
    elif (np.real(eig_val[0])==0 and np.real(eig_val[1])==0):
        print("Fixed Point Type:  Center")
    elif (np.real(eig_val[0])>0 and np.real(eig_val[1])>0):
        print("Fixed Point Type:  Unstable Focus")

if (np.imag(eig_val[0]) ==0 and np.imag(eig_val[1]) ==0):
    if (np.real(eig_val[0])<0 and np.real(eig_val[1])<0):
        print("Fixed Point Type:  Stable Node")
    elif (np.real(eig_val[0])>0 and np.real(eig_val[1])>0):
        print("Fixed Point Type:  Unstable Node")
    elif ((np.real(eig_val[0])>0 and np.real(eig_val[1])<0) or (np.real(eig_val[0])<0 and np.real(eig_val[1])>0)):
        print("Fixed Point Type:  Saddle")


# Time vector
t = np.linspace(0, T, 1000)

# Calculate the analytical solution using eigenvalue decomposition
# x(t) = sum(V[:, i] * b_vec[i] * exp(eig_val[i] * t)) for i = 0, 1
x_1 = np.zeros((len(t), 2), dtype=complex)

for i in range(len(eig_val)):
    
    # Alternative approach: explicit loop instead of np.outer
    state = np.zeros((len(t), 2), dtype=complex)
    
    for j in range(len(t)):
        # For each time point, calculate: exp(eig_val[i] * t[j]) * V[:, i] * b_vec[i]
        state[j, :] = np.exp(eig_val[i] * t[j]) * V[:, i] * b_vec[i]
    
    x_1 += state

# Take the real part since system is real
x_1 = np.real(x_1)

# Extract position and velocity
position = x_1[:, 0]
velocity = x_1[:, 1]

# Plotting State vs Time
plt.figure(figsize=(15, 10))

# Plot 1: Position vs Time
plt.plot(t, position, 'b-', linewidth=2, label='Position')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')

plt.plot(t, velocity, 'r-', linewidth=2, label='Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')

plt.grid()
plt.legend()
plt.title('Velocity and Position vs Time')
plt.show()

# Plot 2: Phase Plane Projection
plt.figure(figsize=(8, 8))
plt.plot(position, velocity, 'k-', linewidth=2, label='Trajectory')

# Mark start and end points
plt.plot(position[0], velocity[0], 'go', markersize=5, label='Start (t=0)')
plt.plot(position[-1], velocity[-1], 'ro', markersize=5, label='End (t=10)')

plt.xlabel('Position (m)')
plt.ylabel('Velocity (m/s)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.title('Phase Plane Projection (Velocity vs Position)')
plt.show()
