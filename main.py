import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define constants globally
G = 6.67430e-11  # Gravitational constant
c = 299792458    # Speed of light

def schwarzschild_geodesic(t, y, M):
    r, pr, phi, pphi, tau = y
    rs = 2 * G * M / c**2  # Schwarzschild radius
    
    f = 1 - rs / r
    if r <= rs:  # Avoid singularity
        return [0, 0, 0, 0, 0]

    # Updated differential equations
    drdt = pr
    dphidt = pphi / (r**2)
    dprdt = -G * M / (r**2) + (pphi**2) / (r**3) * (1 - rs / r)
    dpphidt = 0
    dtaudt = np.sqrt(f + (pr**2) / (c**2) * (1 / f))

    return [drdt, dprdt, dphidt, dpphidt, dtaudt]

def calculate_time_dilation(r, M):
    rs = 2 * G * M / c**2  # Schwarzschild radius
    if r <= rs:  # Avoid singularity
        return 0
    return np.sqrt(1 - rs / r)

def update_particle(particle, M, dt):
    y0 = [particle['position'], particle['velocity'], particle['phi'], particle['pphi'], 0]
    sol = solve_ivp(schwarzschild_geodesic, [0, dt], y0, args=(M,), method='RK45')

    if len(sol.y[0]) > 0:
        particle['position'] = sol.y[0][-1]
        particle['velocity'] = sol.y[1][-1]
        particle['proper_time'] += sol.y[4][-1]

    particle['phi'] = sol.y[2][-1] 
    particle['time_dilation'] = calculate_time_dilation(particle['position'], M)

    return particle

# Main simulation loop with visualization
running = True
initial_pphi = 1e12 
particles = [{'position': 1e11, 'velocity': 0, 'proper_time': 0, 'phi': 0, 'pphi': initial_pphi}]
star_mass = 1.989e+30
dt = 100
iterations = 0
particle_positions = []
particle_phis = []


while running and iterations < 500:
    for particle in particles:
        particle = update_particle(particle, star_mass, dt)
        particle_positions.append(particle['position'])
        particle_phis.append(particle['phi'])  # Store the current phi value
        if particle['position'] < 2 * G * star_mass / c**2:
            running = False
    iterations += 1


# After simulation, for plotting
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

x_positions = [r * np.sin(phi) for r, phi in zip(particle_positions, particle_phis)]
y_positions = [r * np.cos(phi) for r, phi in zip(particle_positions, particle_phis)]
z_positions = [0] * len(x_positions)  # Assuming motion in a plane

ax.plot(x_positions, y_positions, z_positions)
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
plt.title('3D Trajectory of a Particle in Schwarzschild Spacetime')
plt.show()
