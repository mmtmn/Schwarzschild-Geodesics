import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

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
    y0 = [particle['position'], particle['velocity'], 0, 0, 0]
    sol = solve_ivp(schwarzschild_geodesic, [0, dt], y0, args=(M,), method='RK45')

    if len(sol.y[0]) > 0:
        particle['position'] = sol.y[0][-1]
        particle['velocity'] = sol.y[1][-1]
        particle['proper_time'] += sol.y[4][-1]  # Accumulate proper time

    particle['time_dilation'] = calculate_time_dilation(particle['position'], M)

    return particle

# Main simulation loop with visualization
running = True
particles = [{'position': 1e11, 'velocity': 0, 'proper_time': 0}]
star_mass = 1.989e+30
dt = 100
iterations = 0
particle_positions = []

while running and iterations < 500:  # Increase iterations for a better plot
    for particle in particles:
        particle = update_particle(particle, star_mass, dt)
        particle_positions.append(particle['position'])
        if particle['position'] < 2 * G * star_mass / c**2:
            running = False
    iterations += 1

# Plotting the particle's position over time
plt.figure(figsize=(10, 6))
plt.plot(particle_positions)
plt.xlabel('Time Steps')
plt.ylabel('Distance from Star (meters)')
plt.title('Particle Falling Towards a Star Over Time')
plt.grid(True)
plt.show()
