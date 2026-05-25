# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 22:00:02 2026

@author: alexi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipe, ellipj
from scipy.optimize import brentq
from scipy.integrate import cumulative_trapezoid

# =========================================================
# PART 1: NUMERICAL ENERGY SOLVER
# =========================================================

def action_variable(m, uniform_pressure):
    m = np.clip(m, 1e-12, 1.0 - 1e-12)
    K_val = ellipk(m)
    E_val = ellipe(m)
    return 16.0 * np.sqrt(uniform_pressure) * (E_val - (1.0 - m) * K_val)

def objective_function(m, uniform_pressure, I_0):
    return action_variable(m, uniform_pressure) - I_0

def simulate_tracking(arclength_array, uniform_pressure_array, initial_local_energy):
    N = len(arclength_array)
    local_energy_array = np.full(N, np.nan)
    
    local_energy_array[0] = initial_local_energy
    uniform_pressure_0 = uniform_pressure_array[0]
    
    m_0 = (initial_local_energy + uniform_pressure_0) / (2.0 * uniform_pressure_0)
    I_0 = action_variable(m_0, uniform_pressure_0)
    
    m_min = 1e-8
    m_max = 1.0 - 1e-8
    
    for i in range(1, N):
        current_pressure = uniform_pressure_array[i]
        current_s = arclength_array[i]
        
        try:
            m_i = brentq(objective_function, m_min, m_max, args=(current_pressure, I_0))
            local_energy_array[i] = current_pressure * (2.0 * m_i - 1.0)
            
        except ValueError:
            print(f"Tracking failure registered at step {i}, arclength s = {current_s:.4f}")
            break 
            
    return local_energy_array, I_0

# =========================================================
# PART 2: INITIALIZATION AND EXECUTION
# =========================================================

N_steps = 2500 
arclength = np.linspace(0, 15, N_steps)

uniform_pressure = np.linspace(3.0, 0.4, N_steps)
initial_energy = 0.0 

curvature = 0.15
phi_array = curvature * arclength

print("Executing numerical spatial marching algorithm...")
local_energy, invariant_I0 = simulate_tracking(arclength, uniform_pressure, initial_energy)

# =========================================================
# PART 3: ANALYTICAL GEOMETRIC RECONSTRUCTION
# =========================================================
print("Reconstructing exact tracking geometry from calculated energy...")

valid_indices = ~np.isnan(local_energy)
s_valid = arclength[valid_indices]
lambda_valid = uniform_pressure[valid_indices]
E_valid = local_energy[valid_indices]
phi_valid = phi_array[valid_indices]

# --- SEPARATRIX EXTRACTION ---
failure_occurred = len(s_valid) < len(arclength)
s_fail = arclength[len(s_valid)] if failure_occurred else None

m_array = (E_valid + lambda_valid) / (2.0 * lambda_valid)

frequency_array = np.sqrt(lambda_valid)
phase_array = cumulative_trapezoid(frequency_array, s_valid, initial=0)

sn_wave, _, _, _ = ellipj(phase_array, m_array)
alpha_array = 2.0 * np.arcsin(np.sqrt(m_array) * sn_wave)
theta_array = phi_valid + alpha_array

x = cumulative_trapezoid(np.cos(phi_valid), s_valid, initial=0)
y = cumulative_trapezoid(np.sin(phi_valid), s_valid, initial=0)

xi = cumulative_trapezoid(np.cos(theta_array), s_valid, initial=0)
eta = cumulative_trapezoid(np.sin(theta_array), s_valid, initial=0)

# Extract final Cartesian coordinates prior to failure
if failure_occurred:
    xi_fail, eta_fail = xi[-1], eta[-1]

# =========================================================
# PART 4: VISUALIZATION (Explicit Boundary Marking)
# =========================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))

# Plot 1: Adiabatic Energy Evolution
ax1.plot(arclength, uniform_pressure, 'r--', linewidth=2, label='Upper Bound ($+\\lambda$)')
ax1.plot(arclength, -uniform_pressure, 'b--', linewidth=2, label='Lower Bound ($-\\lambda$)')
ax1.plot(arclength, local_energy, 'k-', linewidth=1.5, label='Local Energy ($E$)')

if failure_occurred:
    ax1.axvline(x=s_fail, color='m', linestyle='-.', linewidth=2, label=f'Separatrix ($s_{{fail}}={s_fail:.2f}$)')

ax1.set_title('Adiabatic Energy Evolution ($E$ vs $s$)')
ax1.set_xlabel('Arclength ($s$)')
ax1.set_ylabel('Energy')
ax1.legend()
ax1.grid(True)
ax1.set_xlim([0, arclength[-1]])

# Plot 2: Absolute Angular Tracking
ax2.plot(s_valid, theta_array, 'k-', linewidth=1, label='Material Heading $\\theta(s)$')
ax2.plot(s_valid, phi_valid, 'g--', linewidth=2, label='Target Heading $\\phi(s)$')

alpha_max = 2.0 * np.arcsin(np.sqrt(m_array))
ax2.plot(s_valid, phi_valid + alpha_max, 'r:', linewidth=1.5, label='Upper Envelope ($\\phi + \\alpha_{max}$)')
ax2.plot(s_valid, phi_valid - alpha_max, 'r:', linewidth=1.5, label='Lower Envelope ($\\phi - \\alpha_{max}$)')

if failure_occurred:
    ax2.axvline(x=s_fail, color='m', linestyle='-.', linewidth=2, label='Numerical Collapse')

ax2.set_title('Absolute Angular Tracking ($\\theta$ vs $s$)')
ax2.set_xlabel('Arclength ($s$)')
ax2.set_ylabel('Absolute Angle (radians)')
ax2.legend()
ax2.grid(True)
ax2.set_xlim([0, arclength[-1]])

# Plot 3: Global Cartesian Geometry
ax3.plot(xi, eta, 'k-', linewidth=1.2, label='Tracker Coordinates ($\\xi$, $\\eta$)')
ax3.plot(x, y, 'r--', linewidth=2, label='Target Coordinates ($x$, $y$)')

if failure_occurred:
    ax3.plot(xi_fail, eta_fail, 'mX', markersize=10, label='Numerical Collapse')

ax3.set_title('Global Coordinate Frame')
ax3.set_xlabel('Global Coordinate ($x$ / $\\xi$)')
ax3.set_ylabel('Global Coordinate ($y$ / $\\eta$)')
ax3.axis('equal') 
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()