import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipe
from scipy.optimize import root_scalar
from scipy.integrate import solve_ivp

# ==========================================
# 1. THE MATHEMATICAL SOLVER
# ==========================================
def get_elastica_parameters(v):
    """
    Computes uniform pressure and wrinkle amplitude for a given velocity 
    parameter v = A/L. Includes precision safeguards.
    """
    def objective(m, v_target):
        return (2.0 * ellipe(m) / ellipk(m)) - 1.0 - v_target
        
    try:
        # Bracket extended to the absolute limit of float64 stability
        solution = root_scalar(objective, args=(v,), bracket=[1e-8, 1.0 - 1e-14], method='brentq')
        m_root = solution.root
        
        uniform_pressure = 16.0 * (ellipk(m_root)**2)
        theta_0_rad = 2.0 * np.arcsin(np.sqrt(m_root))
        
        return uniform_pressure, np.degrees(theta_0_rad), theta_0_rad
    except ValueError:
        # If machine precision is exceeded, return Not-a-Number (NaN)
        return np.nan, np.nan, np.nan

# ==========================================
# 2. SPATIAL ODE SYSTEM
# ==========================================
def elastica_kinematics(s_bar, state, uniform_pressure):
    """System of 4 first-order ODEs mapping the geometry."""
    theta, omega_bar, x_bar, y_bar = state
    
    dtheta_ds = omega_bar
    domega_ds = -uniform_pressure * np.sin(theta)
    dx_ds = np.cos(theta)
    dy_ds = np.sin(theta)
    
    return [dtheta_ds, domega_ds, dx_ds, dy_ds]

# ==========================================
# 3. FIGURE LAYOUT & EXECUTION
# ==========================================
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, 0]) 
ax2 = fig.add_subplot(gs[0, 1]) 
ax3 = fig.add_subplot(gs[1, :]) 

# --- Panel 1 & 2: Parameter Sweeps ---
# Array restricted to evaluable machine limits [-0.85, 0.99]
v_array = np.linspace(-0.85, 0.99, 300)
pressure_array = np.zeros_like(v_array)
amplitude_array = np.zeros_like(v_array)

for i, v in enumerate(v_array):
    p_val, amp_deg, _ = get_elastica_parameters(v)
    pressure_array[i] = p_val
    amplitude_array[i] = amp_deg

# Plot Uniform Pressure
ax1.plot(v_array, pressure_array, 'b-', linewidth=2.5)
ax1.set_title('Uniform Pressure vs. Velocity Parameter', fontsize=13)
ax1.set_xlabel('Velocity Parameter ($A/L$)', fontsize=11)
ax1.set_ylabel('Dimensionless Uniform Pressure ($\lambda L^2$)', fontsize=11)
ax1.set_xlim(-1, 1)
ax1.set_ylim(0, 500)
ax1.grid(True, linestyle='--', alpha=0.7)

# Plot Wrinkle Amplitude
ax2.plot(v_array, amplitude_array, 'r-', linewidth=2.5)
ax2.set_title('Wrinkle Amplitude vs. Velocity Parameter', fontsize=13)
ax2.set_xlabel('Velocity Parameter ($A/L$)', fontsize=11)
ax2.set_ylabel('Maximum Angle $\\theta_0$ (Degrees)', fontsize=11)
ax2.set_xlim(-1, 1)
ax2.set_ylim(0, 190)
ax2.grid(True, linestyle='--', alpha=0.7)

# --- Panel 3: Physical Spatial Shapes ---
# Defined strict limits within machine precision capacity
sample_velocities = [0.8, 0.4, 0.0, -0.6]
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']

for v, color in zip(sample_velocities, colors):
    p_val, _, theta_0_rad = get_elastica_parameters(v)
    
    if np.isnan(p_val):
        continue # Skip if precision exceeded
        
    omega_0 = np.sqrt(2.0 * p_val * (1.0 - np.cos(theta_0_rad)))
    init_state = [0.0, omega_0, 0.0, 0.0]
    
    sol = solve_ivp(elastica_kinematics, [0, 1], init_state, args=(p_val,), 
                    t_eval=np.linspace(0, 1, 500), method='RK45', rtol=1e-8, atol=1e-8)
    
    ax3.plot(sol.y[2], sol.y[3], color=color, linewidth=2, label=f'v = {v}')
    ax3.plot(sol.y[2][-1], sol.y[3][-1], marker='o', color=color) 

ax3.set_title('Physical Shape of Wrinkles at Selected Velocity Parameters', fontsize=13)
ax3.set_xlabel('Horizontal Position ($\\xi/L$)', fontsize=11)
ax3.set_ylabel('Vertical Position ($\eta/L$)', fontsize=11)
ax3.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax3.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.axis('equal') 

plt.tight_layout()
plt.show()