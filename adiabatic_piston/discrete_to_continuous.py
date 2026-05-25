import numpy as np
import matplotlib.pyplot as plt

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    plt.style.use('seaborn-whitegrid')

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.edgecolor': '#cccccc',
    'legend.facecolor': 'white',
    'legend.framealpha': 1.0
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle('Transition from Discrete Jumps to Continuous Decay', fontsize=14)

# ---------------------------------------------------------
# Subplot 1: Fast Scale (Steps: 5)
# ---------------------------------------------------------
eps_fast = 0.15
t0_fast = -1.3

# Explicit collision times mapping the physics of the slowing piston
t_fast = [-1.3, 0.0, 1.15, 2.65, 4.4, 6.6, 8.5]

# Enforce exact intersection: post-collision energy lies perfectly on the continuous envelope
E_fast = [np.exp(-1.5 * eps_fast * (t - t0_fast)) for t in t_fast]

ax1.step(t_fast, E_fast, where='post', color='#2c3e50', linewidth=2, 
         label=f'Discrete Energy $\\mathcal{{E}}_n$\n($\\varepsilon = {eps_fast}$)')

t_cont_f = np.linspace(-1.3, 8.5, 100)
E_cont_f = np.exp(-1.5 * eps_fast * (t_cont_f - t0_fast))

ax1.plot(t_cont_f, E_cont_f, '--', color='#e74c3c', linewidth=2,
         label='Continuous $\\overline{\\mathcal{{E}}}_\\varepsilon(\\theta)$')

ax1.set_title(f'Fast Scale (Steps: {len(t_fast)-2})')
ax1.set_xlabel('Physical Time ($t$)')
ax1.set_ylabel('Energy ($E$)')
ax1.set_xlim(-1.3, 8.5)
ax1.set_ylim(0.2, 1.1)
ax1.set_xticks([-1.3, 0.0, 1.3, 2.7, 4.0, 5.3, 6.7, 8.0])
ax1.legend(loc='upper right')

# ---------------------------------------------------------
# Subplot 2: Slow Scale (Steps: 21)
# ---------------------------------------------------------
eps_slow = 0.03
theta_slow_list = [0.0]
E_slow_list = [1.0]

curr_theta = 0.0
curr_E = 1.0

# Integration across adiabatic time steps mapping to the same envelope
while curr_theta < 1.6:
    # Collision frequency decreases as energy drops (dt ~ 1/sqrt(E))
    d_theta = eps_slow * (1.3 / np.sqrt(curr_E))
    curr_theta += d_theta
    curr_E = np.exp(-1.5 * curr_theta)
    
    theta_slow_list.append(curr_theta)
    E_slow_list.append(curr_E)

ax2.step(theta_slow_list, E_slow_list, where='post', color='#2c3e50', linewidth=2,
         label=f'Discrete Energy $\\mathcal{{E}}_n$\n($\\varepsilon = {eps_slow}$)')

theta_cont_s = np.linspace(0.0, 1.6, 100)
E_cont_s = np.exp(-1.5 * theta_cont_s)

ax2.plot(theta_cont_s, E_cont_s, '--', color='#e74c3c', linewidth=2,
         label='Continuous $\\overline{\\mathcal{{E}}}_\\varepsilon(\\theta)$')

ax2.set_title(f'Slow Scale (Steps: {len(theta_slow_list)-1})')
ax2.set_xlabel('Slow Time ($\\theta = \\varepsilon t$)')
ax2.set_ylabel('Energy ($E$)')
ax2.set_xlim(0.0, 1.5)
ax2.set_ylim(0.2, 1.1)
ax2.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()