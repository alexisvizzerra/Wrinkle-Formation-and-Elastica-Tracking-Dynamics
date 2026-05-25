
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import matplotlib.patches as patches

# ==========================================
# 1. DEFINE THE CURVES
# ==========================================
x = np.linspace(0, 4 * np.pi, 2000)
y_target = np.sin(0.5 * x)
y_tracker = y_target + 0.15 * np.sin(12 * x)

# ==========================================
# 2. INITIALIZE THE MAIN FIGURE
# ==========================================
fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(x, y_target, color='#1f77b4', linestyle='--', linewidth=2.5, label=r'Target Path')
ax.plot(x, y_tracker, color='black', linewidth=1.5, label=r'Irregular Tracker Path ')

ax.set_aspect('equal')
ax.grid(True, linestyle=':', alpha=0.6)
ax.set_xlim(0, 4 * np.pi)
ax.set_ylim(-1.5, 2.5)
ax.set_xticks([])
ax.set_yticks([])
ax.legend(loc='upper left', fontsize=12)

# ==========================================
# 3. CONSTRUCT THE MAGNIFIED INSET
# ==========================================
eval_x = 7 * np.pi / 12  # Chosen intersection point
eval_y = np.sin(0.5 * eval_x)

axins = zoomed_inset_axes(ax, zoom=5, loc='lower right')
axins.plot(x, y_target, color='#1f77b4', linestyle='--', linewidth=3)
axins.plot(x, y_tracker, color='black', linewidth=2)

x_window, y_window = 0.4, 0.4
axins.set_xlim(eval_x - x_window, eval_x + x_window)
axins.set_ylim(eval_y - y_window, eval_y + y_window)
axins.set_xticks([]); axins.set_yticks([])

patch = patches.Circle((0.5, 0.5), radius=0.5, transform=axins.transAxes, 
                       edgecolor='gray', facecolor='none', linewidth=1.5)
axins.add_patch(patch); axins.set_clip_path(patch)

# ==========================================
# 4. FINAL PINPOINT LABELING
# ==========================================
dy_tgt_dx = 0.5 * np.cos(0.5 * eval_x)
dy_trk_dx = dy_tgt_dx + 0.15 * 12 * np.cos(12 * eval_x) 
phi_deg = np.degrees(np.arctan(dy_tgt_dx))
theta_deg = np.degrees(np.arctan(dy_trk_dx))

# Reference Axis
axins.plot([eval_x - 0.2, eval_x + 0.2], [eval_y, eval_y], color='gray', linestyle='--', linewidth=1.5)

# Vectors
dx_vec = 0.3
axins.arrow(eval_x, eval_y, dx_vec, dx_vec * dy_tgt_dx, head_width=0.03, head_length=0.05, 
            fc='#1f77b4', ec='#1f77b4', linewidth=2, zorder=5)
axins.arrow(eval_x, eval_y, dx_vec, dx_vec * dy_trk_dx, head_width=0.03, head_length=0.05, 
            fc='red', ec='red', linewidth=2, zorder=5)

# Arcs
axins.add_patch(patches.Arc((eval_x, eval_y), 0.25, 0.25, angle=0, theta1=0, theta2=phi_deg, color='#1f77b4', linewidth=2))
axins.add_patch(patches.Arc((eval_x, eval_y), 0.15, 0.15, angle=0, theta1=theta_deg, theta2=0, color='red', linewidth=2))

# --- TARGETED PLACEMENT ---
# s: Directly above the dot, in the gap between black curve and blue arrow
axins.text(eval_x + 0.02, eval_y + 0.045, r'$s$', fontsize=18, fontweight='bold', ha='center', zorder=10)

# phi(s): Tucked inside the wedge between grey horizontal and blue path
axins.text(eval_x + 0.14, eval_y + 0.0027, r'$\phi(s)$', color='#1f77b4', fontsize=15, ha='left', va='bottom', zorder=10)

# theta(s): Positioned below
axins.text(eval_x + 0.08, eval_y - 0.06, r'$\theta(s)$', color='red', fontsize=15, zorder=10)

axins.plot(eval_x, eval_y, 'ko', markersize=7, zorder=11) 
mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5", linestyle='--')


plt.show()
