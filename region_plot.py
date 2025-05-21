import numpy as np
import matplotlib.pyplot as plt

# Plot settings
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{mathpazo}'
plt.rcParams['font.size'] = 16  # Global font size

# Fixed parameters
omega = 1

# Define the function as per Mathematica code
def func(beta_m, n, epsilon1, epsilon2, t):
    term1 = np.exp(-beta_m * (n / 2) * (epsilon1 + epsilon2))
    term2 = np.exp(-beta_m * n * epsilon1)
    return -np.log(np.abs(term1 - term2 - t)) + np.log(t)

# Generate meshgrid for plotting
n_vals = np.linspace(2, 1200, 400)
beta_s_vals = np.linspace(0, 2, 300)
N, BETA_S = np.meshgrid(n_vals, beta_s_vals)

# Calculate function values
epsilon1 = 6 / (N**2)
epsilon2 = 1 / (N**2)
t = 0.1

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5))
cmap = plt.get_cmap('Pastel2', 2)

# Plot for beta_M = 2
beta_M1 = 2.5
lower_bound1 = (N * (epsilon1) * beta_M1) + np.log(1 - (1 - np.exp(-beta_M1*(epsilon1)))**N)
region1 = (BETA_S > lower_bound1) & (BETA_S < func(beta_M1, N, epsilon1, epsilon2, t))
c1 = ax1.contourf(N, BETA_S, region1, levels=[0, 0.5, 1], cmap=cmap, alpha=0.7)
ax1.text(0.675, 0.2,
    fr'$E_1=6/N^2$, $E_2=1/N^2$' +"\n" + 
    fr'$\beta_M = 2.5, t=0.1$',
    transform=ax1.transAxes,
    fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
#ax1.set_title(r"Region Plot for $\beta_M = 3$", fontsize=18)
ax1.set_xlim(2, 1100)
ax1.set_ylim(0, 2)

# Plot for beta_M = 5
beta_M2 = 5
lower_bound2 = (N * (10 / (N**2)) * beta_M2) + np.log(1 - (1 - np.exp(-beta_M1*(10 / (N**2))))**N)
region2 = (BETA_S > lower_bound2) & (BETA_S < func(beta_M2, N, 10 / (N**2), epsilon2, t))
c2 = ax2.contourf(N, BETA_S, region2, levels=[0, 0.5, 1], cmap=cmap, alpha=0.7)
#ax2.set_title(r"Region Plot for $\beta_M = 8$", fontsize=18)
ax2.text(0.675, 0.2,
    fr'$E_1=10/N^2$, $E_2=1/N^2$' +"\n" + 
    fr'$\beta_M = 5, t=0.1$',
    transform=ax2.transAxes,
    fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax2.set_xlim(2, 1100)
ax2.set_ylim(0, 2)

# Shared axis labels
fig.supxlabel(r"Size of Domain of $f(x)$, $N = 2^n$", fontsize=16, y=0.08)
fig.supylabel(r"Probe Parameters, $\omega/T_S$", fontsize=16, y=0.5, x=0.05)

# Add legend to the first plot
legend_labels = ['Inequality Not Satisfied', 'Inequality Satisfied']
for i, label in enumerate(legend_labels):
    ax1.plot([], [], color=cmap(i), label=label)
ax1.legend(loc='upper right')

plt.tight_layout()
save_path = "region_plot_2.png"
plt.savefig(save_path, dpi=600)
plt.show()
