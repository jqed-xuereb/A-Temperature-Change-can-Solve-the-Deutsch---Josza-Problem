import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{mathpazo}'
plt.rcParams['font.size'] = 16  # for all elements (axes labels, ticks, etc.
# --------------------------------------------------------------------------- #
# Functions
# --------------------------------------------------------------------------- #
def beta_probe_post(omega, beta_s, beta_m, gamma):
    gamma = np.asarray(gamma, dtype=float)
    n = len(gamma)
    prod_term = np.prod(1.0 + np.exp(-beta_m * gamma))
    exp_s = np.exp(-beta_s * omega)
    exp_m_sum = np.exp(-beta_m * gamma.sum())
    num = 1.0 + (prod_term**(-1)) * (exp_s - exp_m_sum)
    den = (1.0 + exp_s) - (prod_term**(-1)) * (exp_s - exp_m_sum) -1
    if den <= 0 or num <= 0:
        return np.nan
    return omega * np.log(num / den)

# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
omega = 1.0
n = 4
beta_s_grid = np.linspace(1, 3, 300)
gammas = [
    np.array([1, 1, 1, 1]),
    np.array([2, 2, 2, 2]),
    np.array([1, 1, 2, 2])
]
gamma_labels = [
    r'$f(x)$ constant, $\Gamma = [1,1,1,1]$',
    r'$f(x)$ constant, $\Gamma = [2,2,2,2]$',
    r'$f(x)$ balanced, $\Gamma = [1,1,2,2]$'
]
colors = ['red', 'blue', 'green']

# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
# Let's adjust the plot to have two groups (two values of β_M)
# and keep the machine colder than the probe: β_M > β_S

# New setup
beta_m_values = [1, 2.2]  # machine inverse temperatures (colder machine)
colors = ['red', 'blue', 'green']
linestyles = ['-', '--']    # solid and dashed to distinguish beta_M

# Redo the plot
fig, ax = plt.subplots(figsize=(8, 5))

for idx, beta_m in enumerate(beta_m_values):
    for gamma, label, color in zip(gammas, gamma_labels, colors):
        y = np.array([beta_probe_post(omega, beta_s, beta_m, gamma) for beta_s in beta_s_grid])
        mask = np.isfinite(y)
        ax.plot(beta_s_grid[mask], y[mask],
                linestyle=linestyles[idx], color=color,
                label=fr'{label}, $\beta_M={beta_m}$')

ax.set_xlabel(r'Probe Initial inv. temp. $\beta_S$')
ax.set_ylabel(r'Probe Post Query inv. temp. $\beta^\prime_S$')
#ax.set_title(r'Post Query inv. Temp. $\beta^\prime_S$ against initial Probe inv. Temp. $\beta_S$ for different $\Gamma$ and colder $\beta_M$')
ax.grid(True)
ax.legend(fontsize=16, loc='best')
fig.tight_layout()
save_path="dj_temps.png"
plt.savefig(save_path, dpi=600)


plt.show()

