import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Plot settings
# --------------------------------------------------------------------------- #
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{mathpazo}'
plt.rcParams['font.size'] = 16  # Global font size

# --------------------------------------------------------------------------- #
# Functions
# --------------------------------------------------------------------------- #
def beta_probe_post(omega, beta_s, beta_m, gamma):
    gamma = np.asarray(gamma, dtype=float)
    prod_term = np.prod(1.0 + np.exp(-beta_m * gamma))
    exp_s = np.exp(-beta_s * omega)
    exp_m_sum = np.exp(-beta_m * gamma.sum())
    num = 1.0 + (prod_term**(-1)) * (exp_s - exp_m_sum)
    den = (1.0 + exp_s) - (prod_term**(-1)) * (exp_s - exp_m_sum) -1
    if den <= 0 or num <= 0:
        return np.nan
    return omega**(-1) * np.log(num / den)

# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
omega = 1.0
a = 0.5
b = 2
E1 = 1
E2 = 2
E3 = 3
beta_s_grid = np.linspace(1, 3, 300)

# Generate all 8 gamma combinations
gammas = [
    np.array([x1*E1, x2*E2, x3*E3])
    for x1, x2, x3 in [(a,a,a), (a,a,b), (a,b,a), (a,b,b), (b,a,a), (b,a,b), (b,b,a), (b,b,b)]
]

# Generate labels dynamically
gamma_labels = [
    fr'$s={i:03b}$, $\Gamma = [{g[0]:.1f}, {g[1]:.1f}, {g[2]:.1f}]$'
    for i, g in enumerate(gammas)
]

# Generate colors
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']

# Machine inverse temperature
beta_m_values = [2]
linestyles = ['-']

# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(8, 5))

for idx, beta_m in enumerate(beta_m_values):
    for gamma, label, color in zip(gammas, gamma_labels, colors):
        y = np.array([beta_probe_post(omega, beta_s, beta_m, gamma) for beta_s in beta_s_grid])

        if np.all(np.isnan(y)):
            print(f"Warning: All NaNs for {label} (Gamma = {gamma}) — Skipping plot.")
            continue  # Skip plotting bad curve

        mask = np.isfinite(y)
        ax.plot(beta_s_grid[mask], y[mask],
                linestyle=linestyles[idx % len(linestyles)],
                color=color,
                label=fr'{label}')

ax.set_xlabel(r'Probe Initial inv. temp. $\beta_S$')
ax.set_ylabel(r'Probe Post Query inv. temp. $\beta^\prime_S$')
ax.grid(True)
ax.legend(fontsize=14, loc='best', ncol=2)  # 2 columns for nicer legend
textstr = (
    fr'$E_1={a}$, $E_2={b}$, $\beta_m = 2$' + "\n" +
    fr'$\gamma_1={E1}$, $\gamma_2={E2}$, $\gamma_3={E3}$'
)

# Position relative to the axis
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
ax.text(0.665, 0.55, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='center', bbox=props)
fig.tight_layout()

save_path = "BV_temps_2.png"
plt.savefig(save_path, dpi=600)
plt.show()
