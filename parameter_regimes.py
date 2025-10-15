import numpy as np
import matplotlib.pyplot as plt

# If you have a full LaTeX install, you can uncomment these lines.
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{mathpazo}'
plt.rcParams['font.size'] = 16  # for all elements

# --------------------------------------------------------------------------- #
# Functions (Mathematica -> Python)
# PartConst[β, E1, n] = (1 + e^{-β E1})^n
# PartBal[β, E1, E2, n] = (1 + e^{-β E1})^(n/2) * (1 + e^{-β E2})^(n/2)
# Func[β, E1, E2, n] = 1/PartConst - 1/PartBal
# --------------------------------------------------------------------------- #

def _safe_log_part_const(beta, E1):
    """Return log((1 + exp(-beta*E1))) once, to reuse for all n."""
    return np.log1p(np.exp(-beta * E1))

def _safe_log_part_bal(beta, E1, E2):
    """Return 0.5 * [log(1 + exp(-βE1)) + log(1 + exp(-βE2))]."""
    return 0.5 * (np.log1p(np.exp(-beta * E1)) + np.log1p(np.exp(-beta * E2)))

def PartConst(beta, E1, n):
    # (1 + e^{-β E1})^n = exp(n * log(1 + e^{-β E1}))
    logpc = _safe_log_part_const(beta, E1)
    return np.exp(n * logpc)

def PartBal(beta, E1, E2, n):
    # (1 + e^{-β E1})^{n/2} (1 + e^{-β E2})^{n/2}
    logpb = _safe_log_part_bal(beta, E1, E2)
    return np.exp(n * logpb)

def Func(beta, E1, E2, n):
    # Compute 1/PartConst - 1/PartBal in a numerically stable way:
    # 1/PartConst = exp(-n * log(1 + e^{-βE1}))
    # 1/PartBal   = exp(-n * 0.5*(log(1 + e^{-βE1}) + log(1 + e^{-βE2})))
    logpc = _safe_log_part_const(beta, E1)
    logpb = _safe_log_part_bal(beta, E1, E2)
    term1 = np.exp(-n * logpc)
    term2 = np.exp(-n * logpb)
    return term1 - term2

# --------------------------------------------------------------------------- #
# Parameters & grid
# --------------------------------------------------------------------------- #
beta = 1.0
E2 = 1.0
E1_values = [10,8, 6, 4, 2]
labels = [
    r'$E_2 = 1, E_1 = 10$',
    r'$E_2 = 1, E_1 = 8$',
    r'$E_2 = 1, E_1 = 6$',
    r'$E_2 = 1, E_1 = 4$',
    r'$E_2 = 1, E_1 = 2$',
]

# n is real in the Mathematica Plot; we’ll sample densely across [2, 2^15]
n_min, n_max = 2.0, 2.0**14
n_grid = np.linspace(n_min, n_max, 2000)

# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(8, 4.5))

for E1, label in zip(E1_values, labels):
    y = Func(beta, E1, E2, n_grid)
    # Mask any non-finite (shouldn’t happen with log-space, but just in case)
    mask = np.isfinite(y)
    ax.plot(n_grid[mask], y[mask], label=label)

ax.set_xscale('log', base=2)
ax.set_xlabel(r'$N = 2^n$, no. of oracle qubits (log scale)')
ax.set_ylim(bottom=-0.02)
ax.set_ylabel(r'$\frac{1}{\mathcal{Z}^\mathrm{Const}_f} - \frac{1}{\mathcal{Z}^\mathrm{Bal}_f}$')
ax.grid(True)
ax.legend(fontsize=14, loc='lower right', framealpha=0.55)
fig.tight_layout()

# Save if you like
plt.savefig("parameter_plot.png", dpi=400)
plt.show()


