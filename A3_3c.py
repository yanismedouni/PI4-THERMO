import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
N = int(1e6)

def sample_turbine(N):
    u = np.random.uniform(0, 1, N)
    v = 9 * np.sqrt(-np.log(1 - u))
    P = np.zeros(N)
    mask_partial = (v >= 2)  & (v <= 14)
    mask_rated   = (v > 14) & (v <= 25)
    P[mask_partial] = 5.787e-4 * (v[mask_partial] - 2)**3
    P[mask_rated]   = 1.0
    return P

# --- Simulate farms ---
n_values = [1, 2, 10, 50]
farms = {}
for n in n_values:
    farms[n] = sum(sample_turbine(N) for _ in range(n))

# --- Print statistics ---
for n, P_farm in farms.items():
    print(f"\nn = {n} turbines:")
    print(f"  E[P_farm]      = {np.mean(P_farm):.3f} MW")
    print(f"  Median[P_farm] = {np.median(P_farm):.3f} MW")
    print(f"  P90 = {np.percentile(P_farm, 10):.4f} MW")
    print(f"  P95 = {np.percentile(P_farm,  5):.4f} MW")
    print(f"  P99 = {np.percentile(P_farm,  1):.4f} MW")

# --- 2x2 plot ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for ax, (n, P_farm) in zip(axes, farms.items()):
    mean_P   = np.mean(P_farm)
    med_P    = np.median(P_farm)
    p90      = np.percentile(P_farm, 10)
    p95      = np.percentile(P_farm,  5)
    p99      = np.percentile(P_farm,  1)

    ax.hist(P_farm, bins=100, color='steelblue',
            edgecolor='gray', alpha=0.7)
    ax.axvline(mean_P, color='red',    lw=2, linestyle='--',
               label=f'E[P]   = {mean_P:.3f} MW')
    ax.axvline(med_P,  color='orange', lw=2, linestyle='--',
               label=f'Median = {med_P:.3f} MW')
    ax.axvline(p90,    color='purple', lw=2, linestyle=':',
               label=f'P90    = {p90:.3f} MW')
    ax.axvline(p95,    color='green',  lw=2, linestyle='-.',
               label=f'P95    = {p95:.3f} MW')
    ax.axvline(p99,    color='black',  lw=2, linestyle=':',
               label=f'P99    = {p99:.3f} MW')
    ax.set_xlabel('Total farm power (MW)')
    ax.set_ylabel('Empirical frequency')
    ax.set_title(f'n = {n} turbine{"s" if n > 1 else ""}')
    ax.legend(fontsize=7)

plt.suptitle('Simulated wind farm power PDF for increasing n', fontsize=13)
plt.tight_layout()
plt.savefig('fig_pdf_2x2.png', dpi=150)
plt.show()