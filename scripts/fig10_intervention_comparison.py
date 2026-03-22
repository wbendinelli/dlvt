"""
Figure 10: DLVT Recovery Policy Comparison
Key insight: β reduction provides only transient relief (C compensates);
R increase provides permanent V* shift. Combined is fastest.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

# Parameters
R_def, Vmax, delta, gamma = 3.0, 10.0, 0.02, 2.0
O0, beta_def, eta = 1.0, 0.25, 1.0
alpha, phi, mu, eps = 0.1, 0.15, 0.2, 0.1
V_strat = 5.0

def dlvt(t, y, R_, beta_):
    V, C = max(y[0], 0), max(y[1], 0)
    O = O0 + beta_ * C**eta
    dV = R_*(1 - V/Vmax) - delta * O**gamma * V/(V + eps)
    dC = alpha * C * V/(1 + phi*O) - mu * C
    return [dV, dC]

def solve_two_phase(R1, b1, R2, b2, t_int=20, t_end=120):
    sol1 = solve_ivp(dlvt, (0, t_int), [3.0, 15.0], args=(R1, b1),
                     method='RK45', dense_output=True, max_step=0.1, rtol=1e-8, atol=1e-10)
    state = sol1.y[:, -1]
    sol2 = solve_ivp(dlvt, (t_int, t_end), state, args=(R2, b2),
                     method='RK45', dense_output=True, max_step=0.1, rtol=1e-8, atol=1e-10)
    t = np.concatenate([sol1.t, sol2.t])
    V = np.concatenate([sol1.y[0], sol2.y[0]])
    C = np.concatenate([sol1.y[1], sol2.y[1]])
    return t, V, C

# Scenarios
t1, V1, C1 = solve_two_phase(R_def, beta_def, R_def, beta_def)        # baseline
t2, V2, C2 = solve_two_phase(R_def, beta_def, R_def, 0.10)            # β→0.10 (-60%)
t3, V3, C3 = solve_two_phase(R_def, beta_def, 4.2, beta_def)          # R+40%
t4, V4, C4 = solve_two_phase(R_def, beta_def, 4.2, 0.10)             # both

# Print results
for name, V, C in [('Baseline', V1, C1), ('β→0.10', V2, C2), 
                    ('R+40%', V3, C3), ('Both', V4, C4)]:
    print(f'{name:12s}: V(120)={V[-1]:.3f}  C(120)={C[-1]:.3f}  escape={V[-1]>5.0}')

# === FIGURE ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

# Panel (a): Vitality trajectories
ax1.plot(t1, V1, 'k--', lw=2, alpha=0.6, label='No intervention')
ax1.plot(t2, V2, color='#2196F3', lw=2, alpha=0.85, label=r'Reduce $\beta$ by 60%')
ax1.plot(t3, V3, color='#4CAF50', lw=2, alpha=0.85, label=r'Increase $R$ by 40%')
ax1.plot(t4, V4, color='#FF5722', lw=2.5, alpha=0.85, label='Both simultaneously')
ax1.axhline(V_strat, color='gray', ls=':', lw=1.2, alpha=0.5)
ax1.axvline(20, color='gray', ls=':', lw=1.2, alpha=0.5)
ax1.text(21, 2.2, 'Intervention', fontsize=9, color='gray', style='italic')
ax1.text(85, 5.15, r'$V_{\mathrm{strategic}}$', fontsize=10, color='gray')
ax1.set_xlabel('Time (arbitrary units)', fontsize=11)
ax1.set_ylabel(r'Vitality $V(t)$', fontsize=11)
ax1.set_xlim(0, 120)
ax1.set_ylim(2, 7)
ax1.legend(loc='upper left', fontsize=9, framealpha=0.92)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_title('(a) Vitality trajectories', fontsize=11, pad=10)

# Panel (b): Capital trajectories
ax2.plot(t1, C1, 'k--', lw=2, alpha=0.6, label='No intervention')
ax2.plot(t2, C2, color='#2196F3', lw=2, alpha=0.85, label=r'Reduce $\beta$ by 60%')
ax2.plot(t3, C3, color='#4CAF50', lw=2, alpha=0.85, label=r'Increase $R$ by 40%')
ax2.plot(t4, C4, color='#FF5722', lw=2.5, alpha=0.85, label='Both simultaneously')
ax2.axvline(20, color='gray', ls=':', lw=1.2, alpha=0.5)
ax2.text(21, 5, 'Intervention', fontsize=9, color='gray', style='italic')
ax2.set_xlabel('Time (arbitrary units)', fontsize=11)
ax2.set_ylabel(r'Career Capital $C(t)$', fontsize=11)
ax2.set_xlim(0, 120)
ax2.legend(loc='upper left', fontsize=9, framealpha=0.92)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_title('(b) Career capital trajectories', fontsize=11, pad=10)

plt.tight_layout()
out = '/tmp/dlvt-work/manuscript/figures/fig10_intervention_comparison.pdf'
plt.savefig(out, format='pdf', dpi=300, bbox_inches='tight')
plt.close()
print(f'\nFigure saved: {out}')

# Summary
with open('/tmp/dlvt-work/manuscript/figures/fig10_summary.txt', 'w') as f:
    f.write('Fig 10: Intervention Comparison\n\n')
    f.write('KEY STRUCTURAL INSIGHT:\n')
    f.write('V* at equilibrium is beta-INDEPENDENT. C adjusts to maintain the same O*.\n')
    f.write('Only R shifts V*. Beta reduction raises C*_max but not V*.\n\n')
    f.write('Scenarios (intervention at t=20, from zombie V0=3.0, C0=15.0):\n')
    f.write(f'  Baseline:        V*={V1[-1]:.3f}, C*={C1[-1]:.1f}\n')
    f.write(f'  beta->0.10:      V*={V2[-1]:.3f}, C*={C2[-1]:.1f} (transient spike only)\n')
    f.write(f'  R+40%:           V*={V3[-1]:.3f}, C*={C3[-1]:.1f} (permanent escape)\n')
    f.write(f'  Both:            V*={V4[-1]:.3f}, C*={C4[-1]:.1f} (fastest + highest C)\n')
print('Summary saved')
