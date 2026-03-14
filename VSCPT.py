import numpy as np
from lib import *

# parameters, and It's needless to say hbar = 1 kL = 1
delta = 0.0
Omega0 = 1.0
Gamma = 1.5
# time parameters
N = 100000
dt = 0.0002
t = np.linspace(0.0, N * dt, N + 1)
# initial conditions
M = 800.0
p0 = -1.5
v0 = p0 / M
sigma_p0 = 2.0

n_max = int(M*v0) + 50 # larger than M*v0
p_grid = np.linspace(-n_max, n_max, 2*n_max+1)
psi0 = np.zeros((3, 2*n_max+1), dtype=complex)
psi0[0] = np.exp(-0.5 * ((p_grid - p0) / sigma_p0) ** 2)

JUMP_CHANNELS = (
    (2, 0, 0.5),
    (2, 1, 0.5),
)

class H_eff:
    def __init__(self, p_grid: np.ndarray, M: float, delta: float, Gamma: float, Omega0: float):
        p = np.asarray(p_grid, dtype=float)
        if p.ndim != 1:
            raise ValueError("p_grid 必须是一维数组")

        self.p = p
        self.M = float(M)
        self.delta = float(delta)
        self.Gamma = float(Gamma)
        self.Omega0 = float(Omega0)

        self.T = (p * p) / (2.0 * self.M)
        self.e_diag = self.T + ((-self.delta) - 0.5j * self.Gamma)
        self.coupling = self.Omega0 / np.sqrt(2.0)

    def apply(self, psi: np.ndarray) -> np.ndarray:
        psi_arr = np.asarray(psi, dtype=np.complex128)
        expected_shape = (3, self.p.shape[0])
        if psi_arr.shape != expected_shape:
            raise ValueError(f"psi shape 应为 {expected_shape}，实际得到 {psi_arr.shape}")

        gm = psi_arr[0]
        gp = psi_arr[1]
        e0 = psi_arr[2]

        coupling = self.coupling

        out = np.empty_like(psi_arr)

        out[0] = self.T * gm
        out[0, :-1] += -coupling * e0[1:]

        out[1] = self.T * gp
        out[1, 1:] += -coupling * e0[:-1]

        out[2] = self.e_diag * e0
        out[2, 1:] += -coupling * gm[:-1]
        out[2, :-1] += -coupling * gp[1:]

        return out

def main():
    heff = H_eff(p_grid, M, delta, Gamma, Omega0)
    ket = Quantumket(
        psi0,
        ["g-", "g+", "e0"],
        tuple([p_grid]),
        "p",
    ).normalize()

    v_history = np.zeros(N + 1, dtype=float)
    x_history = np.zeros(N + 1, dtype=float)
    rhoe_total_history = np.zeros(N + 1, dtype=float)

    rng = np.random.default_rng()

    v_history[0] = ket.expect() / M
    x_history[0] = 0.0
    rhoe_total_history[0] = ket.population("e0")

    for step in range(N):
        ket = RKstepper(ket, heff, dt)
        ket, jumped = mcwf_step(ket, rng, JUMP_CHANNELS)

        v_history[step + 1] = ket.expect() / M
        x_history[step + 1] = x_history[step] + v_history[step] * dt
        rhoe_total_history[step + 1] = ket.population("e0")

        if step % 1000 == 0:
            print(f"step: {step} v: {v_history[step + 1]:.6f}, total excited: {rhoe_total_history[step + 1]:.4f}")

    plot_dual_axis(t, v_history, x_history, "Velocity", "Position", save_path="vscpt_velocity_position.png")
    plot_populations(t, rhoe_total_history, labels=["e0"], title="Excited State Population", save_path="vscpt_excited_population.png")

    # Final momentum-space populations of two ground states.
    g_minus_pop = np.abs(ket.data[0]) ** 2
    g_plus_pop = np.abs(ket.data[1]) ** 2

    p_mask = (p_grid >= -10.0) & (p_grid <= 10.0)
    p_plot = p_grid[p_mask]
    g_minus_plot = g_minus_pop[p_mask]
    g_plus_plot = g_plus_pop[p_mask]

    fig_mom, ax_mom = plt.subplots(figsize=(8, 4.5))
    ax_mom.scatter(p_plot, g_minus_plot, s=12, color="C0", label=r"$|g_-(p)|^2$")
    ax_mom.scatter(p_plot, g_plus_plot, s=12, color="C1", label=r"$|g_+(p)|^2$")
    ax_mom.set_xlabel("p")
    ax_mom.set_ylabel("population density")
    ax_mom.set_title("Final Momentum-Space Ground-State Populations")
    ax_mom.set_xlim(-8, 8)
    ax_mom.legend()
    fig_mom.tight_layout()
    fig_mom.savefig("vscpt_ground_momentum_populations.png", dpi=220, bbox_inches="tight")
    plt.show()
    plt.close(fig_mom)
    print("figure saved: vscpt_ground_momentum_populations.png")

if __name__ == "__main__":
    main()