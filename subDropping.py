import numpy as np
from lib import *

# parameters, and It's needless to say hbar = 1 kL = 1
delta = -10.0
Omega0 = 1.0
Gamma = 1.0 
# time parameters
N = 5000000
dt = 0.0004
t = np.linspace(0.0, N * dt, N + 1)
# initial conditions
M = 800.0
v0 = 0.02

n_max = int(M*v0) + 100 # larger than M*v0
psi0 = np.zeros((6, 2*n_max+1), dtype=complex)
psi0[0, int(n_max + M*v0)] = 1.0 
p_grid = np.linspace(-n_max, n_max, 2*n_max+1)

JUMP_CHANNELS = (
    (2, 0, 1.0),
    (3, 0, 2.0 / 3.0),
    (3, 1, 1.0 / 3.0),
    (4, 0, 1.0 / 3.0),
    (4, 1, 2.0 / 3.0),
    (5, 1, 1.0),
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
        self.c1 = 0.5 * self.Omega0
        self.c3 = self.c1 / np.sqrt(3.0)

    def apply(self, psi: np.ndarray) -> np.ndarray:
        psi_arr = np.asarray(psi, dtype=np.complex128)
        expected_shape = (6, self.p.shape[0])
        if psi_arr.shape != expected_shape:
            raise ValueError(f"psi shape 应为 {expected_shape}，实际得到 {psi_arr.shape}")

        gp = psi_arr[0]
        gm = psi_arr[1]
        e32 = psi_arr[2]
        e12 = psi_arr[3]
        em12 = psi_arr[4]
        em32 = psi_arr[5]

        c1 = self.c1
        c3 = self.c3

        out = np.empty_like(psi_arr)

        out[0] = self.T * gp
        out[0, :-1] += -c1 * e32[1:] - c3 * em12[1:]
        out[0, 1:] += 1j * c1 * e32[:-1] - 1j * c3 * em12[:-1]

        out[1] = self.T * gm
        out[1, :-1] += -c3 * e12[1:] - c1 * em32[1:]
        out[1, 1:] += 1j * c3 * e12[:-1] - 1j * c1 * em32[:-1]

        out[2] = self.e_diag * e32
        out[2, 1:] += -c1 * gp[:-1]
        out[2, :-1] += -1j * c1 * gp[1:]

        out[3] = self.e_diag * e12
        out[3, 1:] += -c3 * gm[:-1]
        out[3, :-1] += -1j * c3 * gm[1:]

        out[4] = self.e_diag * em12
        out[4, 1:] += -c3 * gp[:-1]
        out[4, :-1] += 1j * c3 * gp[1:]

        out[5] = self.e_diag * em32
        out[5, 1:] += -c1 * gm[:-1]
        out[5, :-1] += 1j * c1 * gm[1:]

        return out

def main():
    heff = H_eff(p_grid, M, delta, Gamma, Omega0)
    ket = Quantumket(
        psi0,
        ["g+", "g-", "e+3/2", "e+1/2", "e-1/2", "e-3/2"],
        tuple([p_grid]),
        "p",
    )

    v_history = np.zeros(N + 1, dtype=float)
    x_history = np.zeros(N + 1, dtype=float)
    rhoe_total_history = np.zeros(N + 1, dtype=float)
    rng = np.random.default_rng()

    v_history[0] = ket.expect() / M
    x_history[0] = 0.0
    rhoe_total_history[0] = (
        ket.population("e+3/2") + ket.population("e+1/2") + ket.population("e-1/2") + ket.population("e-3/2")
    )

    for step in range(N):
        ket = RKstepper(ket, heff, dt)
        ket, jumped = mcwf_step(ket, rng, JUMP_CHANNELS)

        v_history[step + 1] = ket.expect() / M
        x_history[step + 1] = x_history[step] + v_history[step] * dt
        rhoe_total_history[step + 1] = (
            ket.population("e+3/2") + ket.population("e+1/2") + ket.population("e-1/2") + ket.population("e-3/2")
        )

        if step % 1000 == 0:
            print(f"step: {step} v: {v_history[step + 1]:.4f}, total excited: {rhoe_total_history[step + 1]:.4f}")

    plot_dual_axis(t, v_history, x_history, "Velocity", "Position", save_path="subdropping_velocity_position.png")
    plot_populations(t, rhoe_total_history, labels=["e"], title="Excited State Population", save_path="subdropping_excited_population.png")

if __name__ == "__main__":
    main()