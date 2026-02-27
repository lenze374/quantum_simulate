from lib import *
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
# parameters, and It's needless to say hbar = 1 kL = 1
delta = -2.0
Omega0 = 0.7
Gamma = 1.0
# time parameters
N = 15000000
dt = 0.0004
t = np.linspace(0.0, N * dt, N + 1)
# initial conditions
M = 800.0
v0 = 0.07


n_max = int(M*v0) + 1000 # larger than M*v0
psi0 = np.zeros((2, 2*n_max+1), dtype=complex)
psi0[0, int(n_max + M*v0)] = 1.0 
p_grid = np.linspace(-n_max, n_max, 2*n_max+1)

class H_eff:
    def __init__(self, p_grid: np.ndarray, M: float, delta: float, Gamma: float, Omega0: float):
        p = np.asarray(p_grid, dtype=float)
        self.p = p
        self.M = float(M)
        self.delta = float(delta)
        self.Gamma = float(Gamma)
        self.Omega0 = float(Omega0)

        self.T = (p * p) / (2.0 * self.M)
        self.coupling = 0.5 * self.Omega0
        self.e_shift = (-self.delta) - 0.5j * self.Gamma
        self._build_cn(float(dt))

    def _build_cn(self, dt_val: float):
        Np = int(self.p.shape[0])
        T = np.asarray(self.T, dtype=np.complex128)
        coupling = complex(self.coupling)

        Hgg = sp.diags(T, 0, shape=(Np, Np), format="csc")
        Hee = sp.diags((T + complex(self.e_shift)), 0, shape=(Np, Np), format="csc")

        if Np >= 2:
            off = np.ones(Np - 1, dtype=np.complex128)
            shift_plus = sp.diags(off, 1, shape=(Np, Np), format="csc")
            shift_minus = sp.diags(off, -1, shape=(Np, Np), format="csc")
            S = shift_plus + shift_minus
        else:
            S = sp.csc_matrix((Np, Np), dtype=np.complex128)

        Hge = coupling * S
        Heg = coupling * S
        H = sp.bmat([[Hgg, Hge], [Heg, Hee]], format="csc", dtype=np.complex128)
        self.H_sparse = H

        dim = int(H.shape[0])
        I = sp.eye(dim, format="csc", dtype=np.complex128)
        A = (I + 1j * dt_val / 2.0 * H).tocsc()
        B = (I - 1j * dt_val / 2.0 * H).tocsc()

        self.cn_dt = float(dt_val)
        self.B = B
        self.solve_A = spla.factorized(A)

    def apply(self, psi: np.ndarray) -> np.ndarray:  # noqa: ARG002
        psi_arr = np.asarray(psi, dtype=np.complex128)
        g = psi_arr[0]
        e = psi_arr[1]

        e_neighbors = np.zeros_like(e)
        e_neighbors[1:] += e[:-1]
        e_neighbors[:-1] += e[1:]

        g_neighbors = np.zeros_like(g)
        g_neighbors[1:] += g[:-1]
        g_neighbors[:-1] += g[1:]

        Hg = self.T * g + self.coupling * e_neighbors
        He = (self.T + self.e_shift) * e + self.coupling * g_neighbors

        out = np.empty_like(psi_arr)
        out[0] = Hg
        out[1] = He
        return out

def main():
    heff = H_eff(p_grid, M, delta, Gamma, Omega0)
    ket = Quantumket(psi0, ["g", "e"], tuple([p_grid]), "p")
    v_history = np.zeros(N + 1, dtype=float)
    rhoe_history = np.zeros(N + 1, dtype=float)
    x_history = np.zeros(N + 1, dtype=float)

    jumped_history = np.zeros(N + 1, dtype=np.uint8)
    recoil_history = np.zeros(N + 1, dtype=np.int8)
    n_jump_left = 0
    n_jump_right = 0

    rng = np.random.default_rng()

    # Record initial observables at t[0]
    v_history[0] = ket.expect() / M
    rhoe_history[0] = ket.population('e')
    x_history[0] = 0.0

    for step in range(N):

        ket = RKstepper(ket, heff, dt)
        # ket = CNstepper(ket, heff, dt)

        # MCWF dissipation: jump with probability dp = 1 - ||psi||^2
        dp = 1.0 - ket.norm2()
        if dp < 0.0:
            print(f"Warning: negative jump probability dp={dp:.2e} at step {step}")
            dp = 0.0

        if rng.random() >= dp:
            ket = ket.normalize()
            jumped_history[step + 1] = 0
            recoil_history[step + 1] = 0
        else:
            psi = ket.data
            e = psi[1].copy()
            psi[:] = 0.0
            if rng.random() < 0.5:
                # recoil to + direction
                psi[0][1:] = e[:-1]
                jumped_history[step + 1] = 1
                recoil_history[step + 1] = 1
                n_jump_right += 1
            else:
                # recoil to - direction
                psi[0][:-1] = e[1:]
                jumped_history[step + 1] = 1
                recoil_history[step + 1] = -1
                n_jump_left += 1
            ket = Quantumket._fast_from(ket, psi).normalize()

        # Record observables at t[step+1]
        v_history[step + 1] = ket.expect() / M
        rhoe_history[step + 1] = ket.population('e')
        x_history[step + 1] = x_history[step] + v_history[step] * dt

        if step % 10000 == 0:
            print(f"step: {step} v: {v_history[step + 1]:.4f}, population e: {rhoe_history[step + 1]:.4f}")

    data = {
        "t": t,
        "v": v_history,
        "x": x_history,
        "rhoe": rhoe_history,
        "jumped": jumped_history,
        "recoil": recoil_history,
    }
    np.savez("simulation_data.npz", **data)
    n_jump_total = int(n_jump_left + n_jump_right)
    print("\nSpontaneous emission (MCWF) statistics:")
    print(f"  total jumps: {n_jump_total}")
    print(f"  recoil -1 (left): {n_jump_left}")
    print(f"  recoil +1 (right): {n_jump_right}")

    plot_dual_axis(t, v_history, x_history, "Velocity", "Position", save_path="velocity_position01.png")
    plot_populations(t, rhoe_history, labels=["e"], title="Excited State Population", save_path="population01.png")

if __name__ == "__main__":
    main()