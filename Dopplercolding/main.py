from lib import *

# parameters, and It's needless to say hbar = 1 kL = 1
delta = -2.0
Omega0 = 0.7
Gamma = 1.0
# time parameters
N = 20000
dt = 0.01
t = np.arange(0, N*dt, dt)
# initial conditions
M = 800.0
v0 = 1.5


n_max = int(M*v0) + 100 # larger than M*v0
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
    H_eff = H_eff(p_grid, M, delta, Gamma, Omega0)
    ket = Quantumket(psi0, ["g", "e"], tuple([p_grid]), "p")
    v_history = np.zeros(N, dtype=float)
    rhoe_history = np.zeros(N, dtype=float)

    for step in range(N):
        v_history[step] = ket.expect() / M
        rhoe_history[step] = ket.population('e')

        ket = RKstepper(ket, H_eff, dt)
        dp = 1 - ket.norm2()

        if np.random.rand() > dp:
            ket = ket.normalize()

        else:
            psi = ket.data
            merged = np.sum(psi, axis=0)
            psi[:] = 0.0
            if np.random.rand() > 0.5:
                psi[0][1:] = merged[:-1]
            else:
                psi[0][:-1] = merged[1:]
            ket = Quantumket._fast_from(ket, psi).normalize()

        