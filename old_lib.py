import numpy as np 
import scipy.sparse as sp 
import scipy.sparse.linalg as spla 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.integrate as spi
from typing import Callable, Literal, overload
import numpy.typing as npt


# =========================
# Momentum-basis (1D) MCWF
# =========================

def build_momentum_grid(
    p_center: float,
    kL: float,
    n_max: int,
) -> npt.NDArray[np.floating]:
    """Build a 1D momentum grid: p = p_center + m*kL, m in [-n_max, n_max]."""
    m = np.arange(-int(n_max), int(n_max) + 1, dtype=float)
    return p_center + m * float(kL)


def _heff_apply_momentum(
    psi_ge: npt.NDArray[np.complexfloating],
    p_grid: npt.NDArray[np.floating],
    *,
    M: float,
    delta: float,
    Gamma: float,
    Omega0: float,
) -> npt.NDArray[np.complexfloating]:
    """Apply effective non-Hermitian Hamiltonian in momentum basis.

    Basis: |g, p_n>, |e, p_n> with coupling induced by two counter-propagating beams.
    Off-diagonal coupling uses: (Omega0/2) * (shift +1 + shift -1).

    H_eff includes decay: -i*Gamma/2 on excited manifold.
    """
    psi = np.asarray(psi_ge, dtype=np.complex128)
    if psi.ndim != 2 or psi.shape[0] != 2:
        raise ValueError(f"psi_ge must have shape (2, Np), got {psi.shape}")

    g = psi[0]
    e = psi[1]
    p = np.asarray(p_grid, dtype=float)

    T = (p * p) / (2.0 * float(M))
    coupling = 0.5 * float(Omega0)

    # Neighbor sums with zero boundary conditions.
    e_left = np.zeros_like(e)
    e_right = np.zeros_like(e)
    e_left[1:] = e[:-1]
    e_right[:-1] = e[1:]
    e_neighbors = e_left + e_right

    g_left = np.zeros_like(g)
    g_right = np.zeros_like(g)
    g_left[1:] = g[:-1]
    g_right[:-1] = g[1:]
    g_neighbors = g_left + g_right

    Hg = T * g + coupling * e_neighbors
    He = (T - float(delta) - 0.5j * float(Gamma)) * e + coupling * g_neighbors

    out = np.empty_like(psi)
    out[0] = Hg
    out[1] = He
    return out


def evolve_mcwf_momentum_rk4(
    psi_ge: npt.NDArray[np.complexfloating],
    dt: float,
    p_grid: npt.NDArray[np.floating],
    *,
    M: float,
    delta: float,
    Gamma: float,
    Omega0: float,
) -> npt.NDArray[np.complexfloating]:
    """Single RK4 step with H_eff for momentum-basis MCWF."""
    psi0 = np.asarray(psi_ge, dtype=np.complex128)

    def rhs(state):
        return -1j * _heff_apply_momentum(
            state, p_grid, M=M, delta=delta, Gamma=Gamma, Omega0=Omega0
        )

    k1 = rhs(psi0)
    k2 = rhs(psi0 + (dt / 2.0) * k1)
    k3 = rhs(psi0 + (dt / 2.0) * k2)
    k4 = rhs(psi0 + dt * k3)
    return psi0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def build_heff_momentum_sparse(
    p_grid: npt.NDArray[np.floating],
    *,
    M: float,
    delta: float,
    Gamma: float,
    Omega0: float,
) -> sp.spmatrix:
    """Sparse H_eff for momentum-basis evolution.

    Ordering: [g(0..Np-1), e(0..Np-1)].
    Coupling connects nearest neighbors due to exp(±ikx) terms.
    """
    p = np.asarray(p_grid, dtype=float)
    Np = int(p.shape[0])
    T = (p * p) / (2.0 * float(M))

    Hgg = sp.diags(T.astype(np.complex128), 0, shape=(Np, Np), format="csc")
    Hee_diag = (T - float(delta) - 0.5j * float(Gamma)).astype(np.complex128)
    Hee = sp.diags(Hee_diag, 0, shape=(Np, Np), format="csc")

    coupling = 0.5 * float(Omega0)
    off = np.ones(Np - 1, dtype=np.complex128)
    shift_plus = sp.diags(off, 1, shape=(Np, Np), format="csc")
    shift_minus = sp.diags(off, -1, shape=(Np, Np), format="csc")
    S = shift_plus + shift_minus
    Hge = coupling * S
    Heg = coupling * S

    H = sp.bmat(
        [[Hgg, Hge], [Heg, Hee]],
        format="csc",
        dtype=np.complex128,
    )
    return H


def make_cn_propagator(
    H: sp.spmatrix,
    dt: float,
):
    """Return a fast CN stepper psi_{n+1} = A^{-1} (B psi_n)."""
    dim = int(H.shape[0])
    I = sp.eye(dim, format="csc", dtype=np.complex128)  # type: ignore[arg-type]
    Hc = sp.csc_matrix(H, dtype=np.complex128)
    A = sp.csc_matrix(I + 1j * float(dt) / 2.0 * Hc)
    B = sp.csc_matrix(I - 1j * float(dt) / 2.0 * Hc)
    solve_A = spla.factorized(A)

    def step(psi_vec: npt.NDArray[np.complexfloating]) -> npt.NDArray[np.complexfloating]:
        rhs = B @ np.asarray(psi_vec, dtype=np.complex128)
        return np.asarray(solve_A(rhs), dtype=np.complex128)

    return step


def evolve_mcwf_momentum_cn(
    psi_ge: npt.NDArray[np.complexfloating],
    cn_step,
) -> npt.NDArray[np.complexfloating]:
    """One CN step for momentum-basis MCWF, keeping (2, Np) shape."""
    psi = np.asarray(psi_ge, dtype=np.complex128)
    if psi.ndim != 2 or psi.shape[0] != 2:
        raise ValueError(f"psi_ge must have shape (2, Np), got {psi.shape}")
    Np = int(psi.shape[1])
    vec = psi.reshape(2 * Np)
    vec2 = cn_step(vec)
    return vec2.reshape((2, Np))


def dissipation_mcwf_momentum(
    psi_ge: npt.NDArray[np.complexfloating],
    *,
    recoil_steps: int = 1,
    return_info: bool = False,
):
    """MCWF dissipation/jump for momentum-basis wavefunction.

    After a non-Hermitian step, norm loss gives jump probability.
    Jump operator models spontaneous emission with 1D recoil +/- recoil_steps.

    On jump: |psi> -> L_±|psi> / ||L_±|psi>||, with L_± mapping e->g and shifting momentum.
    We use equal probability for ± in 1D.
    """
    psi = np.asarray(psi_ge, dtype=np.complex128)
    if psi.ndim != 2 or psi.shape[0] != 2:
        raise ValueError(f"psi_ge must have shape (2, Np), got {psi.shape}")

    norm_sq = float(np.vdot(psi.ravel(), psi.ravel()).real)
    if norm_sq <= 0.0 or norm_sq > 1.0:
        raise ValueError(f"MCWF invalid norm: ||psi||^2 = {norm_sq}")

    p_jump = 1.0 - norm_sq
    rng = np.random.default_rng()

    if rng.random() >= p_jump:
        normalized = psi / np.sqrt(norm_sq)
        return (normalized, False, 0) if return_info else normalized

    # Jump: collapse from excited manifold to ground with recoil.
    g = psi[0]
    e = psi[1]
    Np = int(g.shape[0])
    step = int(recoil_steps)
    if step <= 0:
        step = 1

    recoil_sign = 1 if rng.random() < 0.5 else -1
    new_g = np.zeros_like(g)

    if recoil_sign > 0:
        # atom momentum increases by +step*k
        if step < Np:
            new_g[step:] = e[:-step]
    else:
        # atom momentum decreases by -step*k
        if step < Np:
            new_g[:-step] = e[step:]

    new_e = np.zeros_like(e)
    jumped = np.stack([new_g, new_e], axis=0)
    jumped_norm = float(np.vdot(jumped.ravel(), jumped.ravel()).real)
    if jumped_norm <= 0.0:
        # If recoil pushes all amplitude out of grid, fall back to normalized no-jump.
        normalized = psi / np.sqrt(norm_sq)
        return (normalized, False, 0) if return_info else normalized

    jumped = jumped / np.sqrt(jumped_norm)
    return (jumped, True, recoil_sign) if return_info else jumped


def make_initial_momentum_state(
    p_grid: npt.NDArray[np.floating],
    *,
    p0: float,
    sigma_p: float | None = None,
    internal: Literal["g", "e"] = "g",
) -> npt.NDArray[np.complexfloating]:
    """Create an initial |internal> ⊗ |p> state on the provided grid.

    - If sigma_p is None: choose nearest grid point (delta-like momentum eigenstate).
    - Else: Gaussian wavepacket in momentum space centered at p0.
    """
    p = np.asarray(p_grid, dtype=float)
    Np = int(p.shape[0])
    g = np.zeros(Np, dtype=np.complex128)
    e = np.zeros(Np, dtype=np.complex128)

    if sigma_p is None or sigma_p <= 0:
        idx = int(np.argmin(np.abs(p - float(p0))))
        if internal == "g":
            g[idx] = 1.0 + 0.0j
        else:
            e[idx] = 1.0 + 0.0j
    else:
        amp = np.exp(-0.5 * ((p - float(p0)) / float(sigma_p)) ** 2)
        amp = amp.astype(np.complex128)
        amp /= np.sqrt(float(np.vdot(amp, amp).real))
        if internal == "g":
            g = amp
        else:
            e = amp

    return np.stack([g, e], axis=0)


def expectation_p(
    psi_ge: npt.NDArray[np.complexfloating],
    p_grid: npt.NDArray[np.floating],
) -> float:
    psi = np.asarray(psi_ge, dtype=np.complex128)
    p = np.asarray(p_grid, dtype=float)
    prob = np.abs(psi[0]) ** 2 + np.abs(psi[1]) ** 2
    return float(np.sum(p * prob))


def populations_ge(psi_ge: npt.NDArray[np.complexfloating]) -> tuple[float, float]:
    psi = np.asarray(psi_ge, dtype=np.complex128)
    pop_g = float(np.vdot(psi[0], psi[0]).real)
    pop_e = float(np.vdot(psi[1], psi[1]).real)
    return pop_g, pop_e


def plot_velocity_and_populations(
    t: npt.NDArray[np.floating],
    v_expect: npt.NDArray[np.floating],
    pop_g: npt.NDArray[np.floating],
    pop_e: npt.NDArray[np.floating],
    *,
    img_path: str = "vpop_momentum.png",
):
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(t, v_expect, color="tab:blue", lw=1.2)
    ax1.set_xlabel("t")
    ax1.set_ylabel("<v>")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(t, pop_g, color="tab:green", lw=1.0, label="|g|^2")
    ax2.plot(t, pop_e, color="tab:red", lw=1.0, label="|e|^2")
    ax2.set_ylabel("populations")
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(img_path, dpi=180)
    plt.close(fig)

# 空间格点工具
def build_hamiltonian(potential_func, x_grid):
    dx = x_grid[1] - x_grid[0]
    num_points = len(x_grid)
    main = np.zeros(num_points)
    for i in range(num_points):
        main[i] = (1.0 / dx**2) + potential_func(x_grid[i])       # 主对角
    off = -0.5 / dx**2 * np.ones(num_points - 1)        # 副对角
    H = sp.diags([off, main, off], offsets=(-1, 0, 1), format='csr')  # type: ignore[arg-type]
    return H

def eigen_decomposition(H, num_states):
    energies, states = spla.eigsh(H, k=num_states, which='SA')
    idx = np.argsort(energies)
    energies = energies[idx]
    states = states[:, idx]
    return energies, states

def normalize(psi: np.ndarray, dx: float) -> np.ndarray:
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    if norm == 0:
        return psi
    return psi / norm

def evolve_grid(
    psi0: npt.NDArray[np.complexfloating],
    t: npt.NDArray[np.floating],
    H: sp.spmatrix,
    method: str = "CN",
    post_step: Callable[[float, npt.NDArray[np.complexfloating]], npt.NDArray[np.complexfloating]] | None = None,
    max_state: int = 10,
) -> npt.NDArray[np.complexfloating]:
    """空间格点 (1D 网格波函数) 的时间演化 (稀疏哈密顿量)。

    参数：
    - psi0：初始波函数，形状 (N,)
    - t：时间数组，形状 (T,)
    - H：稀疏哈密顿量，形状 (N, N)
    - method: "CN" (Crank–Nicolson) | "RK4" (solve_ivp/RK45) | "eigen" (本征分解)
    - post_step: 每一步后处理钩子，形式为 `psi <- post_step(dt, psi)` (可用于耗散/量子跳跃)
    - max_state: method=="eigen" 时使用的本征态数
    """
    dt = float(t[1] - t[0])
    num_steps = int(len(t))
    psi0 = np.asarray(psi0, dtype=np.complex128)

    if method == "CN":
        I = sp.eye(len(psi0), format="csr", dtype=np.complex128)  # type: ignore[arg-type]
        # Pylance 的 scipy stubs 对 spmatrix/astype 支持不完整，这里显式转为 csr_matrix 并指定 dtype。
        Hc = sp.csr_matrix(H, dtype=np.complex128)
        A = sp.csc_matrix(I + 1j * dt / 2 * Hc)
        B = sp.csc_matrix(I - 1j * dt / 2 * Hc)
        solve_A = spla.factorized(A)
        psi_history = np.zeros((len(psi0), num_steps), dtype=np.complex128)
        psi_history[:, 0] = psi0
        for n in range(num_steps - 1):
            if post_step is not None:
                psi_history[:, n] = post_step(dt, psi_history[:, n])
            b = B @ psi_history[:, n]
            psi_history[:, n + 1] = solve_A(b)
        return psi_history

    if method == "RK4":
        def rhs(_t, psi_vec):
            return -1j * (H @ psi_vec)

        sol = spi.solve_ivp(rhs, [float(t[0]), float(t[-1])], psi0, t_eval=t, method="RK45")
        if not sol.success:
            raise RuntimeError("Time evolution failed in RK4 method.")
        psi_history = np.asarray(sol.y, dtype=np.complex128)
        if post_step is not None:
            for n in range(num_steps):
                psi_history[:, n] = post_step(dt, psi_history[:, n])
        return psi_history

    if method == "eigen":
        psi_history = np.zeros((len(psi0), num_steps), dtype=np.complex128)
        psi_history[:, 0] = psi0
        energies, states = eigen_decomposition(H, max_state)
        coeffs_initial = states.conj().T @ psi_history[:, 0]
        for n in range(num_steps):
            time_factors = np.exp(-1j * energies * float(t[n]))
            psi_history[:, n] = np.dot(states, coeffs_initial * time_factors)
            if post_step is not None:
                psi_history[:, n] = post_step(dt, psi_history[:, n])
        return psi_history

    raise ValueError(f"Unknown method for evolve_grid: {method}")




# 时间工具
def evolve_state(
    psi0: npt.NDArray[np.complexfloating],
    t: npt.NDArray[np.floating],
    H: npt.NDArray[np.complexfloating],
    method: str = "RK4",
    post_step: Callable[[float, npt.NDArray[np.complexfloating]], npt.NDArray[np.complexfloating]] | None = None,
    max_state: int = 10,
) -> npt.NDArray[np.complexfloating]:
    """有限维态矢的时间演化 (稠密哈密顿量)。
    多步演化
    适用于多能级系统、两能级/三能级模型等 (H 为小维度的 dense 矩阵)。
    """
    dt = float(t[1] - t[0])
    num_steps = int(len(t))
    psi0 = np.asarray(psi0, dtype=np.complex128)
    H = np.asarray(H, dtype=np.complex128)

    if method == "CN":
        dim = int(psi0.shape[0])
        I = np.eye(dim, dtype=np.complex128)
        A = I + 1j * dt / 2 * H
        B = I - 1j * dt / 2 * H
        psi_history = np.zeros((dim, num_steps), dtype=np.complex128)
        psi_history[:, 0] = psi0
        for n in range(num_steps - 1):
            if post_step is not None:
                psi_history[:, n] = post_step(dt, psi_history[:, n])
            psi_history[:, n + 1] = np.linalg.solve(A, B @ psi_history[:, n])
        return psi_history

    if method == "RK4":
        def rhs(_t, psi_vec):
            return -1j * (H @ psi_vec)

        sol = spi.solve_ivp(rhs, [float(t[0]), float(t[-1])], psi0, t_eval=t, method="RK45")
        if not sol.success:
            raise RuntimeError("Time evolution failed in RK4 method.")
        psi_history = np.asarray(sol.y, dtype=np.complex128)
        if post_step is not None:
            for n in range(num_steps):
                psi_history[:, n] = post_step(dt, psi_history[:, n])
        return psi_history

    if method == "eigen":
        energies, states = np.linalg.eigh(H)
        idx = np.argsort(energies)
        energies = energies[idx]
        states = states[:, idx]
        coeffs_initial = states.conj().T @ psi0
        psi_history = np.zeros((psi0.shape[0], num_steps), dtype=np.complex128)
        for n in range(num_steps):
            time_factors = np.exp(-1j * energies[:max_state] * float(t[n]))
            psi_t = states[:, :max_state] @ (coeffs_initial[:max_state] * time_factors)
            psi_history[:, n] = psi_t
            if post_step is not None:
                psi_history[:, n] = post_step(dt, psi_history[:, n])
        return psi_history

    raise ValueError(f"Unknown method for evolve_state: {method}")

def dissipation_mcwf(
    psi: npt.NDArray[np.complexfloating],
    jump_states: list[npt.NDArray[np.complexfloating]],
    weights: npt.NDArray[np.floating] | list[float] | None = None,
    return_info: bool = False,
): # type: ignore[return]
    """基于范数损失的 MCWF (量子跳跃) 耗散-跳跃步骤。

    已经用非厄米哈密顿量做了一步演化，使得 $||psi||<1$。
    跳跃概率定义为: $p_{jump}=1-||psi||^2$。

    - 若发生跳跃：按照 `weights(跃迁概率权重)` 给出的分支比，从 `jump_states(多个基态)` 中抽样并投影到对应基态；
    - 若不跳跃：对当前态矢做归一化。
    """
    psi_arr = np.asarray(psi, dtype=np.complex128)
    norm_sq = float(np.vdot(psi_arr, psi_arr).real)
    p_jump = 1.0 - norm_sq
    rng = np.random.default_rng()

    if norm_sq <= 0.0 or norm_sq > 1.0:
        raise ValueError(f"RK4 error: ||psi||^2 = {norm_sq} invalid for MCWF.")

    if rng.random() < p_jump:
        if weights is None:
            idx = int(rng.integers(0, len(jump_states)))
        else:
            w = np.asarray(weights, dtype=float)
            s = float(np.sum(w))
            if s <= 0:
                idx = int(rng.integers(0, len(jump_states)))
            else:
                w = w / s
                idx = int(rng.choice(len(jump_states), p=w))
        if return_info:
            return np.asarray(jump_states[idx], dtype=np.complex128), True
        return np.asarray(jump_states[idx], dtype=np.complex128)

    if return_info:
        return psi_arr / np.sqrt(norm_sq), False
    return psi_arr / np.sqrt(norm_sq)


def evolve_mcwf(
    psi0: npt.NDArray[np.complexfloating],
    dt: float,
    H: npt.NDArray[np.complexfloating],
) -> npt.NDArray[np.complexfloating]:
    """
    单步 MCWF (量子跳跃) 演化，使用RK4方法。
    """
    H = np.asarray(H, dtype=np.complex128)
    psi0 = np.asarray(psi0, dtype=np.complex128)
    def rhs(_t, psi_vec):
        return -1j * (H @ psi_vec)
    k1 = rhs(0, psi0)
    k2 = rhs(0, psi0 + dt / 2 * k1)
    k3 = rhs(0, psi0 + dt / 2 * k2)
    k4 = rhs(0, psi0 + dt * k3)
    psi_next = psi0 + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return psi_next

def energy_expt(psi, H) -> float:
    psi = np.asarray(psi, dtype=np.complex128)
    H_psi = H @ psi
    E = np.vdot(psi, H_psi)
    return float(E.real)

def potential_expt(psi, potential_func, x) -> float:
    psi = np.asarray(psi, dtype=np.complex128)
    V_vals = np.array([potential_func(xi) for xi in x], dtype=float)
    Vpsi = V_vals * psi
    E_V = np.vdot(psi, Vpsi)
    return float(E_V.real)

def plot_time_evolution(
    x: npt.NDArray[np.floating],
    vectors_t: npt.NDArray[np.complexfloating],
    dt: float,
    V: Callable[[float], float],
    gif_path: str = "wave.gif",
    mode: str = "density",
    fps: int = 60,
    dpi: int = 120
):
    """
    mode: "density" | "real" | "imag" | "all"
    当 mode == "all" 时并排/堆叠绘制 density, real, imag 三个子图并同步播放。
    """
    vectors_t = np.asarray(vectors_t, dtype=np.complex128)
    N, T = vectors_t.shape

    # 预计算三种量
    rho_density = np.abs(vectors_t) ** 2
    rho_real = vectors_t.real
    rho_imag = vectors_t.imag

    # 计算能量时间序列（用于标题显示）
    H = build_hamiltonian(V, x)
    E = np.empty(T)
    for ti in range(T):
        E[ti] = energy_expt(vectors_t[:, ti], H)

    if mode == "all":
        # 三个子图（密度 / 实部 / 虚部），共享 x 轴
        fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        ax_d, ax_r, ax_i = axs

        # y 范围分别根据数据设置
        max_density = float(np.max(rho_density))
        y_min_d = 0.0
        y_max_d = 1.1 * max_density if max_density > 0 else 1.0
        y_abs_r = float(np.max(np.abs(rho_real)))
        y_min_r, y_max_r = (-1.1 * y_abs_r, 1.1 * y_abs_r) if y_abs_r > 0 else (-1.0, 1.0)
        y_abs_i = float(np.max(np.abs(rho_imag)))
        y_min_i, y_max_i = (-1.1 * y_abs_i, 1.1 * y_abs_i) if y_abs_i > 0 else (-1.0, 1.0)

        line_d, = ax_d.plot(x, rho_density[:, 0], color='C1')
        ax_d.set_ylabel("P(x,t)")
        ax_d.set_ylim(y_min_d, y_max_d)
        ax_d.grid(True, alpha=0.3)

        line_r, = ax_r.plot(x, rho_real[:, 0], color='C2')
        ax_r.set_ylabel("Re[ψ]")
        ax_r.set_ylim(y_min_r, y_max_r)
        ax_r.grid(True, alpha=0.3)

        line_i, = ax_i.plot(x, rho_imag[:, 0], color='C3')
        ax_i.set_ylabel("Im[ψ]")
        ax_i.set_ylim(y_min_i, y_max_i)
        ax_i.set_xlabel("x")
        ax_i.grid(True, alpha=0.3)

        def update(frame):
            line_d.set_ydata(rho_density[:, frame])
            line_r.set_ydata(rho_real[:, frame])
            line_i.set_ydata(rho_imag[:, frame])
            fig.suptitle(f"t={frame * dt:.5f}, E={E[frame]:.8f}")
            return (line_d, line_r, line_i)

        ani = animation.FuncAnimation(fig, update, frames=T, blit=True, interval=1000 / fps)
        ani.save(gif_path, writer='pillow', fps=fps, dpi=dpi)
        plt.close(fig)
        return

    # 单图模式选择
    if mode == "density":
        rho_t = rho_density
        max_rho = float(np.max(rho_t))
        y_min = 0.0
        y_max = 1.1 * max_rho if max_rho > 0 else 1.0
        ylabel = "P(x,t)"
    elif mode == "real":
        rho_t = rho_real
        y_abs = float(np.max(np.abs(rho_t)))
        y_min, y_max = (-1.1 * y_abs, 1.1 * y_abs) if y_abs > 0 else (-1.0, 1.0)
        ylabel = "Re[ψ(x,t)]"
    elif mode == "imag":
        rho_t = rho_imag
        y_abs = float(np.max(np.abs(rho_t)))
        y_min, y_max = (-1.1 * y_abs, 1.1 * y_abs) if y_abs > 0 else (-1.0, 1.0)
        ylabel = "Im[ψ(x,t)]"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    fig, ax = plt.subplots(figsize=(8, 4))
    line, = ax.plot(x, rho_t[:, 0], color='C1')
    ax.axhline(0.0, color='k', lw=0.8, alpha=0.3)
    ax.set_title("ψ(x,t)")
    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)

    def update_single(frame):
        line.set_ydata(rho_t[:, frame])
        ax.set_title(f"t={frame * dt:.5f}, E={E[frame]:.8f}")
        return (line,)

    ani = animation.FuncAnimation(fig, update_single, frames=T, blit=True, interval=1000 / fps)
    ani.save(gif_path, writer='pillow', fps=fps, dpi=dpi)
    plt.close(fig)

def plot_xv_dual_axis(
    t: npt.NDArray[np.floating],
    x_t: npt.NDArray[np.floating],
    v_t: npt.NDArray[np.floating],
    img_path: str = "xv_dual_axis.png",
    title: str = "x(t) and v(t)",
):
    """双 y 轴绘制 x(t) 与 v(t)。"""
    t = np.asarray(t, dtype=float)
    x_t = np.asarray(x_t, dtype=float)
    v_t = np.asarray(v_t, dtype=float)

    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax2 = ax1.twinx()

    ln1 = ax1.plot(t, x_t, color="C0", lw=1.6, label="x(t)")
    ln2 = ax2.plot(t, v_t, color="C1", lw=1.6, label="v(t)")

    ax1.set_xlabel("t")
    ax1.set_ylabel("x(t)", color="C0")
    ax2.set_ylabel("v(t)", color="C1")
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title)

    lines = ln1 + ln2
    labels = [str(line.get_label()) for line in lines]
    ax1.legend(lines, labels, loc="best")

    plt.tight_layout()
    plt.savefig(img_path, dpi=150)
    plt.close(fig)


def plot_populations(
    t: npt.NDArray[np.floating],
    psi_t: npt.NDArray[np.complexfloating],
    img_path: str = "populations.png",
    title: str = "Populations",
):
    """单 y 轴双曲线绘制 |c_g|^2 与 |c_e|^2。

    psi_t 形状支持 (2, T) 或 (T, 2)。
    """
    t = np.asarray(t, dtype=float)
    psi_t = np.asarray(psi_t, dtype=np.complex128)
    if psi_t.ndim != 2:
        raise ValueError(f"psi_t must be 2D, got shape={psi_t.shape}")

    if psi_t.shape[0] == 2:
        cg = psi_t[0, :]
        ce = psi_t[1, :]
    elif psi_t.shape[1] == 2:
        cg = psi_t[:, 0]
        ce = psi_t[:, 1]
    else:
        raise ValueError(f"psi_t must have a dimension of size 2, got shape={psi_t.shape}")

    Pg = np.abs(cg) ** 2
    Pe = np.abs(ce) ** 2

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(t, Pg, color="C0", lw=1.6, label="|c_g|^2")
    ax.plot(t, Pe, color="C1", lw=1.6, label="|c_e|^2")
    ax.set_xlabel("t")
    ax.set_ylabel("Population")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(img_path, dpi=150)
    plt.close(fig)


def plot_v_pop_dual_axis(
    t: npt.NDArray[np.floating],
    v_t: npt.NDArray[np.floating],
    psi_t: npt.NDArray[np.complexfloating],
    img_path: str = "v_pop_dual_axis.png",
    title: str = "v(t) and populations",
):
    """双 y 轴三曲线绘制 v(t) 与 |c_g|^2, |c_e|^2。

    左轴：v(t)
    右轴：Population
    psi_t 形状支持 (2, T) 或 (T, 2)。
    """
    t = np.asarray(t, dtype=float)
    v_t = np.asarray(v_t, dtype=float)
    psi_t = np.asarray(psi_t, dtype=np.complex128)
    if psi_t.ndim != 2:
        raise ValueError(f"psi_t must be 2D, got shape={psi_t.shape}")

    if psi_t.shape[0] == 2:
        cg = psi_t[0, :]
        ce = psi_t[1, :]
    elif psi_t.shape[1] == 2:
        cg = psi_t[:, 0]
        ce = psi_t[:, 1]
    else:
        raise ValueError(f"psi_t must have a dimension of size 2, got shape={psi_t.shape}")

    Pg = np.abs(cg) ** 2
    Pe = np.abs(ce) ** 2

    fig, ax_v = plt.subplots(figsize=(9, 4.8))
    ax_p = ax_v.twinx()

    ln_v = ax_v.plot(t, v_t, color="C1", lw=1.6, label="v(t)")
    ln_pg = ax_p.plot(t, Pg, color="C0", lw=1.2, label="|c_g|^2")
    ln_pe = ax_p.plot(t, Pe, color="C2", lw=1.2, label="|c_e|^2")

    ax_v.set_xlabel("t")
    ax_v.set_ylabel("v(t)", color="C1")
    ax_p.set_ylabel("Population")
    ax_p.set_ylim(-0.02, 1.02)
    ax_v.grid(True, alpha=0.3)
    ax_v.set_title(title)

    lines = ln_v + ln_pg + ln_pe
    labels = [str(line.get_label()) for line in lines]
    ax_v.legend(lines, labels, loc="best")

    plt.tight_layout()
    plt.savefig(img_path, dpi=150)
    plt.close(fig)