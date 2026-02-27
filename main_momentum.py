import numpy as np
import lib


# Dimensionless units: hbar = 1, kL = 1, Gamma = 1
M = 800.0
Gamma = 1.0
kL = 1.0

delta = -2.0  # red detuning
Omega0 = 0.7   # beam Rabi amplitude (see lib._heff_apply_momentum)


def main():
    # Simulation time grid
    N = 20000
    dt = 0.01
    t = np.linspace(0.0, N * dt, N + 1)

    # Initial (mean) velocity and momentum
    v0 = 0.67
    p0 = M * v0

    # Momentum grid: needs to be wide enough to contain recoil random walk.
    # For v0~0.67 this is manageable; for v0~4.5 you'd need thousands of steps.
    n_max = 1500
    p_grid = lib.build_momentum_grid(p_center=p0, kL=kL, n_max=n_max)

    # Initial state: ground internal state, localized at p0 (or use sigma_p for a packet)
    psi = lib.make_initial_momentum_state(p_grid, p0=p0, sigma_p=None, internal="g")

    # Stable propagator for large momentum grids
    H_eff = lib.build_heff_momentum_sparse(
        p_grid,
        M=M,
        delta=delta,
        Gamma=Gamma,
        Omega0=Omega0,
    )
    cn_step = lib.make_cn_propagator(H_eff, dt)

    v_expect = np.zeros(N + 1, dtype=float)
    pop_g = np.zeros(N + 1, dtype=float)
    pop_e = np.zeros(N + 1, dtype=float)

    # Initial observables
    v_expect[0] = lib.expectation_p(psi, p_grid) / M
    pop_g[0], pop_e[0] = lib.populations_ge(psi)

    for step in range(N):
        psi = lib.evolve_mcwf_momentum_cn(psi, cn_step)
        psi, jumped, _recoil = lib.dissipation_mcwf_momentum(psi, recoil_steps=1, return_info=True)

        v_expect[step + 1] = lib.expectation_p(psi, p_grid) / M
        pop_g[step + 1], pop_e[step + 1] = lib.populations_ge(psi)

        if step % 2000 == 0 and step > 0:
            print(
                f"step={step:6d}  t={t[step]:8.3f}  <v>={v_expect[step]: .6f}  pop_e={pop_e[step]: .4f}  jumped={jumped}"
            )

    lib.plot_velocity_and_populations(
        t,
        v_expect,
        pop_g,
        pop_e,
        img_path="result/vpop_momentum_v0=0.67.png",
    )
    print("Done. Saved: result/vpop_momentum_v0=0.67.png")


if __name__ == "__main__":
    main()
