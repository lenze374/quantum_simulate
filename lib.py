import numpy as np
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Quantumket:
    # 这个基本功能居然就写了200行，感觉到通用付出的代价了233
    # 波矢容器, 功能：归一化(normalize)，计算模方(norm2)，取得布居数(population)，取得位置动量期望（expect）,位置/动量空间转换(transform)，x_origin（绝对位置原点）
    def __init__(
        self,
        state: np.ndarray,
        internal_labels = None,
        Grid = None,
        external_types = None,
    ):
        # state: 波矢数据，要求为numpy数组，且维度至少为1。
        # internal_labels: 内部标签，用于标识波矢的内部结构，如自旋态、能级等，要求为列表或数组，长度与state的第一维相同
        # Grid: 位置/动量网格，用于标识波矢在位置空间中的坐标或动量空间中的坐标，要求为1D数组（单轴）或由1D数组组成的tuple（多轴），每个数组长度必须与state对应的外部维度大小相同,
        # external_types: 外部类型，标识波矢的外部空间类型，如位置空间（Position）或动量空间（Momentum），默认位置空间
        self.data = np.asarray(state, dtype=np.complex128)
        self.grid = None
        # external_types:
        # - None: 如果纯内态（外部空间存在时必须显式声明 position/momentum）

        if external_types is None:
            self.external_types = None
        else:
            ext = str(external_types).strip().lower()
            if ext == "x":
                ext = "position"
            elif ext in {"p", "k"}:
                ext = "momentum"
            self.external_types = ext

        # 绝对位置原点（不强行保存巨大的 x 绝对网格）
        # 约定：当在 position 表象下，实际坐标可理解为 x_abs = x_origin + x_centered
        self.x_origin = 0.0

        if internal_labels is not None:
            if len(internal_labels) != self.data.shape[0]:
                raise ValueError("internal_labels的长度必须与state的第一维相同")
            else:
                self.internal_labels = tuple(internal_labels)
        else:
            self.internal_labels = tuple(f"s{i}" for i in range(self.data.shape[0]))

        if self.data.ndim == 1:
            if Grid is not None:
                raise ValueError("当state为一维数组时，Grid必须为None")
        else:
            if Grid is None:
                raise ValueError("当state具有外部空间维度(ndim>=2)时，必须提供 Grid")

        if self.external_types is not None and self.external_types not in {"position", "momentum"}:
            raise ValueError("external_types 必须为 None/'position'/'momentum'，也允许别名：'x' 或 'p'/'k'")

        if Grid is not None:
            if self.external_types is None:
                raise ValueError("提供 Grid 时，必须显式指定 external_types='position' 或 'momentum'")
            if self.data.ndim < 2:
                raise ValueError("提供 Grid 时，state 必须具有外部空间维度 (ndim>=2)")
            if not isinstance(Grid, (tuple, list)) or len(Grid) == 0:
                raise TypeError("Grid 必须为由 1D ndarray 组成的 tuple/list，例如 (x,) 或 (x,y)")

            grids = []
            for g in Grid:
                arr = np.asarray(g, dtype=float)
                if arr.ndim != 1:
                    raise ValueError(f"Grid 的每个轴必须为 1D 数组，但得到 shape={arr.shape}")
                if arr.shape[0] < 2:
                    raise ValueError("每个 Grid 轴至少需要 2 个点，才能定义步长")
                grids.append(arr)
            self.grid = tuple(grids)

            if len(self.grid) != (self.data.ndim - 1):
                raise ValueError(
                    f"Grid 维数={len(self.grid)}，但 state 外部维数={self.data.ndim - 1}"
                )
            for axis, (g, n) in enumerate(zip(self.grid, self.data.shape[1:])):
                if g.shape[0] != n:
                    raise ValueError(
                        f"Grid 第 {axis} 轴长度必须等于外部维度大小 {n}，但得到 {g.shape[0]}"
                    )

    def _dV(self) -> float:
        """外部空间体元：dx, dx*dy, ...；若无 grid 或无外部空间则返回 1。"""
        if self.grid is None:
            return 1.0
        dV = 1.0
        for g in self.grid:
            dV *= float(abs(g[1] - g[0]))
        return float(dV)
        
    def norm2(self) -> float:
        """总范数 ||psi||^2。
        - 若提供 grid（连续采样网格），则乘以体元 dV。
        """
        n2 = float(np.vdot(self.data, self.data).real)
        if self.data.ndim >= 2 and self.grid is not None:
            n2 *= self._dV()
        return float(n2)

    def normalize(self):
        """把整个态归一化到 norm2=1（自动处理外部维数与 dV）。"""
        n2 = self.norm2()
        if n2 <= 0.0:
            raise ValueError("波矢的模方为零，无法归一化")
        self.data = self.data / np.sqrt(n2)
        return self

    def population(self, label = None, index = None) -> float:
        # 获取布居数
        if index is not None and label is not None:
            raise ValueError("只能指定index或label中的一个，不能同时指定")
        if label is None and index is None:
            raise ValueError("必须指定index或label中的一个，不能同时不指定")
        if label is not None:
            if not hasattr(self, 'internal_labels'):
                raise ValueError("当前Quantumket对象没有internal_labels属性，无法通过label获取布居数")
            if label not in self.internal_labels:
                raise ValueError(f"指定的label '{label}' 不在internal_labels中")
            index = self.internal_labels.index(label)
        if index is None:
            raise ValueError("index 解析失败")
        if index < 0 or index >= self.data.shape[0]:
            raise IndexError("index超出范围")

        psi_a = self.data[index]
        pa = float(np.vdot(psi_a, psi_a).real)
        if psi_a.ndim >= 1 and self.grid is not None:
            pa *= self._dV()
        return float(pa)

    def transform(self):
        """位置<->动量空间转换（不修改当前对象，返回一个新的 Quantumket）。
        """
        if self.data.ndim < 2:
            raise ValueError("transform 需要外部空间维度：data.ndim 必须 >= 2")

        ext = self.external_types
        if ext not in {"position", "momentum"}:
            raise ValueError("当前Quantumket对象的external_types属性值无效，无法进行位置/动量空间转换")
        axes = tuple(range(1, self.data.ndim))

        if ext == "position":
            psi = np.fft.ifftshift(self.data, axes=axes)
            psi = np.fft.fftn(psi, axes=axes, norm="ortho")
            new_data = np.fft.fftshift(psi, axes=axes)
            new_external_types = "momentum"
            # 若有位置网格且为均匀网格，则生成对应的动量(k)网格
            grids = []
            for g in self.grid:
                N = g.shape[0]
                dx = float(g[1] - g[0])
                k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
                k = np.fft.fftshift(k)
                grids.append(k.astype(float))
            new_grid = tuple(grids)

            new_ket = Quantumket(
                new_data,
                internal_labels=self.internal_labels,
                Grid=new_grid,
                external_types=new_external_types,
            )
            new_ket.x_origin = float(self.x_origin)
            return new_ket

        if ext == "momentum":
            psi = np.fft.ifftshift(self.data, axes=axes)
            psi = np.fft.ifftn(psi, axes=axes, norm="ortho")
            new_data = np.fft.fftshift(psi, axes=axes)
            new_external_types = "position"
            grids = []
            for g in self.grid:
                N = g.shape[0]
                dk = float(g[1] - g[0])
                dx = 2.0 * np.pi / (N * abs(dk))
                # 只能生成“居中坐标系”的 x；绝对起点用 x_origin 另行记录
                x = (np.arange(N) - (N // 2)) * dx
                grids.append(x.astype(float))
            new_grid = tuple(grids)

            new_ket = Quantumket(
                new_data,
                internal_labels=self.internal_labels,
                Grid=new_grid,
                external_types=new_external_types,
            )
            new_ket.x_origin = float(self.x_origin)
            return new_ket

        raise ValueError("当前Quantumket对象的external_types属性值无效，无法进行位置/动量空间转换")

    def expect(self):
        """外部坐标期望值 ⟨x⟩/⟨p⟩（取决于 external_types 与 grid 的含义）。

        约定：
        - self.data shape = (n_internal, N1, N2, ...)
        - self.grid 是由外部每个轴的 1D 网格数组组成的 tuple，例如 (x,) 或 (kx, ky)

        返回：
        - 1D 外部空间：float
        - 多维外部空间：tuple[float, ...]，每个轴一个期望值
        """
        if self.data.ndim < 2:
            raise ValueError("expect 需要外部空间维度：data.ndim 必须 >= 2")
        if self.grid is None:
            raise ValueError("expect 需要 grid")

        prob = np.sum(np.abs(self.data) ** 2, axis=0)
        dV = float(self._dV())
        norm = float(np.sum(prob) * dV)
        if norm <= 0.0:
            raise ValueError("态的范数为 0，无法计算期望值")

        nd = prob.ndim
        exps: list[float] = []
        for axis, g in enumerate(self.grid):
            if axis >= nd:
                raise ValueError("grid 维数与波函数外部维数不匹配")
            shape = [1] * nd
            shape[axis] = int(np.asarray(g).shape[0])
            coord = np.asarray(g, dtype=float).reshape(shape)
            exps.append(float(np.sum(prob * coord) * dV / norm))

        return exps[0] if len(exps) == 1 else tuple(exps)

    def __str__(self):
        return f"Quantumket(internal_labels={getattr(self, 'internal_labels', None)}, grid={getattr(self, 'grid', None)}) \n data={self.data}"

    @staticmethod
    def _fast_from(template: "Quantumket", data: np.ndarray) -> "Quantumket":
        """快速构造一个与 template 共享 meta 的 Quantumket，只替换 data。

        用途：给求解器/步进器返回新 ket 时，避免每一步都跑一次 __init__ 的输入检查。
        注意：这是内部接口，不做 shape/grid 一致性校验。
        """
        k = Quantumket.__new__(Quantumket)
        k.data = np.asarray(data, dtype=np.complex128)
        k.grid = getattr(template, "grid", None)
        k.external_types = getattr(template, "external_types", None)
        k.x_origin = float(getattr(template, "x_origin", 0.0))
        k.internal_labels = getattr(template, "internal_labels", None)
        return k

def CNstepper(ket: Quantumket, Operator, dt: float, tol: float = 1e-10, maxiter=None, restart=None):
    """Crank–Nicolson 步进器。

    约定：Operator.apply(psi) -> ndarray，与 psi 同 shape。
    - psi 是 numpy 数组（通常就是 ket.data）
    """
    shape = ket.data.shape
    n = int(ket.data.size)

    alpha = 0.5j * dt
    Hpsi = Operator.apply(ket.data)
    rhs_vec = np.asarray(ket.data - alpha * Hpsi, dtype=np.complex128).reshape(n)
    x0 = np.asarray(ket.data, dtype=np.complex128).reshape(n)

    def matvec(v: np.ndarray) -> np.ndarray:
        psi = np.asarray(v, dtype=np.complex128).reshape(shape)
        Hv = Operator.apply(psi)
        out = psi + alpha * np.asarray(Hv)
        return np.asarray(out, dtype=np.complex128).reshape(n)

    A = spla.LinearOperator((n, n), matvec=matvec, dtype=np.complex128)

    sol, info = spla.gmres(A,rhs_vec, x0=x0, rtol=float(tol), atol=0.0, restart=restart, maxiter=maxiter)

    if info != 0:
        raise RuntimeError(f"GMRES 未收敛或失败 (info={info})；可尝试减小 dt、放宽 tol、增大 maxiter/restart")

    new_data = np.asarray(sol, dtype=np.complex128).reshape(shape)
    return Quantumket._fast_from(ket, new_data)

def RKstepper(ket: Quantumket, Operator, dt: float):
    """经典 RK4 步进器。

    同 CNstepper 约定：Operator.apply(psi) -> ndarray。
    """
    dt = float(dt)

    # 与 CN 同一物理约定：i dψ/dt = Hψ  =>  dψ/dt = -i Hψ
    def f(data: np.ndarray):
        return (-1j) * np.asarray(Operator.apply(data))

    k1 = f(ket.data)
    k2 = f(ket.data + 0.5 * dt * k1)
    k3 = f(ket.data + 0.5 * dt * k2)
    k4 = f(ket.data + dt * k3)

    new_data = ket.data + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return Quantumket._fast_from(ket, new_data)


class KineticOperator:
    # 动能算符，接口范例，不适用长时间演化
    def __init__(self, grid, *, mass: float = 1.0, hbar: float = 1.0):
        # grid: tuple/list of 1D arrays, e.g. (k,) or (kx, ky)
        if not isinstance(grid, (tuple, list)) or len(grid) == 0:
            raise TypeError("grid 必须是由 1D 数组组成的 tuple/list")
        grids = []
        for g in grid:
            arr = np.asarray(g, dtype=float)
            if arr.ndim != 1:
                raise ValueError("grid 的每个轴必须是一维数组")
            grids.append(arr)
        self.grid = tuple(grids)
        self.mass = float(mass)
        self.hbar = float(hbar)

    def _dV(self) -> float:
        if self.grid is None:
            return 1.0
        dV = 1.0
        for g in self.grid:
            dV *= float(abs(g[1] - g[0])) if g.shape[0] >= 2 else 1.0
        return float(dV)

    def apply(self, psi: np.ndarray) -> np.ndarray:
        """在动量(更准确：波矢 k)表象下作用动能算符。"""
        grid = self.grid
        nd = len(grid)
        k2 = 0.0
        for axis, k in enumerate(grid):
            shape = [1] * nd
            shape[axis] = int(np.asarray(k).shape[0])
            k_axis = np.asarray(k, dtype=float).reshape(shape)
            k2 = k2 + k_axis * k_axis
        T = (float(self.hbar) ** 2) * k2 / (2.0 * float(self.mass))

        psi_arr = np.asarray(psi, dtype=np.complex128)
        return psi_arr * T[np.newaxis, ...]

    def expect(self, psi: np.ndarray) -> float:
        """返回 <T>，使用离散网格体元近似（若有 grid）。"""
        grid = self.grid
        nd = len(grid)
        k2 = 0.0
        for axis, k in enumerate(grid):
            shape = [1] * nd
            shape[axis] = int(np.asarray(k).shape[0])
            k_axis = np.asarray(k, dtype=float).reshape(shape)
            k2 = k2 + k_axis * k_axis
        T = (float(self.hbar) ** 2) * k2 / (2.0 * float(self.mass))

        psi_arr = np.asarray(psi, dtype=np.complex128)
        exp_density = np.vdot(psi_arr, psi_arr * T[np.newaxis, ...]).real
        return float(exp_density) * float(self._dV())


def plot_populations(t, pops, labels=None, ax=None, title=None, save_path=None, show=True):
    """最基础的布居绘图：只画外部传入的 populations。

    参数约定（尽量宽松）：
    - t: 1D 时间轴
    - pops: shape (n_series, T) 或 (T, n_series) 或 1D
    """
    t = np.asarray(t)
    y = np.asarray(pops)
    if y.ndim == 1:
        y = y[np.newaxis, :]
    if y.shape[0] == t.shape[0] and y.shape[-1] != t.shape[0]:
        y = y.T

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    n_series = int(y.shape[0])
    if labels is None:
        labels = [f"s{i}" for i in range(n_series)]

    for i in range(n_series):
        ax.plot(t, y[i], label=str(labels[i]) if i < len(labels) else f"s{i}")

    ax.set_xlabel("t")
    ax.set_ylabel("population")
    if title is not None:
        ax.set_title(str(title))
    ax.legend()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_dual_axis(x, y_left, y_right, left_label="left", right_label="right", title=None, save_path=None, show=True):
    """双 y 轴基础绘图：完全不做任何物理计算，只画两条序列。"""
    x = np.asarray(x)
    y_left = np.asarray(y_left)
    y_right = np.asarray(y_right)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(x, y_left, color="C0")
    ax2.plot(x, y_right, color="C1")

    ax1.set_xlabel("x")
    ax1.set_ylabel(str(left_label), color="C0")
    ax2.set_ylabel(str(right_label), color="C1")
    if title is not None:
        ax1.set_title(str(title))

    if save_path is not None:
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig, (ax1, ax2)


def animate_1d(x, y_t, interval_ms=50, title=None, ylabel=None):
    """最基础 1D 动画：给定 x 与随时间变化的 y_t。

    - x: shape (N,)
    - y_t: shape (T, N) 或 (N, T)

    返回 FuncAnimation；保存由外部调用 `ani.save(...)` 完成。
    """
    x = np.asarray(x)
    y = np.asarray(y_t)
    if y.ndim != 2:
        raise ValueError("y_t 必须是 2D 数组")
    if y.shape[0] == x.shape[0] and y.shape[1] != x.shape[0]:
        y = y.T

    fig, ax = plt.subplots()
    (line,) = ax.plot(x, y[0])
    ax.set_xlabel("x")
    if ylabel is not None:
        ax.set_ylabel(str(ylabel))
    if title is not None:
        ax.set_title(str(title))

    def _update(i):
        line.set_ydata(y[i])
        return (line,)

    ani = animation.FuncAnimation(fig, _update, frames=int(y.shape[0]), interval=float(interval_ms), blit=True)
    return ani