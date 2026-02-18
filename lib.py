import numpy as np

class Quantumket:

    # 波矢容器, 功能：归一化(normalize)，计算模方(norm2)，取得布居数(population)，位置/动量空间转换(transform)。
    def __init__(
        self,
        state: np.ndarray,
        internal_labels = None,
        Grid = None,
        external_types = None,
    ):
        # state: 波矢数据，要求为numpy数组，且维度至少为1。内态波矢数据应放在第一维，外态波矢数据应放在第二维及之后的维度。
        # internal_labels: 内部标签，用于标识波矢的内部结构，如自旋态、能级等，要求为列表或数组，长度与state的第一维相同
        # Grid: 位置/动量网格，用于标识波矢在位置空间中的坐标或动量空间中的坐标，要求为1D数组（单轴）或由1D数组组成的tuple（多轴），每个数组长度必须与state对应的外部维度大小相同,
        # external_types: 外部类型，标识波矢的外部空间类型，如位置空间（Position）或动量空间（Momentum），默认位置空间
        self.data = np.asarray(state, dtype=np.complex128)
        self.grid = None
        # external_types:
        # - None: 不显式声明当前表象（transform 会尽量用缓存/网格推断）
        # - 'position' 或 'momentum': 显式声明当前表象
        self.external_types = None if external_types is None else str(external_types).strip().lower()

        # 历史信息（尽量轻量）：用于 external_types=None 时推断当前表象
        self._last_external_types = self.external_types

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

        if self.data.ndim == 1 and Grid is not None:
            raise ValueError("当state为一维数组时，Grid必须为None")

        if self.external_types is not None and self.external_types not in {"position", "momentum"}:
            raise ValueError("external_types 必须为 None/'position'/'momentum'")

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

        - 总是对内态与外态全部求和。
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

    def population(self, index = None, label = None) -> float:
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
        """位置<->动量空间转换。
        约定：
        - self.data.shape = (n_internal, *external_shape)
        - 只对外部轴做 FFT：axes = (1, 2, ..., data.ndim-1)
        - 使用 norm='ortho' 保证往返互逆：ifft(fft(psi)) == psi（数值误差内）
        - 使用 ifftshift/fftshift 成对，避免中心定义不一致导致额外相位/重排
        """
        if self.data.ndim < 2:
            raise ValueError("transform 需要外部空间维度：data.ndim 必须 >= 2")

        ext = self.external_types
        axes = tuple(range(1, self.data.ndim))

        if ext == "position":
            psi = np.fft.ifftshift(self.data, axes=axes)
            psi = np.fft.fftn(psi, axes=axes, norm="ortho")
            self.data = np.fft.fftshift(psi, axes=axes)
            self.external_types = "momentum"
            self._last_external_types = "momentum"
            # 若有位置网格且为均匀网格，则生成对应的动量(k)网格
            if self.grid is not None:
                new_grid = []
                for g in self.grid:
                    N = g.shape[0]
                    dx = float(g[1] - g[0])
                    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
                    k = np.fft.fftshift(k)
                    new_grid.append(k.astype(float))
                self.grid = tuple(new_grid)
            return self

        if ext == "momentum":
            psi = np.fft.ifftshift(self.data, axes=axes)
            psi = np.fft.ifftn(psi, axes=axes, norm="ortho")
            self.data = np.fft.fftshift(psi, axes=axes)
            self.external_types = "position"
            self._last_external_types = "position"
            # 不保存巨大的“绝对 x 网格”；若有动量(k)网格则反推生成居中 x 网格
            if self.grid is not None:
                # 这里假设 self.grid 是均匀 k 网格（由 position->momentum 生成或用户提供）
                new_grid = []
                for g in self.grid:
                    N = g.shape[0]
                    dk = float(g[1] - g[0])
                    if dk == 0.0:
                        raise ValueError("动量网格步长 dk=0，无法反推位置网格")
                    dx = 2.0 * np.pi / (N * abs(dk))
                    # 只能生成“居中坐标系”的 x；绝对起点用 self.x_origin 另行记录
                    x = (np.arange(N) - (N // 2)) * dx
                    new_grid.append(x.astype(float))
                self.grid = tuple(new_grid)
            return self

        raise ValueError("当前Quantumket对象的external_types属性值无效，无法进行位置/动量空间转换")

    def __str__(self):
        return f"Quantumket(internal_labels={getattr(self, 'internal_labels', None)}, grid={getattr(self, 'grid', None)}) \n data={self.data}"



class KineticOperator:
    # 动能算符，接口范例：
    def apply(self, ket: Quantumket, mass: float):
        if getattr(ket, 'external_types', None) != 'momentum':
            raise ValueError("KineticOperator仅适用于外部类型为'momentum'的Quantumket对象")
        # 计算动能作用后的波矢
        p_date = ket.data[1:]  # 假设动量数据从第二维开始
        kinetic_data = p_date**2 / (2 * mass)
        new_data = np.concatenate([ket.data[:1], kinetic_data], axis=0)
        return Quantumket(new_data, internal_labels=getattr(ket, 'internal_labels', None), external_types=getattr(ket, 'external_types', None))

    def expect(self, ket: Quantumket, mass: float):
    
        if getattr(ket, 'external_types', None) != 'momentum':
            raise ValueError("KineticOperator仅适用于外部类型为'momentum'的Quantumket对象")
        # 计算动能期望
        p_date = ket.data[1:]  # 假设动量数据从第二维开始
        kinetic_data = p_date**2 / (2 * mass)
        kinetic_energy = np.sum(kinetic_data * np.abs(ket.data[0])**2)  # 假设振幅数据在第一维
        return kinetic_energy

