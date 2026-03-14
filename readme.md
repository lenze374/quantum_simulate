# 简介

这是一个基于 `numpy`、`scipy` 和 `matplotlib` 的冷原子一维波函数模拟工具库，当前主要用于 Monte Carlo Wave Function (MCWF) 框架下的动量空间演化。当前仓库已经实现了：

1. 统一的波函数容器 `Quantumket`
2. 两种时间推进器 `RKstepper` 和 `CNstepper`
3. 通用的 MCWF 跳跃接口 `mcwf_step`
4. 基础绘图接口
5. 若干具体模型脚本，例如亚多普勒冷却和 VSCPT

当前代码的组织方式很简单：

1. 波函数统一由 `Quantumket` 表示
2. 哈密顿量/算符类只要提供 `apply(psi)` 即可交给求解器使用
3. 跳跃通道通过 `JUMP_CHANNELS` 元组交给 `mcwf_step`

## 依赖

当前代码依赖：

1. `numpy`
2. `scipy`
3. `matplotlib`

安装示例：

```bash
pip install numpy scipy matplotlib
```

## 单位与约定

目前脚本通常采用一组无量纲单位，常见约定为：

1. `\hbar = 1`
2. `k_L = 1`
3. 时间、失谐、拉比频率、自发辐射速率都按同一频率单位计量

在动量空间脚本中，通常取离散格点

$$
p_n = p_0 + n
$$

因此光场中的 $e^{\pm i k z}$ 对应动量格点上的最近邻平移。

## 核心数据结构

### 波函数 `Quantumket`

`Quantumket` 是库里最核心的对象，用来保存“内态 $\otimes$ 外态”的波函数。

构造形式：

```python
ket = Quantumket(state, internal_labels, Grid, external_types)
```

参数含义：

1. `state`：复振幅数组，形状通常为 `(n_internal, N1, N2, ...)`
2. `internal_labels`：内态标签，例如 `['g', 'e']` 或 `['g-', 'g+', 'e0']`
3. `Grid`：外部网格，必须是由一维数组组成的 `tuple`，例如 `(p_grid,)` 或 `(x_grid,)`
4. `external_types`：外态表象，支持 `"position"`、`"momentum"`，别名 `"x"`、`"p"`、`"k"`

典型示例：

```python
psi0 = np.zeros((2, 401), dtype=complex)
psi0[0, 200] = 1.0
p_grid = np.linspace(-200, 200, 401)

ket = Quantumket(
    psi0,
    ["g", "e"],
    (p_grid,),
    "p",
)
```

`Quantumket` 的主要方法如下。

#### `norm2()`

返回总范数 $\|\psi\|^2$。若提供了外部网格，则自动乘上体元 `dV`。

```python
n2 = ket.norm2()
```

#### `normalize()`

原地归一化，并返回自身。

```python
ket = ket.normalize()
```

#### `population(label=..., index=...)`

返回指定内态的布居。

```python
pop_e = ket.population("e")
pop_g0 = ket.population(index=0)
```

#### `expect()`

返回外态坐标的期望值。在动量表象下就是 $\langle p \rangle$，在位置表象下就是 $\langle x \rangle$。

```python
p_mean = ket.expect()
v_mean = ket.expect() / M
```

#### `transform()`

在位置表象和动量表象之间做 FFT 变换，返回一个新的 `Quantumket`，不会修改原对象。

```python
ket_x = ket_p.transform()
ket_p = ket_x.transform()
```

### 内部快速构造 `_fast_from()`

这是求解器和 MCWF 内部使用的快速构造接口，用于在不重复做输入检查的情况下，用新数据生成一个与旧 `ket` 共享元数据的对象，以跳过检查加速运算。一般脚本层不直接调用，除非你非常清楚自己在做什么。

## 算符/哈密顿量接口

库当前采用一个简单约定：

1. 算符写成类
2. 类提供 `apply(psi)` 方法
3. `apply(psi)` 返回 `H psi`，且返回数组形状必须与 `psi` 相同

最小示例：

```python
class H_eff:
    def __init__(self, p_grid, M, delta):
        p = np.asarray(p_grid, dtype=float)
        self.T = p * p / (2.0 * M)
        self.shift = -delta

    def apply(self, psi: np.ndarray) -> np.ndarray:
        out = np.empty_like(psi)
        out[0] = self.T * psi[0]
        out[1] = (self.T + self.shift) * psi[1]
        return out
```

对于具体模型，可以在 `apply()` 中手工写：

1. 动能对角项
2. 最近邻动量耦合
3. 非厄米衰减项

这是当前仓库里最常用的写法。

## 求解器接口

### `RKstepper(ket, Operator, dt)`

四阶 Runge-Kutta 步进器。适合：

1. 快速验证模型
2. `apply()` 容易写成显式切片形式
3. 哈密顿量不需要构造稀疏矩阵

使用示例：

```python
ket = RKstepper(ket, heff, dt)
```

这里要求：

1. `ket` 是 `Quantumket`
2. `heff` 提供 `apply(psi)`
3. `dt` 是单步时间间隔

### `CNstepper(ket, Operator, dt, tol=1e-10, maxiter=None, restart=None)`

Crank-Nicolson 步进器。适合：

1. 更关注稳定性
2. 希望用隐式格式推进
3. 算符可通过线性方程方式求解

使用示例：

```python
ket = CNstepper(ket, heff, dt)
```

若 `Operator` 提供以下属性，则 `CNstepper` 会走更快的路径：

1. `B`
2. `solve_A`
3. 可选的 `cn_dt`

或者提供 `prepare_cn(dt)` 进行懒加载准备。

## 动能算符示例

### `KineticOperator(grid, mass=1.0, hbar=1.0)`

这是一个简单的动能算符示例类，主要用于：

1. 演示 `apply(psi)` 接口
2. 在动量/波矢表象中作用动能对角项
3. 计算动能期望值

典型示例：

```python
kop = KineticOperator((p_grid,), mass=M, hbar=1.0)
Tpsi = kop.apply(ket.data)
Texp = kop.expect(ket.data)
```

这个类是接口范例，而不是当前长时间 MCWF 演化的主力接口。对于具体模型，一般会直接把动能并入 `H_eff.apply()`。

## MCWF 跳跃接口

### `mcwf_step(ket, rng, JUMP_CHANNELS)`

这是当前仓库统一使用的量子跳跃步骤。调用前默认已经用非厄米哈密顿量推进过一步，因此跳跃概率由范数损失给出：

$$
dp = 1 - \|\psi\|^2
$$

调用示例：

```python
rng = np.random.default_rng()
ket = RKstepper(ket, heff, dt)
ket, jumped = mcwf_step(ket, rng, JUMP_CHANNELS)
```

其中 `JUMP_CHANNELS` 的约定是一个元组，每个元素形如：

```python
(e_idx, g_idx, branch_ratio)
```

含义分别是：

1. `e_idx`：激发态分量索引
2. `g_idx`：跃迁后的基态分量索引
3. `branch_ratio`：该通道的分支比权重

示例一：两能级模型

```python
JUMP_CHANNELS = (
    (1, 0, 1.0),
)
```

示例二：VSCPT 三能级模型

```python
JUMP_CHANNELS = (
    (2, 0, 0.5),
    (2, 1, 0.5),
)
```

当前实现的 recoil 模型是：

1. 一维动量空间
2. 左右两个方向等概率
3. recoil 对应动量格点平移一格

如果以后需要更精细的角分布或多维 recoil，需要扩展这一接口。

## 绘图接口

### `plot_populations(t, pops, labels=None, ax=None, title=None, save_path=None, show=True)`

用于画随时间变化的布居曲线。

```python
plot_populations(
    t,
    rhoe_total_history,
    labels=["e"],
    title="Excited State Population",
    save_path="population.png",
)
```

支持：

1. 单条曲线 `shape (T,)`
2. 多条曲线 `shape (n_series, T)`
3. 转置后的 `shape (T, n_series)`

### `plot_dual_axis(x, y_left, y_right, left_label="left", right_label="right", title=None, save_path=None, show=True)`

用于画双纵轴时间曲线。当前仓库里通常拿它画 `v(t)` 和 `x(t)`。

```python
plot_dual_axis(
    t,
    v_history,
    x_history,
    "Velocity",
    "Position",
    save_path="velocity_position.png",
)
```

### `animate_1d(x, y_t, interval_ms=50, title=None, ylabel=None)`

用于生成最简单的一维动画对象，返回 `matplotlib.animation.FuncAnimation`。

```python
ani = animate_1d(x_grid, density_t, interval_ms=30, title="density")
ani.save("density.mp4")
```

## 当前模型脚本

### `subDropping.py`

六能级亚多普勒冷却模型，当前采用：

1. 动量空间表示
2. `H_eff.apply()` 手写最近邻耦合
3. `RKstepper + mcwf_step`
4. 输出速度-位置图和总激发态布居图

运行：

```bash
python subDropping.py
```

### `VSCPT.py`

三能级 Λ 型 VSCPT 模型，结构与 `subDropping.py` 基本一致，但内部态和跃迁通道更简单。

运行：

```bash
python VSCPT.py
```

## 推荐的建模流程

若要新建一个模型，建议按下面顺序写：

1. 定义参数、网格和初态
2. 构造 `Quantumket`
3. 写一个 `H_eff` 类，并实现 `apply(psi)`
4. 定义 `JUMP_CHANNELS`
5. 用 `RKstepper` 或 `CNstepper` 推进
6. 每步之后调用 `mcwf_step`
7. 记录需要的物理量并绘图

一个最小骨架如下：

```python
import numpy as np
from lib import *

p_grid = np.linspace(-100, 100, 201)
psi0 = np.zeros((2, 201), dtype=complex)
psi0[0, 100] = 1.0

class H_eff:
    def __init__(self, p_grid):
        self.p = np.asarray(p_grid, dtype=float)
        self.T = self.p * self.p / 2.0

    def apply(self, psi: np.ndarray) -> np.ndarray:
        out = np.empty_like(psi)
        out[0] = self.T * psi[0]
        out[1] = self.T * psi[1] - 1j * 0.5 * psi[1]
        return out

ket = Quantumket(psi0, ["g", "e"], (p_grid,), "p")
heff = H_eff(p_grid)
rng = np.random.default_rng()
JUMP_CHANNELS = ((1, 0, 1.0),)

for _ in range(1000):
    ket = RKstepper(ket, heff, 1e-3)
    ket, jumped = mcwf_step(ket, rng, JUMP_CHANNELS)
```

## 当前限制

当前代码已经能支持一批一维动量空间模型，但仍有一些明确限制：

1. `mcwf_step` 默认是一维 recoil，且左右等概率，如果要处理更复杂的模型，需要重写 `mcwf_step` 并拓展接口。
2. 边界处理目前是直接截断，长时间演化时要自行检查动量尾部是否碰边界
3. `Quantumket.transform()` 仅适用于均匀网格 FFT 语义

