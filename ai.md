## 亚反冲冷却（VSCPT）

### 物理机制

**前提**：需要存在两个基态通过同一个激发态形成的 Λ 型结构，并且两束反向传播激光分别选择性耦合这两个基态。这样在合适的动量子空间中，会出现一个不与激光耦合的暗态；原子经过不断的吸收-自发辐射循环后，逐渐被抽运并积累到暗态附近，从而在动量空间中压缩，实现低于单光子反冲极限的冷却。

**激光组态**：两束频率均为 $\omega$ 的激光沿 $z$ 轴反向传播，其中一束为沿 $+z$ 方向传播的 $\sigma^+$ 圆偏振光，另一束为沿 $-z$ 方向传播的 $\sigma^-$ 圆偏振光。

电场可写为

$$
\mathbf{E}(z,t)=\frac{E_0}{2}\left(\hat{\mathbf e}_+e^{i(kz-\omega t)}+\hat{\mathbf e}_-e^{i(-kz-\omega t)}\right)+\mathrm{c.c.}
$$
其中 $\hat{\mathbf e}_{\pm}$ 为圆偏振基矢量。

**原子能级**：考虑 $J_g=1\to J_e=1$ 跃迁。基态有三个磁子能级 $m_g=-1,0,+1$，激发态有三个磁子能级 $m_e=-1,0,+1$。在理想偏振条件下，仅有以下两条耦合被保留：

1. $\sigma^+$ 光耦合 $\ket{g,-1}\leftrightarrow\ket{e,0}$
2. $\sigma^-$ 光耦合 $\ket{g,+1}\leftrightarrow\ket{e,0}$

因此，$\ket{g,-1}$ 与 $\ket{g,+1}$ 通过共同激发态 $\ket{e,0}$ 构成一个 Λ 型三能级系统，而 $\ket{g,0}$ 在该理想模型中不参与动力学。

+ 由 Clebsch-Gordan 系数平方给出自发辐射分支比：

| G\E | $\ket{e,0}$ |
| --- | ---: |
| $\ket{g,-1}$ | $1/2$ |
| $\ket{g,0}$ | $0$ |
| $\ket{g,+1}$ | $1/2$ |

在理想模型中，自发辐射只会把原子从 $\ket{e,0}$ 带回到 $\ket{g,\pm1}$ 两个态，而不会进入 $\ket{g,0}$。

+ 以及偶极矩阵元：

与上文类似，偶极耦合仍由 Clebsch-Gordan 系数给出。这里两条非零跃迁的系数大小相同，均为 $1/\sqrt{2}$。

定义

$$
\Omega_0\equiv\frac{dE_0}{\hbar}
$$

则非零耦合为

| 跃迁 | 偏振 | $C^{1,0}_{1,m_g;1,q}$ | 偶极矩阵元 |
| --- | ---: | ---: | ---: |
| $\ket{g,-1}\to\ket{e,0}$ | $\sigma^+$ | $1/\sqrt{2}$ | $\dfrac{\hbar\Omega_0}{\sqrt{2}}e^{i(kz-\omega t)}$ |
| $\ket{g,+1}\to\ket{e,0}$ | $\sigma^-$ | $1/\sqrt{2}$ | $\dfrac{\hbar\Omega_0}{\sqrt{2}}e^{i(-kz-\omega t)}$ |

在激光旋转参考系中（并采用 RWA）下，设失谐

$$
\Delta=\omega-\omega_{eg}
$$

则相互作用哈密顿量写为

$$
H_{\mathrm{rot}}(z)= -\hbar\Delta|e,0\rangle\langle e,0|
-\frac{\hbar\Omega_0}{\sqrt{2}}\Big[e^{ikz}|e,0\rangle\langle g,-1|
+e^{-ikz}|e,0\rangle\langle g,+1|\Big]+\mathrm{h.c.}
$$

若取基底顺序

$$
\{|g,-1\rangle,|g,+1\rangle,|e,0\rangle\}
$$

对于 MCWF 模拟，再加入自发辐射导致的非厄米项。若激发态总衰减率为 $\Gamma$，则有效哈密顿量为

$$
H_{\mathrm{eff}}(z)=H_{\mathrm{rot}}(z)-\frac{i\hbar\Gamma}{2}|e,0\rangle\langle e,0|
$$

在同一基底下，其矩阵形式为

$$
H_{\mathrm{eff}}(z)=\hbar
\begin{pmatrix}
0 & 0 & -\dfrac{\Omega_0}{\sqrt{2}}e^{-ikz} \\
0 & 0 & -\dfrac{\Omega_0}{\sqrt{2}}e^{ikz} \\
-\dfrac{\Omega_0}{\sqrt{2}}e^{ikz} & -\dfrac{\Omega_0}{\sqrt{2}}e^{-ikz} & -\Delta - i\Gamma/2
\end{pmatrix}
$$

再由动能给出对角项，且光场中的 $e^{\pm ikz}$ 在离散动量格点上对应最近邻平移，即可。

