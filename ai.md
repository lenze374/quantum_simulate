我先快速扫一眼当前工作区里的 lib.py 是否已经包含“动量表象 MCWF”的重构版接口（比如 `build_momentum_grid / build_heff_momentum_sparse`），然后基于你现有的实现把三件事讲清楚：初态、动量网格、哈密顿量（矩阵结构与物理含义）。我不会直接改代码，只做解释对照。

Read [](file:///Users/lenze/code/quantum_simulate/lib.py#1-1), lines 1 to 260

Read [](file:///Users/lenze/code/quantum_simulate/lib.py#260-260), lines 260 to 401

Read [](file:///Users/lenze/code/quantum_simulate/lib.py#401-401), lines 401 to 401

Read [](file:///Users/lenze/code/quantum_simulate/readme.md)

要用你“重构后的工具”（lib.py 里的 `Quantumket` + `CNstepper/RKstepper` + 自己写一个 `Operator.apply`）复现这个 1D 动量表象的 MCWF，多半需要先把“这个模型到底在什么基底上算”讲清楚：它不是 FFT 意义下的连续动量采样，而是“动量梯（recoil ladder）”的离散基底。

**1) 初态应该长什么样？（动量表象、两内态）**  
把态写成
\[
|\psi\rangle=\sum_{m=-n_{\max}}^{n_{\max}}\Big(g_m |g,p_m\rangle + e_m |e,p_m\rangle\Big),
\quad \psi=\begin{pmatrix}g\\ e\end{pmatrix},\ g,e\in\mathbb{C}^{N_p}
\]
其中 \(N_p=2n_{\max}+1\)。

在你 main_momentum.py 的设定里：
- 初始速度 \(v_0\)，初始动量 \(p_0=M v_0\)。
- 最常用的“最干净”初态是**纯基态 + 单一动量本征态**（离散梯上的一个格点）：
  \[
  g_m=\delta_{m,0},\quad e_m=0
  \]
  也就是“全部在 \(|g,p_0\rangle\)”上。因为你把网格中心就放在 \(p_0\)，所以 \(m=0\) 那个点就是 \(p_0\)。

如果你不想用 δ-like，而想更物理一些（一个动量波包），那就用高斯包在 \(p_0\) 附近：
\[
g_m \propto \exp\!\Big(-\frac{(p_m-p_0)^2}{4\sigma_p^2}\Big),\quad e_m=0
\]
最后按离散和式归一化：\(\sum_m (|g_m|^2+|e_m|^2)=1\)。

**关键提醒**：你在 main.py 里写的 `psi0 = np.zeros((2, n_max))` 更像是“还没写完”；做动量梯时外部维度通常是 \(N_p=2n_{\max}+1\)，而不是 \(n_{\max}\)。

---

**2) 动量网格应该是什么样？（recoil ladder，不是 FFT 动量轴）**  
这里的动量网格是
\[
p_m = p_{\text{center}} + m\,k_L,\quad m=-n_{\max},\ldots,n_{\max}
\]
- 在你的无量纲约定里常取 \(k_L=1\)，所以步长就是 1。
- 你在 main_momentum.py 用的是 `p_center=p0`，这样初态就天然落在中心格点上（\(m=0\)）。
- 选择 \(n_{\max}\) 的原则：**大到边界从来“不被碰到”**。最实用的判断是演化时监控边缘几格的总概率是否始终极小（比如 \(\ll 10^{-8}\)）。

一个估算思路（帮助你直觉选多宽）：
- 每次“量子跳跃”（自发辐射）会给动量一个 \(\pm k_L\) 的随机步进（你 old 版本里就是 shift）。
- 若总跳跃次数典型量级 \(N_{\text{jump}}\)，则随机游走展宽 \(\sim \sqrt{N_{\text{jump}}}\,k_L\)。
- 而 \(N_{\text{jump}}\approx \int_0^T \Gamma\,P_e(t)\,dt\)（粗略：激发态占据越大，跳得越多）。

所以经验上会取 \(n_{\max}\) 至少覆盖“几倍 \(\sqrt{N_{\text{jump}}}\)”的范围，再留余量。

---

**3) 哈密顿量（有效非厄米）到底是什么样？**  
这个模拟的核心是两能级原子在两束对向光场下的有效哈密顿量（旋转波近似 + 把自发辐射写进非厄米项）：

把系数向量按动量格点排成 \(g_m,e_m\)，有效薛定谔方程（\(\hbar=1\)）是
\[
i\frac{d}{dt}g_m = \frac{p_m^2}{2M}\,g_m + \frac{\Omega_0}{2}(e_{m-1}+e_{m+1})
\]
\[
i\frac{d}{dt}e_m = \Big(\frac{p_m^2}{2M}-\delta-\frac{i\Gamma}{2}\Big)e_m + \frac{\Omega_0}{2}(g_{m-1}+g_{m+1})
\]

写成矩阵块形式（这就是你 old 版本 old_lib.py 里 `build_heff_momentum_sparse` / `_heff_apply_momentum` 的结构）：
\[
H_{\rm eff}=
\begin{pmatrix}
T & \frac{\Omega_0}{2}S\\
\frac{\Omega_0}{2}S & T-\delta-\frac{i\Gamma}{2}
\end{pmatrix}
\]
- \(T=\mathrm{diag}\big(\frac{p_m^2}{2M}\big)\) 是动能（对 \(g/e\) 两个流形都一样）。
- \(S\) 是“最近邻求和”的稀疏矩阵：只有上下副对角是 1（对应 \(m\leftrightarrow m\pm1\) 的动量跃迁）。
- \(-i\Gamma/2\) 只加在激发态块上，表示非厄米损失（用于 MCWF 的“无跳跃”演化）。

---

**4) MCWF 的“跳跃”在这个基底里长什么样？（帮助你把全流程串起来）**  
一步“无跳跃”演化用 \(H_{\rm eff}\)（可用 CN 或 RK4）。演化后范数变小：
\[
\|\psi\|^2 < 1,\quad p_{\text{jump}} = 1-\|\psi\|^2
\]
然后按概率决定是否跳跃。若跳跃，塌缩算符在 1D 简化成“把激发态幅度搬运到基态并加 recoil shift”：
- 选 \(\pm\)（1D 下常取等概率）
- 做映射（以一步 recoil 为例）：
  \[
  g'_{m\pm1} \leftarrow e_m,\quad e'\leftarrow 0
  \]
- 再归一化。

这就解释了为什么你的哈密顿量耦合是最近邻、而跳跃也是最近邻 shift：两者都来自 \(\pm k_L\) 的动量改变。

---

如果你愿意，我可以按“用 lib.py 的 `Quantumket` + `CNstepper`”的视角，再把上面这套东西翻译成一个清晰的 `Operator.apply` 需要实现的三项（动能、耦合、非厄米衰减）分别怎么作用在 `ket.data[0]`/`ket.data[1]` 上；仍然只讲思路，不改代码。