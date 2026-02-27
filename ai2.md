把希尔伯特空间写成张量积最清楚：

\[
\mathcal H=\mathcal H_{\rm int}\otimes \mathcal H_{\rm ext},
\qquad
\mathcal H_{\rm int}=\mathrm{span}\{|g\rangle,|e\rangle\},
\qquad
\mathcal H_{\rm ext}=\mathrm{span}\{|p_m\rangle\}_{m=-n_{\max}}^{n_{\max}}.
\]

其中外态基矢满足
\[
p_m=p_{\rm center}+m\,k_L,\qquad \hat p|p_m\rangle=p_m|p_m\rangle.
\]

---

**1) 外态上的“动量算符”和动能**
\[
\hat p=\sum_m p_m\,|p_m\rangle\langle p_m|,
\qquad
\hat T=\frac{\hat p^2}{2M}
=\sum_m \frac{p_m^2}{2M}\,|p_m\rangle\langle p_m|.
\]
它只作用在外态空间，所以在总空间里是
\[
\hat H_{\rm kin}= \mathbb I_{\rm int}\otimes \hat T.
\]

---

**2) 内态上的失谐与非厄米衰减（有效哈密顿量）**
在你用的约定里，激发态能量相对项是 \(-\delta\)，衰减写进有效非厄米项 \(-i\Gamma/2\)：
\[
\hat H_{\rm int,eff}=\left(-\delta-\frac{i\Gamma}{2}\right)|e\rangle\langle e|.
\]
提升到总空间就是
\[
\hat H_{\rm det+dec}=\hat H_{\rm int,eff}\otimes \mathbb I_{\rm ext}.
\]

---

**3) 耦合项：内态翻转 ⊗ 外态动量步进**
在“动量梯”基底下，两束对向光导致动量改变 \(\pm k_L\)，对应外态的移位算符（在截断空间上，边界之外视为 0）：
\[
\hat S_{+}|p_m\rangle=
\begin{cases}
|p_{m+1}\rangle,& m<n_{\max}\\
0,& m=n_{\max}
\end{cases},
\qquad
\hat S_{-}|p_m\rangle=
\begin{cases}
|p_{m-1}\rangle,& m>-n_{\max}\\
0,& m=-n_{\max}
\end{cases}.
\]
把“最近邻求和”记成
\[
\hat S=\hat S_{+}+\hat S_{-}.
\]

内态翻转算符（只在两能级子空间）：
\[
\sigma_x = |g\rangle\langle e|+|e\rangle\langle g|.
\]

与你的离散方程一致的耦合项就是
\[
\hat H_{\rm coup}= \frac{\Omega_0}{2}\;\sigma_x\otimes \hat S.
\]

（如果你愿意也可以写成两项：\(\frac{\Omega_0}{2}\big(|g\rangle\langle e|\otimes \hat S + |e\rangle\langle g|\otimes \hat S\big)\)，同一件事。）

---

**4) 最终总有效哈密顿量**
把三部分加起来：
\[
\boxed{
\hat H_{\rm eff}
=
\mathbb I_{\rm int}\otimes \frac{\hat p^2}{2M}
+
\left(-\delta-\frac{i\Gamma}{2}\right)|e\rangle\langle e|\otimes \mathbb I_{\rm ext}
+
\frac{\Omega_0}{2}\,\sigma_x\otimes (\hat S_{+}+\hat S_{-})
}
\]

---

**5) 等价的块矩阵写法（对应代码里的实现形状）**
若把态写成列向量 \(\psi=(g,e)^T\)，其中 \(g,e\in\mathbb C^{N_p}\)，则
\[
H_{\rm eff}=
\begin{pmatrix}
T & \frac{\Omega_0}{2}S\\
\frac{\Omega_0}{2}S & T-\delta-\frac{i\Gamma}{2}
\end{pmatrix},
\]
这里 \(T=\mathrm{diag}(p_m^2/2M)\)，\(S\) 是只有上下副对角为 1 的稀疏矩阵（实现了 \(m\leftrightarrow m\pm1\) 的动量耦合）。

如果你接下来想把它和“位置表象里的 \(\cos(kx)\)”联系起来，我也可以把 \(\hat S_{+}+\hat S_{-}\) 如何对应到 \(\cos x\)（在合适无量纲下）推一遍。