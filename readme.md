# 简介

这是一个用于个人学习的冷原子模拟工具库，主要基于numpy和scipy的数据结构和工具进行开发。实践中主要使用MCWF（）方法模拟冷原子实验。

## 端口及规范（也是计划书）

### 波函数

psi.shape = (n_internal_states, n_external_states, ……)
空间：内态 $\otimes$ 外态
内态：$\ket{g}$, $\ket{e}$, 不同的F等
外态：位置空间 $\ket{x}$, 动量空间 $\ket{p}$
需要定义一个类实现

### 算符

用不同函数，但统一接口，返回psi

### 求解器

RK4 or CN

### Hamitanian

定义一个类，分开time-independent和time-dependent的部分，方便后续扩展，同时避免重复运算造成的性能开支。

### 绘图工具

需要一个统一的（物理量-time）的接口，方便统一绘图工具。

## 单位约定

可能需要一些方式来约束，以免出现问题

## 保存文件名称的约定
