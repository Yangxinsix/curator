# ACVR 项目报告：目标、实现状态与验证计划

> 结果更新（2026-09-08）：SPICE 队列 31/31 任务已完成，物理距离外推得到改善，但尚无严格容限下的摊销加速证据。复核还发现 `last.ckpt` 停留在最优轮、未保存实际终点的问题。见 [第二阶段结果核对](SPICE_STAGE2_RESULTS_20260908.md)。

> 下一阶段更新（2026-09-07 08:07 UTC）：SPICE 真实数据的三种模型、三个随机种子及后续 proposal／评估任务已启动为持久化 GPU 队列。完整结果仍在运行中，配置、范围与状态入口见 [SPICE 实验说明](spice_stage2.md)。

> 后续更新（2026-09-07）：本报告记录的训练接口问题已修复，学习型合成 oracle 已完成两阶段训练、checkpoint 重建／恢复及独立评估。见 [训练验证记录](TRAINING_VALIDATION_20260907.md) 和 [使用说明](small_training.md)。下文保留本次修复之前的核对记录。

> 核对日期：2026-09-07  
> 代码仓库：`curator`；当前分支：`acvr-unified-framework`；HEAD：`2da307c`  
> 定位：统一变分响应与训练摊销的工程参考实现，尚不是完成科学验证的大规模生产框架。  
> 本报告区分：本次复测结果、历史实验记录、代码已有能力、后续计划。测试通过不等于物理模型正确，也不等于已经获得加速。

## 阅读导航

- [1. 我们到底要完成什么](#1-我们到底要完成什么)
- [2. 框架统一什么，不统一什么](#2-框架统一什么不统一什么)
- [3. 已完成的代码与调用链](#3-已完成的代码与调用链)
- [4. 已做的测试和实验，实际证明了什么](#4-已做的测试和实验实际证明了什么)
- [5. 当前缺口与本次发现的问题](#5-当前缺口与本次发现的问题)
- [6. 训练、标签和边界应该如何配合](#6-训练标签和边界应该如何配合)
- [7. 下一步开发顺序与交付物](#7-下一步开发顺序与交付物)
- [8. 测试与实验设计](#8-测试与实验设计)
- [9. 本次复测命令与环境](#9-本次复测命令与环境)
- [10. 现有资料入口](#10-现有资料入口)

## 1. 我们到底要完成什么

### 1.1 最终目标

不是再增加一个 QEq 模块，而是为机器学习势提供可复用的流程：

```text
DFT / 响应标签 + 明确的计算条件
    → 学习内部变量的物理泛函
    → 编译用户声明的边界条件
    → 在训练中学习“结构和边界 → 平衡内部变量”的映射
    → 模拟时快速预测、投影、认证，必要时修正
    → 一致地输出能量、力、应力和响应性质
```

对于已经支持的物理插件，用户应主要准备数据、选择模型、填写边界配置，而不需要重新开发求解器、训练器和部署流程。

一个 checkpoint 应能在**模型适用范围、数据可识别范围和已实现边界范围内**切换条件。这里不包含“用任意普通 DFT 数据就能预测任意外场、反应或电荷转移”的承诺。

### 1.2 项目完成需要同时满足的四件事

| 目标 | 完成标准 | 当前判断 |
|---|---|---|
| 数学与软件统一 | 不同内部变量复用问题、边界、求解、导数和 loss 合同 | 二次参考核心已实现；尚缺第二个完整物理插件验证 |
| 训练摊销有效 | 未见结构/条件上，proposal 达到物理误差要求，减少实际求解工作 | 固定解析泛函的训练集演示已通过；泛化待验证 |
| 物理预测可信 | 与对应边界下的独立标签、响应和模拟性质一致 | 数据读取和解析测试已完成；真实学习型模型的科学验证未完成 |
| 用户直接可用 | 配置 → 训练 → checkpoint → 恢复 → 评估 → MD 全链路可复现 | 现有组件可复用，但端到端产品化未完成 |

### 1.3 当前优先级

先完成**后端无关的学习型泛函 + proposal 训练闭环**。电荷/Ewald 保留为第一个物理特例；第二变量用于检验抽象是否真正通用。Latent Ewald 暂缓，不作为当前验收依赖。

## 2. 框架统一什么，不统一什么

### 2.1 最小问题合同

固定一个原子构型 $\mathbf R$ 和一个边界条件后，当前主要计算对象是：

$$
\min_{\mathbf x}\;\Phi(\mathbf x)
=\frac12\mathbf x^\mathrm T H\mathbf x
+\mathbf g^\mathrm T\mathbf x+c,
\qquad A\mathbf x=\mathbf b.
$$

- $\mathbf x$：内部变量，可以是电荷，也可以是展开后的向量变量。
- $H,\mathbf g,c$：**边界编译后**的二次系数、一阶系数和常数项。
- $A$：每行一个等式约束；$\mathbf b$ 是对应目标。
- 稳定性要求是 $H$ 在允许变化的方向上正定；不必在被约束掉的全部方向上正定。

物理插件先产生基础泛函 $F_\theta(\mathbf R,\mathbf x)$。例如外部控制项是 $-\mathbf h^\mathrm T B\mathbf x$，编译器就把 $\mathbf g_0$ 变成 $\mathbf g_0-B^\mathrm T\mathbf h$。二次环境项还会修改 $H$ 和 $c$。因此代码里的 $\mathbf g$ 不是另一个独立物理理论，也没有丢失原来的控制项。

采用拉格朗日函数 $\mathcal L=\Phi+\boldsymbol\lambda^\mathrm T(A\mathbf x-\mathbf b)$ 后，求解与认证使用同一组方程：

$$
\begin{pmatrix}
H&A^\mathrm T\\
A&0
\end{pmatrix}
\begin{pmatrix}\mathbf x\\\boldsymbol\lambda\end{pmatrix}
=
\begin{pmatrix}-\mathbf g\\\mathbf b\end{pmatrix}.
$$

当前主要实现是等式约束 KKT。一般 KKT 还包括不等式、对偶可行性和互补条件；这些不能因为已有 `kkt.py` 就算已经支持。

### 2.2 边界能力的范围

| 条件 | 在共同数学对象中的位置 | 已有程度与物理限制 |
|---|---|---|
| 固定总量，如总电荷 | 一行 $A\mathbf x=\mathbf b$ | 已实现；总电荷守恒不意味着每个原子电荷固定 |
| 多片段固定总量 | $A$ 的多行分别选取各片段 | 已实现；片段约束限制了片段间净转移 |
| 固定共轭场，如电极电势 | $-\mathbf h^\mathrm T B\mathbf x$ | 通用编译和 charge recipes 已实现；电势零点、储库与静电环境必须明确 |
| 浮动导体、恒势/恒荷混合 | 不同组分别放入约束或控制项 | charge recipes 已实现；电解液如何极化、转移电子仍由物理模型决定 |
| 有限容量环境、Robin 型条件 | 关于选定线性量的二次环境能 | 代数编译已实现；真实电容、储库等含义需要对应插件和参数 |
| 诱导偶极、fixed-$E$/fixed-$D$ | 可能涉及线性耦合或二次环境项 | 有可复用代数组件，但尚无完整偶极插件及极化分支/电边界实现 |
| 不等式、固定模长、自旋流形等 | 不等式或非线性约束 | 主要仍待开发；已有 nonlinear oracle 不等于这些生产功能已完成 |

恒电势并不意味着电解液和两个不同偏压的电极上所有原子的同一个标量都相等。还应区分静电势、原子电荷能量导数和电子电化学势，不能仅凭一个共同符号混用它们。

### 2.3 Ewald 的位置

统一的是**变分、约束、响应、训练和认证流程**，不是所有物理系统的相互作用核。

电荷插件可用 $H=\operatorname{diag}(\boldsymbol\eta)+K_{\mathrm{Coulomb}}$；周期库仑算符可采用 Ewald/PPPM。其他变量需要自己的局域项、耦合算符、单位和对称性。偶极长程即使也采用 Ewald 分解，也不是直接复用标量电荷矩阵；弹性、自旋等更不能默认采用电荷 Ewald。

## 3. 已完成的代码与调用链

### 3.1 按阅读和调用顺序排列

以下链接指向当前实现，不代表每项都已具备生产性能。

| 层 | 主要入口 | 职责与已实现能力 |
|---|---|---|
| 问题定义 | [functional.py](../../curator/variational/functional.py)、[spec.py](../../curator/variational/spec.py) | `QuadraticProblem` 存储系数；变量布局、分组、等式、控制、环境和 gauge 规范 |
| 边界编译 | [compiler.py](../../curator/variational/compiler.py) | `compile_boundary`，编译 $A,\mathbf b,B,\mathbf h$ 等；`to_problem`/`to_operator_problem` 合入基础泛函 |
| 正确性检查 | [validation.py](../../curator/variational/validation.py)、[kkt.py](../../curator/variational/kkt.py) | 约束秩、一致性、对称性、允许方向的曲率；组装 KKT 并重新计算真实残差 |
| 精确参考求解 | [dense.py](../../curator/variational/backends/dense.py) | `solve_dense` 使用 `torch.linalg.solve`；另有 Schur 对照路径 |
| 投影和迭代 | [projection.py](../../curator/variational/projection.py)、[iterative.py](../../curator/variational/backends/iterative.py) | 度量投影；仅通过 `matvec` 作用 Hessian 的 projected-CG 参考实现 |
| 摊销推理 | [amortized.py](../../curator/variational/amortized.py) | proposal → 投影 → 残差认证 → KKT 修正 → exact fallback |
| 梯度和响应 | [implicit.py](../../curator/variational/implicit.py)、[response.py](../../curator/variational/response.py)、[forces.py](../../curator/variational/forces.py) | implicit backward、线性响应及三种导数模式；[nonlinear.py](../../curator/variational/nonlinear.py) 提供 dense Newton–KKT oracle |
| 摊销训练 | [amortized_training.py](../../curator/variational/amortized_training.py)、[amortized_variational.py](../../curator/train/amortized_variational.py) | exact teacher、缓存失效检查、修正前 proposal loss、geometry-balanced 聚合及普通 `run_step` 适配 |
| 多条件数据和标签 loss | [variational_data.py](../../curator/data/variational_data.py)、[variational_losses.py](../../curator/train/variational_losses.py) | 一条 geometry 多个 conditions、ragged 数据、缺失标签 mask、单位/gauge、物理标签与响应联合训练 |
| 电荷特例 | [charge.py](../../curator/variational/variables/charge.py)、[charge_boundary.py](../../curator/variational/variables/charge_boundary.py)、[charge_runtime.py](../../curator/variational/variables/charge_runtime.py) | learned onsite 参数接口、多边界 charge 求解、完整势的力/应力参考路径 |
| 长程后端特例 | [ewald.py](../../curator/variational/backends/ewald.py)、[compute_acvr_kspace.cpp](../../interface/compute_acvr_kspace.cpp) | Torch 电荷核参考；LAMMPS point-charge $Kq$ 与能量对照接口 |

### 3.2 四条核心调用链

**边界与精确解：**

```text
VariableLayout + BoundarySpec
  → compile_boundary(...)
  → CompiledBoundary.to_problem(H0, g0, c0)
  → solve_dense(problem)
      → validate_problem
      → 组装 KKT → torch.linalg.solve
      → evaluate_kkt
  → state / multipliers / energy / residuals / diagnostics
```

**近似求解：**

```text
物理插件的 operator.matvec + 编译后的边界
  → OperatorQuadraticProblem
  → solve_projected_cg_problem
  → 投影到可行域，在允许变化的方向上迭代
  → 重新计算真实残差，并单独报告稳定性是否有依据
```

这是数值迭代，不是机器学习摊销。CG 收敛也不能自动证明不存在未被迭代激发的负曲率方向。

**摊销推理：**

```text
插件预测 proposal
  → solve_amortized
      → validate_problem
      → project_to_equalities
      → 估计 multiplier → evaluate_kkt
      → accepted_projected
        或 solve_linear_response 做 KKT correction → corrected
        或 solve_dense → fallback
        或明确报告 failed
```

**已有摊销训练：**

```text
固定/当前的 QuadraticProblem + proposal 网络输出
  → build_amortized_teacher_batch（可选缓存，泛函变化后必须重建）
  → prepare_amortized_training_batch
      → 投影 proposal
      → 计算修正前 KKT residual
      → 提供 detached exact teacher
  → AmortizedVariationalLoss
  → AmortizedTrainingModelOutput
  → Curator run_step → backward → optimizer.step
```

最后一条才是把“学会求解”转移到训练的过程。仅调用 `solve_amortized`，不会自动训练任何网络。

### 3.3 分支和兼容性

上述 ACVR 开发已经集中在 `acvr-unified-framework`，不是继续直接向 `agent-curator` 写 ACVR 提交。自基线 `31a1eee` 起，依次有核心、边界/修正、导数、多条件数据、联合 loss、charge runtime、LAMMPS 算符、通用摊销训练共八个提交。

本次工作区并非干净状态：还有已有的 trainer/metrics、相关测试、文档修改和未跟踪的 `evaluate/`。因此本报告复测对象是 **HEAD 加当前工作区**，不是单独检出 HEAD 的洁净复现。报告不接管这些修改，也未实施代码修复、提交或推送。

新 ACVR charge runtime 与旧 `ChargeEquilibration` 保持语义隔离。旧路径的 `ewald_forces` 不能当成完整 learned QEq 势的全部力；不能因为新实现完整，就静默声称旧 checkpoint 的力语义也已改变。

## 4. 已做的测试和实验，实际证明了什么

### 4.1 本次重新执行的定向回归

2026-09-07，启用本地 SPICE、PySCF、TRECI 数据和 ACVR-enabled LAMMPS 后：

```text
163 passed, 68 warnings in 20.51s
```

无跳过项。范围包括 ACVR 解析求解、边界、修正、迭代、隐式导数、力、非线性参考、多条件数据/loss、charge runtime、摊销训练，以及原 QEq/Ewald/SPICE 相关回归。完整命令见第 9 节。

68 条 warning 来自 ASE/NumPy 接口迁移提示和现有 TorchScript 类型提示；不是本次新增的测试失败。此结果**不是全仓库测试结果，也未覆盖完整 CLI 训练生命周期**。

首次未指定外部资源时为 `155 passed, 8 skipped`；补齐资源后的 163 项是同一套测试的完整复跑，两者不能相加。

### 4.2 通用摊销训练：已经有真实优化步骤，但证明范围很窄

[test_acvr_amortized_training.py](../../test/test_acvr_amortized_training.py) 中的主实验：

| 设置 | 实际内容 |
|---|---|
| 内部变量 | 四个数，可解释为两个实体、每个实体两个分量；没有 charge/Ewald |
| 物理泛函 | 固定 $4\times4$ 正定 $H$；$\mathbf g$、$\mathbf b$ 按预设公式依赖条件；不是学习型泛函 |
| 数据 | 12 个三维条件向量；geometry 索引用于测试聚合，不是真实原子几何数据集 |
| proposal | `nn.Linear(3, 4)` |
| 优化 | Adam，学习率 0.07，100 次普通 `run_step` |
| 训练项 | state teacher loss + 修正前 KKT loss |
| 通过的误差断言 | projected-state MSE 小于 $3\times10^{-6}$，且小于初始值的 $2\times10^{-5}$ |
| 推理检查 | 在这 12 个**训练条件**上，归一化驻定残差容限设为 $6\times10^{-3}$；全部 `accepted_projected`，零修正、零 fallback |

另外三项测试验证：proposal loss 默认不更新物理问题系数；修正前残差确实向 proposal 传梯度；缓存 teacher 与当前系数不一致时拒绝使用。

这些结果证明训练摊销的代码合同可运行。没有证明：学习 $H_\theta,\mathbf g_\theta$、未见几何泛化、未见边界泛化、默认 $10^{-8}$ 容限下零修正、真实 DFT 精度或运行时间加速。这个固定二次问题族的解映射本身也很简单，不能代替复杂体系实验。

### 4.3 数据准备和真实数据读取

以下规模来自已有数据审计；本次重新读取了真实文件，并核对了 SPICE、PySCF 文件和 TRECI split 的 SHA-256 与记录一致。

| 数据 | 已有成果 | 能用于什么，不能用于什么 |
|---|---|---|
| SPICE Ion Pairs | 28 组、1428 个双原子构型；距离划分 868/280/280；E/F 和偶极标签，当前小文件没有 MBIS | 可做分离曲线和截断外长程测试；不是恒电势电极数据 |
| 周期 PySCF H₂ | 3 个键长，每个 $Q=-1,0,+1$，共 9 个电子态；保存计算设置、能量和力 | 可检验多条件读取和小型 DFT 对照；不是充分训练集，也不是金属电极模型 |
| TRECI Cu/水 | 236 个公共几何，各有 PZC 加 7 个数值电势条件；按几何分成 188/24/24 | 可开展配对条件 E/F 评估；仅 15 个几何具有全部条件的完整电子数和 Bader 标签 |

TRECI 的 `free_energy` 到目标热力学势的映射仍须审计；Bader 分区电荷不自动等于电极自由电荷；PZC 也不是第八个固定数值电压。当前报告不能据此宣称恒电势 ACVR 已经训练成功。

### 4.4 LAMMPS：已有什么对照

- **历史官方示例实验：** 3776 原子的 `graph-il`，conp/conq 各运行 500 步。conp 固定底/顶电势为 −1/+1 V，电荷随结构变化；conq 固定底/顶电荷为 −1/+1 e，电势差从约 20.029 V 变为 8.767 V。它说明两种控制方式的含义，不是我们的 MLIP 实验。本次未重新运行这两条 500 步轨迹。
- **本次重跑的自定义算符测试：** `compute acvr/kspace` 在 Ewald/electrode、PPPM/electrode 上检查 $\tfrac12\mathbf q^\mathrm T K\mathbf q=E_{\mathrm{coul}}+E_{\mathrm{long}}$ 和 $K\mathbf q=\partial E/\partial\mathbf q$；包括 Torch 核对照与不支持接口时的拒绝行为。
- **仍未完成：** 完整 `fix curator/acvr`、共享 timestep 缓存、MLIP 势与求解器的 production 耦合、多 MPI/GPU 验证和长轨迹加速。

这个 compute 是算符接口，不是恒电势求解器。它要求具有相应接口的 KSpace 和暴露 `cut_coul` 的兼容 pair；并非理论上只能使用一个特定名称的 `coul/long` pair。

## 5. 当前缺口与本次发现的问题

### 5.1 一个已经复现的训练生命周期接口问题

当前 [lit_module.py](../../curator/model/lit_module.py) 的 epoch 收尾调用：

```python
output.reset_metrics(subset=subset)
```

而 `AmortizedTrainingModelOutput.reset_metrics` 当前不接受 `subset`。本次直接调用复现：

```text
TypeError: AmortizedTrainingModelOutput.reset_metrics() got an unexpected keyword argument 'subset'
```

因此，把这个输出类直接放进当前 Lightning epoch 生命周期会遇到接口错误；单元测试里的 `run_step` 不覆盖这一点。应先增加失败回归，再统一接口以及 train/val/test 指标隔离语义。本报告只记录问题，没有修改训练代码。

### 5.2 “零修正”目前不能等同于“省掉昂贵求解”

`solve_amortized` 的入口先调用 `validate_problem`。当前检查包含约束 SVD、显式构造允许变化的子空间、reduced-Hessian 特征值等；随自由度增长，这些稠密诊断本身可能具有三次复杂度。残差还需要作用 Hessian；如果使用一般稠密投影 metric，投影也可能昂贵。

因此，即便结果是 `accepted_projected`，也不能忽略入口诊断后宣称获得加速。当前一次 dense KKT correction 对二次问题基本等价于再解一个完整线性系统，也不是天然廉价修正。

生产路径需要在保留可靠性依据的前提下使用结构性稳定保证、有效期明确的缓存、小约束投影、matrix-free 认证和迭代修正。不能简单删除检查或无依据地设置 `reduced_stability_certified=True`。

### 5.3 其余关键缺口

1. 没有完成从结构学习 $H_\theta,\mathbf g_\theta$，再联合管理 proposal 的完整训练示例。
2. 没有证明同一 checkpoint 在未见结构、未见控制数值或不同边界类型上的预测能力。
3. 没有完整第二物理变量插件，尚不能用“有四维 toy”代替物理通用性验证。
4. 没有大规模 matrix-free 摊销训练/推理闭环；当前 charge Ewald 核仍是稠密参考路径。
5. 没有完整模型在真实 DFT 数据上的精度、响应和模拟性质结果。
6. 没有新 ACVR 模型长时间 MD 的误差—稳定性—速度证据。
7. 有限残差下，envelope 力不自动是已报告近似能量的严格梯度；当前 detached adjoint 修正只支持一阶导数，不能用于冒充二阶响应支持。

## 6. 训练、标签和边界应该如何配合

本节是后续实施合同，不是已经完成的训练结果。

### 6.1 两个学习任务必须分开

**任务 A：学习物理泛函。**

模型从结构产生 $F_\theta$ 的系数或算符；按该条标签对应的条件求精确/隐式解；用能量、力和选定响应等标签训练 $\theta$。

$$
\mathcal L_{\mathrm{phys}}
=w_E\ell_E+w_F\ell_F+w_\sigma\ell_\sigma
+w_O\ell_O+w_{\mathrm{resp}}\ell_{\mathrm{resp}}.
$$

其中 $O$ 是插件对应的可观测量，如偶极或明确语义的群总量。每项使用明确单位、尺度和缺失标签 mask；并非每个数据集都必须提供全部项。固定电荷能量与开电子储库的热力学势必须按正确约定比较。

**任务 B：学习快速求解映射。**

$$
\widehat{\mathbf x}_\phi
=f_\phi(\mathbf R,\text{boundary}),
\qquad
\mathbf x_0=\Pi_{A\mathbf x=\mathbf b}(\widehat{\mathbf x}_\phi).
$$

用当前物理泛函的 exact 解作 teacher，训练 projected-state 误差与**修正前** KKT 残差。现有 loss 还支持可选 energy-gap 和 multiplier 项。

物理步骤更新 $\theta$；proposal 步骤冻结物理问题和 teacher，更新 $\phi$。若共享 backbone，还要审计共享参数和 feature 上的梯度路径：仅 detach 一份 $H$ 张量，并不保证共享 backbone 不会通过 proposal 分支改变物理模型。

teacher 是模型自身求解所得，不要求 DFT 给出每个内部变量的“真值”。但蒸馏出精确 teacher 只代表解对了当前模型，不代表当前模型就是正确物理。

### 6.2 不要求为每个推理边界都重新做 DFT，但不能省略可识别性

可以从普通电子基态 DFT 的 E/F 数据起步；核构型不必都处于几何平衡态。但每次 DFT 本来就有电子数、周期/表面处理、外场等条件，数据不能丢掉这些元信息。

对固定模型，数学上能在任意已支持边界下求解；物理上是否可信，还取决于训练数据是否识别了这些方向上的响应。

一个直观反例是：

$$
F(\mathbf R,\mathbf x)
=E_0(\mathbf R)
+\frac12[\mathbf x-\mathbf a(\mathbf R)]^\mathrm T
D(\mathbf R)[\mathbf x-\mathbf a(\mathbf R)].
$$

对于任意正定 $D$，无约束极小值都是 $E_0(\mathbf R)$，对应的结构力也相同；但加入 $-\mathbf h^\mathrm T\mathbf x$ 后，响应是 $D^{-1}$。所以零场 relaxed E/F 可以完全相同，外场响应却不同。

这说明：**没有必要默认要求每个电压都有 DFT 标签；但也不能默认单一条件 E/F 足以学习任意响应。** 可通过少量多条件数据、有限场/DFPT 响应、可观测量标签或有依据的物理先验补足，并用留出的独立条件验证。响应标签也只约束被探测的方向，不一定确定整个内部变量 Hessian。

### 6.3 $H$ 和一阶系数怎么获得

- **解析测试：** 人工构造已知的 $H,\mathbf g$，用于检验算法。
- **学习型二次插件：** 网络输出有约束的系数或算符参数，由物理 loss 间接学习；不要求 DFT 直接提供原子电荷 Hessian。
- **特定物理插件：** 神经网络学习局域项，已知相互作用构造非局域算符；电荷/Ewald 是一个例子。
- **非线性泛函：** 局部线性项和 Hessian 可由自动微分获得，但二次近似的有效范围、稳定分支和求解成本要另行验证。

任意光滑内部变量都可以在合适参考态附近做 Taylor 展开；这不意味着二次截断在大外场、相变、电子转移或多稳态过程中仍然充分。也要区分对内部变量 $\mathbf x$ 的 Hessian 与对核坐标 $\mathbf R$ 的 Hessian。

## 7. 下一步开发顺序与交付物

以下是建议顺序；本次未开始实施这些代码或启动新训练/DFT 任务。

### P0：打通正式训练生命周期

修复第 5.1 节接口，补齐 ACVR outputs 的分阶段指标、epoch 收尾、checkpoint、resume 和 evaluator 集成测试。交付一个可执行配置，在真实训练入口运行至少两个 epoch，保存、加载、恢复，并复现评估输出。

这一步复用现有 trainer，不另写一个 ACVR 专用 trainer。还要保留当前工作区其他训练修改，避免把接入问题当成重写训练基础设施的理由。

### P1：完成学习型二次泛函与 proposal 闭环

新增不依赖电荷/Ewald 的 oracle plugin：结构特征产生 $c_\theta,H_\theta,\mathbf g_\theta$，边界由现有 compiler 处理，独立 proposal 接收结构和边界特征。

先选可验证的正定构造，例如有限维 toy 的 $H_\theta=L_\theta L_\theta^\mathrm T+\epsilon I$，并明确这只是测试模型，不是所有物理插件必须全空间正定的要求。

交付内容：可重建的合成数据、按结构划分的 split、physical/proposal 两阶段及交替训练配置、teacher 失效策略、梯度隔离测试、checkpoint 元信息和评估脚本。明确报告学会的是 $F_\theta$ 还是仅 proposal，不能把两者混在一个 loss 曲线里。

### P2：证明摊销的泛化和误差控制

使用同一个训练好的泛函，对比 exact、无训练 proposal、训练 proposal、proposal+认证修正、带前一步 warm start 的迭代求解。区分未见结构、控制值插值、控制值外推和未见边界类型；最后一种是单独探索项，不保证自动成功。

通过标准是：未见样本上达到预设物理误差，且确实减少 matvec/求解工作；不能只展示训练 loss 降低或大量 fallback 后的准确结果。

### P3：接入电荷特例与第二个物理插件

电荷先复用现有 runtime，用 SPICE 和小型多电子数数据做分层验证。TRECI 在热力学语义审计通过后再承担跨电势科学验证。

第二变量选诱导偶极：先验证有限系统中自己的极化率、偶极相互作用张量和外场耦合，再考虑周期 fixed-$E$/fixed-$D$。不得把电荷 Ewald 名字换成 dipole 就算完成插件。

验收时统计新增代码位置：共享 trainer/loss/solver 应能复用；新物理应主要出现在变量、算符、对称性和标签适配层。若必须修改核心合同，应明确记录抽象的不足，而不是复制另一套框架。

### P4：实现可扩展推理和 LAMMPS 生产闭环

完善 matrix-free proposal 认证/修正、结构性稳定保证、预条件器和缓存；再连接 `fix curator/acvr`、局域模型和 KSpace。明确总能量只计一次长程项，统一力、应力、电势参考和自能/背景项。

通过 Python/CPU、LAMMPS 单 rank、MPI、GPU 的一致性与端到端性能测试后，再进行长轨迹模拟。保持 dense oracle 作为小体系独立对照，不用于代表大体系性能。

## 8. 测试与实验设计

### 8.1 必须补齐的自动测试

| 类别 | 测试内容 | 应阻止的错误 |
|---|---|---|
| 数学/边界 | 已知解；固定量/共轭场/混合/环境；冗余、不相容、gauge、单位错误 | 解错问题，却得到很小的残差 |
| 稳定性/尺度 | reduced curvature；病态谱；不同单位尺度；未激发负模；非有限数 | 把驻点或缩放后的假小残差当可靠极小值 |
| 梯度/力 | 参数 backward；核坐标和 cell 有限差分；结构相关约束；响应对称性 | 漏约束乘子项、漏局域系数导数、错用 envelope |
| 训练隔离 | physical/proposal 梯度路由；共享 backbone；缓存 teacher 过期；修正前 residual | 泛函向坏 proposal 迁就、或 teacher loss 没有梯度 |
| 数据/指标 | 缺失与零标签；geometry/condition 权重；split 无泄漏；train/val/test 指标独立 | 同一几何泄漏、标签缺失误作零、指标跨阶段污染 |
| 生命周期 | 正式训练入口、epoch hook、保存/恢复、评估、单位/gauge/变量布局持久化 | 单步可运行但训练一轮就失败，或加载后模型语义改变 |
| 后端/性能 | 同态 $Kx$、E/F/virial；所有诊断和通信计时；MPI/GPU 一致性 | 算符与能量不一致、重复计能、只计网络前向的虚假加速 |

解析测试优先使用 float64 和多个尺度；有限差分使用步长扫描，避免偶然选到一个“看起来吻合”的步长。不得将坏数据或不支持的物理条件静默回退成别的模型。

### 8.2 分层实验矩阵

以下规模和门槛是**拟定起点，不是已取得的结果**。每层先做小试验，失败时定位原因，不直接放大数据和算力。

| 实验 | 设计与对照 | 主要输出/进入下一层的条件 |
|---|---|---|
| E0：生命周期 | 一个小 oracle、至少 2 个 epoch，保存后恢复；覆盖 P0 接口 | 训练/验证/测试完整完成，状态与指标可复现 |
| E1：学习型通用 oracle | 4/8 维、明确的结构特征、64/16/16 个结构，多控制条件、3 个种子；对照已知泛函 | 物理拟合与 proposal 两条曲线；held-out state/energy/response；梯度隔离正确 |
| E2：标签可识别性 | 同一 oracle 比较 E/F-only、增加可观测量、增加少量响应/多条件标签；可加入受控先验 | 哪些标签识别哪些响应方向；不能只报告训练能量一样好 |
| E3：摊销泛化 | 冻结物理泛函，按结构和条件评估；扫容限 $10^{-2},10^{-4},10^{-6},10^{-8}$；对比 exact 和 warm-start CG | 精度—接受率—修正量—总耗时曲线；区分训练范围内外 |
| E4：真实电荷数据 | SPICE 距离外推；PySCF 小型语义对照；完成审计后使用 TRECI | E/F、分离曲线、偶极或可解释的群量/响应；不把分区电荷当唯一真值 |
| E5：第二变量 | 已知诱导偶极 oracle，再接有限场/响应标签；复用 E1/E3 流程 | 向量旋转/置换对称性、极化率、外场响应；不修改成专用 trainer |
| E6：部署和模拟 | operator parity → 小体系短 MD → 体系规模/硬件扫描 → 长轨迹 | 相同精度下的端到端收益、尾部延迟、轨迹稳定性与物理性质 |

E0/E1 首轮建议限定在 CPU 小任务预算内，例如每个配置不超过一小时；这是试验预算上限，不是耗时预测。先通过数据可识别性和训练生命周期，再决定新增 DFT 的条件数量与计算预算。

建议在 E1 的无量纲、适度条件数 toy 上先要求 held-out projected-state 相对 RMS 误差不高于 $10^{-3}$、选定响应相对误差不高于 $1\%$。零附近改用预先固定的绝对尺度，避免相对误差失真。实际物理体系的 eV/Å、偶极、电容等阈值须依据标签噪声和模拟用途另定，不能照搬 toy 门槛。

### 8.3 每次实验必须报告的指标

- **模型物理误差：** exact 解上的 E/F/应力/响应误差，隔离“泛函学错”和“proposal 没解准”。
- **proposal 误差：** 修正前 projected-state 误差、驻定残差、可行性残差；报告中位数、尾部和最坏情况。
- **推理策略：** 零修正接受率、平均及尾部修正次数、fallback 率、失败原因；标明对应容限。
- **成本：** 特征计算、物理系数/算符构造、proposal、投影、认证、修正、通信的分项与总耗时；matvec 数、内存、训练 teacher 成本和摊销收支平衡点。
- **泛化分组：** 未见结构、条件插值、条件外推、不同约束分组/边界类型分别报告；同一几何的条件不能随机拆开后冒充未见几何测试。

对未见控制值，训练 DFT 不一定要覆盖每个值；但是否使用训练几何上的合成 teacher 条件必须记录。测试几何不得提前用于 proposal teacher 训练。

### 8.4 MD 验收与停止条件

固定控制参数、静态环境下，先检验与所选有效势一致的保守力和短轨迹漂移；时变电压或外场情况下记录外部做功，不能机械要求系统内部能量恒定。

逐步比较 tighter exact 求解和摊销推理的结构分布、相关响应及时间相关性质。不能用一段“没有 NaN”的轨迹替代正确性和统计误差分析。

出现下列情况应暂停扩大实验：

1. exact 模型本身不能预测目标响应：先查物理模型和标签，不继续调 proposal 掩盖问题。
2. 标签定义或电势参考未明确：不进行跨数据源热力学比较。
3. 大部分样本靠 full fallback 才准确：不宣称训练摊销有效。
4. 包含全部认证成本后不比合理基线快：报告精度/鲁棒性收益，不报告尚不存在的加速。
5. 提高精度后 MD 结果仍不一致：先检查力的计算路径、边界功和模型适用范围。

## 9. 本次复测命令与环境

工作目录：`/workspace/work/agent-curator/curator`。Python 3.11.14，PyTorch `2.9.1+cu128`；本次并未进行 GPU/MPI 性能实验。使用已有隔离 LAMMPS 二进制，没有重建或改动生产安装。

```bash
CURATOR_LAMMPS_ACVR_BIN=/tmp/acvr-kspace-build-20260817/lmp \
CURATOR_SPICE_ION_PAIRS=/workspace/data/datasets/spice/2.0.1/ion_pairs/SPICE.hdf5 \
CURATOR_ACVR_PYSCF_STATES=/workspace/data/datasets/acvr/pyscf_pbc_h2_multicharge/v0.1/states.h5 \
CURATOR_TRECI_DATASET=/workspace/data/datasets/acvr/treci/24a0fe816e8aa93b6b604df9aa354806a9867cf8/Constant_potential/Training/Dataset \
../.venv/bin/python -m pytest -q -ra \
  test/test_acvr_*.py \
  test/test_variational_boundary_compiler.py \
  test/test_charge_equilibration.py \
  test/test_ewald_theory.py \
  test/test_ewald_nacl.py \
  test/test_qeq_task_growth.py \
  test/test_spice_dataset.py
```

外部数据和 `/tmp` 二进制不是随仓库分发的资源。在其他机器上应按对应文档恢复并核对 hash，再替换路径；不能让测试跳过后仍声称验证了真实数据或 LAMMPS。

本次核对的 SHA-256：

```text
SPICE.hdf5
631137e6ee16f6974721a5d6653b9f730fb57bd3dbea7e129e8c8a9f95e51ebd

PySCF states.h5
ed0cdc682be2b46870a743ad27e5bfd65672951f1615008a0676854655f60e30

TRECI geometry_split.json
8a06c7fd2615a8a6e809c375e094829ed54b7aa84a237aa327c16a31ce93e7b6
```

第 5.1 节接口问题的独立复现命令如下；它当前应报 TypeError，**不属于上述 163 个已通过测试**：

```bash
../.venv/bin/python -c 'from curator.train import AmortizedTrainingModelOutput, AmortizedVariationalLoss; output = AmortizedTrainingModelOutput(AmortizedVariationalLoss()); output.reset_metrics(subset="train")'
```

## 10. 现有资料入口

- 总路线图与历史进度：[ACVR_NATURE_COMPUTATIONAL_SCIENCE_ROADMAP.md](../../../ACVR_NATURE_COMPUTATIONAL_SCIENCE_ROADMAP.md)。其中历史“完成”应按当时的任务范围解释，以本报告补充的限制为准。
- 数学约定：[ADR 0001](../adr/0001-acvr-mathematical-conventions.md)；单位与精度：[ADR 0002](../adr/0002-acvr-units-and-precision.md)。
- 边界与求解：[charge_boundaries.md](charge_boundaries.md)、[solver_modes.md](solver_modes.md)。
- 导数与电荷 runtime：[differentiation.md](differentiation.md)、[charge_runtime.md](charge_runtime.md)。
- 物理标签联合训练：[training.md](training.md)；后端无关摊销训练：[amortized_training.md](amortized_training.md)。后者的接受率描述不替代本报告要求的端到端性能测量。
- 数据来源、转换与限制：[datasets.md](datasets.md)。
- LAMMPS 历史示例与算符测试：[lammps_electrode_oracle.md](lammps_electrode_oracle.md)、[lammps_kspace_operator.md](lammps_kspace_operator.md)。

接下来的直接交付物应是 P0/P1：**一个经过正式训练生命周期验证、能够学习泛函并训练 proposal、在留出结构和条件上出具独立评估报告的通用 oracle 示例。** 这是当前从“组件已实现”走向“统一框架可用”的主要缺口。
