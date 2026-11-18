# ACVR proposal 实现与运行入口

本轮实现了可选择的局域消息传递、全局 attention 和标量电荷 Ewald 通信，保留原 MLP。七项正式固定泛函对照均完成 600 步，七项联合 pilot 及匹配评价均完成：周期主对照为 20 个完整 epochs / 640 steps，QM/MM 为预算内最大共同的 8 个完整 epochs / 880 steps。所有候选均未通过固定泛函完整晋级条件，未触发多 seed 扩大训练；结果、终点差异与范围见 [实验报告](/workspace/data/work/agent-curator/experiments/acvr_proposal_global_20260911/RESULTS.md)。以下实现测试只说明训练图、数值和恢复路径可运行，不能替代架构收益证据。

## 1. 可选择的结构

| 选择 | 实际实现 | 用途 |
|---|---|---|
| `proposal: mlp` 或省略 proposal | 原 `VariationalStateHead`，参数结构和默认行为保留 | 历史基线 |
| `BoundaryConditionedProposal(communication="pointwise")` | 新共同输入、等变更新和约束/环境动作，无额外空间通信 | 公平消融基线 |
| `communication="mp"` | 截断内方向张量积消息，默认两个块 | 检查局域传播能解决多少误差 |
| `communication="attention"` | 每个几何、每个条件独立的全原子 attention；不变量权重、等变 values | 可学习的远端传播 |
| `communication="ewald"` | 每块的可行试探电荷经过物理 Ewald 核，再输入可学习更新 | 周期电荷的物理长程传播 |

共同 pointwise 的约束和环境动作本身可能包含全局信息。因此它与完全无跨原子信息的网络不同；局域/远端响应诊断必须说明约束和环境。

代码： [proposal 模块](../../curator/layer/_variational_proposal.py)、[能量与梯度路由](../../curator/layer/_variational_output.py)、[特征接口](../../curator/model/features.py)。

## 2. 网络实际读取什么

输入包括完整的带旋转类型的骨干特征、坐标/晶胞/周期信息、边和边位移、投影后的有效外场、可行原点，以及约束矩阵、约束目标和环境矩阵/刚度。Ewald 电荷泛函额外提供当前电负性、硬度和同一物理核。

PaiNN、MACE、NequIP 均实现 `response_feature_spec()`。它声明完整 irreps 并通过 `flatten(data)` 提供特征；高阶类型进入张量积和不变量门控，没有只截取标量/向量。纯标量骨干也能通过几何方向和外场产生向量 proposal。其他骨干可以提供同样的接口，不能仅凭名称认为已经自动兼容。

记约束为 A、目标为 b，可行原点为 o。投影动作是：

$$
Ao=b,\qquad P=I-A^{\mathsf T}(AA^{\mathsf T})^{-1}A.
$$

实现只对约束数量大小的行系统调用 `solve`，不构造这个完整矩阵，也不求内部物理 Hessian 的逆。

每个块先从隐藏特征读出多通道的状态形状张量。记其中一个通道为 z，试探状态是：

$$
x^{(t)}=o+s_xPz^{(t)}.
$$

这里 s_x 是固定的状态单位尺度；共同更新读取这些通道的切向、约束法向和环境刚度动作。空间通信使用当前隐藏特征；Ewald 分支具体计算：

$$
\psi^{(t)}=G_{\mathrm{Ewald}}x^{(t)}.
$$

最后的读出再次由现有输出层投影到可行空间。块内没有调用完整 KKT、Newton、CG 或参考求解器。

新 proposal 的 `state_scale` 和 `field_scale` 默认均为 1。周期电荷配置使用 0.05 e 和 1 V，对外场、原点、局域系数、试探电荷和核输出执行一致的单位转换。物理能量与驻定 residual 保持原单位；这两个尺度没有改变 loss 权重。首次短训发现直接输入约 20 V/e 的硬度和反复施加核会放大初值，加入上述尺度后各分支初值恢复到同一量级。

## 3. 三种通信的边界

MP 的新增外场信息每层只能沿已有边传播。两个块不能把外场传过三条边；这一点有 Jacobian 测试。

Attention 用逐条件调用确保不同结构和条件隔离。周期几何采用光滑的晶格正弦/余弦相位编码，避免最小镜像切换造成的不连续；这属于可学习几何表示。其计算是稠密的，不作为大体系低复杂度实现。

Ewald 只支持一分量物理电荷及三维周期晶胞，单位为 e、V、eV。各分支使用同一 `EwaldChargeEnergy`；只有 proposal 通信不同。核来自现有 Ewald 物理实现，专用邻居表覆盖完整实空间截断。每次 forward 每个几何构建一次核，在条件、块和能量之间复用；下一次几何/晶胞更新会重新构建。当前实现是稠密参考实现。

## 4. 边界输入与训练

条件级 `inputs` 支持 `local_external_field`、`external_field`、`control_value` 和 `constraint_target`。控制值/目标是 live tensor，原点和有效外场的构造保留其计算图。conq/conp 共轭物性从完整预测能量求导；未驻定时不以电荷求和或估计乘子替代这些导数。

QM/MM 增广仅在训练集启用，幅度来自选中训练样本的外场分量 RMS，固定 seed/几何索引采样。增广条件没有沿用物性标签，只贡献残差；每个几何内部先平均条件损失。

物性监督沿完整 proposal 路径更新骨干、基线、泛函和 proposal。驻定 loss 使用数值相同、输入和物理参数已 detach 的第二次 proposal，只更新 proposal。两条分支的状态逐值比较；不使用 dropout 或训练 BatchNorm。

参考平衡态的一阶/二阶物性由 `StationaryEnergyObservables` 计算：对独立状态和乘子建立拉格朗日量，再通过矩阵自由的线性化 KKT 响应补齐二阶导数。因此参考极化率包含状态弛豫，不是对已 detach 状态直接做部分二阶 AD。该路径仅用于评估，训练不调用参考求解器。

配置入口：

- [acvr_proposal_joint.yaml](../../curator/configs/acvr_proposal_joint.yaml)：小型非周期集成任务。
- [acvr_qmmm_global.yaml](../../curator/configs/acvr_qmmm_global.yaml)：已有 QM/MM 的新联合训练，保留原划分及损失定义。
- [acvr_periodic_charge.yaml](../../curator/configs/acvr_periodic_charge.yaml)：32 站点、128/32/64 几何、每几何 6 条 conq/conp 条件的数值任务。

周期配置本身提供明确的单位尺度 1 以便独立运行；正式对照先使用下面的准备入口，把所选训练数据的 RMS 写入配置。默认物性权重各 1、驻定权重 0.1。精确逐原子电荷不进入周期任务的默认监督。

## 5. 准备和启动

以下命令从仓库目录运行。准备入口只生成配置和来源记录，没有提交作业：

```bash
../.venv/bin/python scripts/acvr_prepare_proposals.py \
  --config curator/configs/acvr_periodic_charge.yaml \
  --output /path/to/new/periodic_charge_joint
```

输出包含五个分支 YAML、训练统计、数据 manifest、源码压缩包和逐文件哈希。QM/MM 使用 `acvr_qmmm_global.yaml`，默认生成 MLP、pointwise、MP、attention 四个分支。重复 `--override` 可以缩小数据/步数做 smoke；准备后的 loss 标度会随实际选中的训练集重算。

运行一个已准备的分支：

```bash
../.venv/bin/python -c 'from curator.commands.train import run_train_config; from omegaconf import OmegaConf; run_train_config(OmegaConf.load("/path/to/new/periodic_charge_joint/attention.yaml"))'
```

准备的配置添加每 5 epochs 的固定验证子集审计。`supervised_limits` 留空时仅记录；获得匹配 pointwise 的验证物性 RMSE 后，将相应 1.05 倍门槛填入配置，才会保存通过门槛且状态误差更小的 `best_proposal_equilibrium.ckpt`。正常训练的 predicted-loss 最佳模型和 last checkpoint 仍由原回调管理。

固定泛函诊断使用 [独立运行说明](PROPOSAL_DIAGNOSTIC.md)：

```bash
../.venv/bin/python scripts/acvr_proposal_diagnostic.py \
  --communication attention --steps 600 \
  --output /path/to/new/periodic_charge_diagnostic_attention
```

无 checkpoint 时使用周期数据的固定教师；指定 `--model-path` 时保留原骨干/泛函，只替换 proposal 并建立新 optimizer。诊断缓存冻结特征/核，分别报告缓存构建、完整前向和参考成本，不能把缓存训练速度当作端到端加速。

需要更细的成本分解时，可使用 PyTorch profiler 的 `acvr.geometry_context`、`acvr.boundary_context`、`acvr.constraint_projection`、`acvr.proposal`、`acvr.ewald_apply` 和 `acvr.reference_solve` 区段；普通评估另保存预测与参考的实际耗时。

## 6. 已验证的范围

2026-09-11 最终联合回归：**178 passed，264.01 秒**，CPU 单线程。包含本轮新增测试，以及旧联合模型、loss、数据、分布式梯度归属和 checkpoint 生命周期回归。记录保存在 `/workspace/data/work/agent-curator/experiments/acvr_proposal_global_20260911/implementation_checks/pytest.log`。

新增测试覆盖三个骨干、标量/向量、高阶特征影响、O(3) 变换、周期镜像、条件/几何隔离、约束、远端外场 Jacobian、力和外场二阶导数、残差梯度归属、Ewald 核及坐标/晶胞导数、live conq/conp 导数、checkpoint 架构与尺度合同。

同 seed 下，新 pointwise/MP/attention/Ewald 的物理参数和共同 proposal 参数逐位相同。原 MLP 保留历史构造和随机初始化顺序，不纳入上述四分支逐位初始化保证；精细架构消融使用新 pointwise，MLP 作为单列的历史结构基线。准备 manifest 明确记录这一差别。

已完成各通信分支的小型两步训练；周期 Ewald 连续 22 步和从第 2 步恢复到第 22 步的参数、Adam、循环计数及数据打乱状态逐位一致。为此修复了有限 `max_steps` 下 Lightning 重放已完成 epoch 的问题。

另外，经过准备入口的新 attention 配置完成了周期电荷及真实 QM/MM 的两步联合训练，包含相应物性导数与残差。固定泛函诊断也跑通了周期三个通信分支和旧 QM/MM checkpoint 的短检查。

默认 48 通道、32 原子 Ewald 配置也完成两步训练；两个真实 QM/MM 验证几何完成预测态/参考平衡态的偶极和极化率比较，参考失败数为零。完整工件和来源记录位于上述 `implementation_checks` 目录。

上述短程结果是正式实验前的实现核查。之后的 600 步固定泛函对照和有界联合 pilot 记录在本页开头链接的实验报告；目前没有证据证明完整目标下的泛化精度或摊销加速提升。178 项回归后训练核心源码未变，仅诊断脚本及其测试更新，当前诊断专项再次通过 4 项检查，见 [源码和复测核验](/workspace/data/work/agent-curator/experiments/acvr_proposal_global_20260911/analysis/source_verification_after_diagnostic_changes.json)。
