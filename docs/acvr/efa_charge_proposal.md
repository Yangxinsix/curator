# 恒电荷 EFA proposal：实现与微调

恒电势与统一数据配置现已支持，见 [统一边界条件接口](efa_boundary_conditions.md)。本文保留恒电荷计算路径的说明；新的 proposal 另外接收经 gauge 投影的电势，开放总电荷时跳过 Fukui 总和修正。

入口配置是 [`acvr_efa_charge.yaml`](../../curator/configs/acvr_efa_charge.yaml)。它将已经训练的 `QConditionedEFAAdapter` 转换为独立的全局电荷 proposal 和显式电荷能量，并复用 Curator 的 `VariationalEnergyOutput`、`VariationalJointLoss`、参数分组及检查点恢复。训练仅使用能量、力和驻定残差，不需要原子电荷标签，也没有内部电荷迭代求解器。

## 完整计算路径

```text
MACE → 完整特征 h(R) ─────────────────────→ 显式 q-EFA 能量 U_EFA(h,R,q)
           │                                             ↑
           ├→ 等变投影 + (Q, Q/N) → 两层全局 EFA → g     │
           │                                      │      │
           └→ Inv(h) ──┐                          │      │
                       └→ [Inv(h), Inv(g), Q, Q/N]        │
                             ↓ 原始电荷、Fukui 权重        │
                             ↓ 恒电荷投影                 │
                             q ──────────────────────────┘

U_total = U_MACE + U_EFA → 能量、完整坐标导数得到的力
                       → 独立 q 偏导得到的驻定残差
```

proposal 的内部投影只作用于其自身的通信分支。能量模块仍直接读取完整 MACE 特征；proposal 的读出还具有 `Inv(h)` 旁路，保留各个偶标量通道及其他 irrep 通道的光滑范数。不存在供能量和 proposal 共用的特征压缩瓶颈。

## 全局 proposal 与 Fukui 投影

[`EFAChargeProposal`](../../curator/layer/efa/proposal.py) 默认使用两个独立参数的全局块，每块复用 `EquivariantEFA` 和 `EquivariantStateFusion`。proposal 与能量模块共享输入特征，不共享可训练参数。默认隐藏表示为 `64x0e + 16x1o`，attention 宽度为 32；`tensor_integration=False`。固定通道数、层数和积分方向数时，attention 中间量随原子数线性增长。

总电荷在 attention 之前进入特征，也进入最终读出。以 `charge_scale = s_q` 表示电荷尺度，条件输入为

\[
b=(Q,Q/N)/s_q,\qquad z_i^{(0)}=W_hh_i+W_bb.
\]

两个全局块得到最终表示后，读出产生两个标量：原始电荷和分配权重的 logits。原始电荷使用线性输出，不使用 tanh 截断：

\[
\widetilde q_i=s_q a_i,\qquad
w_i=\frac{\exp(f_i)}{\sum_j\exp(f_j)},\qquad
q_i=\widetilde q_i+w_i\left(Q-\sum_j\widetilde q_j\right).
\]

因此在精确算术下，\(\sum_iq_i=Q\)。原始电荷与权重都读取全局表示；即使电荷差额为零，proposal 的全局分支仍能改变原子的电荷分布。`predict_components(context)` 可返回投影前电荷和归一化权重；`forward(context)` 返回已经可行的 `[N,1]` 电荷。

现有变分框架最后的正交投影继续作为浮点误差下的可行性保障。在精确算术下，它对已经满足约束的 proposal 输出及其参数导数均为恒等操作。该投影不改变驻定残差的定义。

这里的正归一化权重是 **Fukui 形式的可学习电荷差额分配**，尚不能等同于已验证的真实 \(\partial q_i/\partial Q\)。原始电荷、权重也依赖 Q，完整的 Q 导数必须对整个网络求导。

## 显式 q 能量与检查点迁移

[`QConditionedEFAEnergy`](../../curator/layer/efa/explicit_q.py) 接受 `forward(context, state)`，其中 `state` 是 `[N,1]` 电荷。它保留已训练的 K/V 门控及能量读出：

\[
\mathcal Q_i=Q_0(h_i),\quad
K_i=K_0(h_i)+q_iB_K(h_i),\quad
V_i=V_0(h_i)+q_iB_V(h_i),
\]

\[
U_{\mathrm{EFA}}
=s_E\sum_iw_2^\top\operatorname{SiLU}(W_1G_i+b+w_qq_i).
\]

投影和读出逻辑与旧 q-EFA 共用；位置编码、attention、周期处理及求和直接调用原 `EFAAdapter.forward`。迁移保留原 `energy_scale`，显式能量不包含旧 `q_head`。MACE 基线和 EFA 修正各计入一次。

`efa_charge_proposal` 模型 transform 要求恰有一个 `q_mode='learned'` 的 `QConditionedEFAAdapter`。它复制模型、迁移全部能量参数、新建独立 proposal，替换旧 EFA 输出和旧力输出，随后调用原框架的变分绑定、力输出校验及输出收集。应在构造优化器前执行。相同输入 h、R、q 下，迁移前后的 EFA 修正相同；新 proposal 的 q 不必等于原局部头的 q，因此转换本身不承诺整个模型预测保持不变。

状态、共轭变量及能量的单位契约分别是 `e`、`V`、`eV`。这是新模型的变量和边界条件定义，并不说明旧 latent q 已经具有物理电荷精度，也不替代后续验证。

## 驻定条件与梯度路由

恒电荷只约束 \(\sum_iq_i=Q\)。令独立状态的能量偏导为

\[
g_i=\left.\frac{\partial U}{\partial q_i}\right|_{\mathbf R},\qquad
r_i=g_i-\frac1N\sum_jg_j.
\]

训练使用 \(r_i/s_r\) 的均方，`residual_scale` 是固定物理尺度，默认 `1.0 V`。残差不乘 Fukui 权重；\(r=0\) 表示所有原子的电荷共轭导数相等。总电荷可行性由投影直接保证。

`VariationalEnergyOutput` 保留已有的双计算图路由：

- **能量和力监督**：保留 MACE、proposal、Fukui 投影、显式能量的全部计算图，力为 \(-dU(\mathbf R,\mathbf q(\mathbf R))/d\mathbf R\)，包含完整 \(dq/d\mathbf R\)。监督损失可以更新这三个参数家族。
- **驻定残差监督**：用脱离计算图的特征和几何重新执行同一个确定性 proposal；能量参数通过 `torch.func.functional_call` 使用 detached 副本。保留残差经 q 返回 proposal 的导数，仅更新 proposal，不更新 MACE 或能量参数。两次 proposal 的状态必须逐位一致；模块中不使用 dropout 或 BatchNorm。

原生参数组为 `backbone`、`baseline`、`functional`、`proposal` 等，参数只归属一次。默认配置对全部模型微调 30 epochs：物理参数 Adam 学习率 `1e-5`、`eps=1e-8`；proposal 学习率 `1e-4`、`eps=1e-12`。配置使用确定性 FP32、batch size 4、梯度累积 3、无 scheduler、无 weight decay。能量按每原子误差训练，能量/力监督权重均为 200，驻定残差权重为 1。

## 原生训练与恢复

下面是 5,000 帧数据的命令示例；替换路径，使用与已有 checkpoint 一致的数据划分及能量参考。`ACVREFDataModule` 要求训练、验证、测试索引按指定数量完整且不重复地划分源文件。

```bash
curator-train --config-name acvr_efa_charge \
  run_path=/path/to/new_run \
  model_path.path=/path/to/trained_q_efa.ckpt \
  data.datapath=/path/to/dataset.extxyz \
  data.num_train=4275 data.num_val=475 data.num_test=250 \
  data.split_file=/path/to/split.json \
  data.energy_reference_file=/path/to/energy_reference.json \
  data.boundary.total_charge=0.0
```

默认 `model_path.mode=model` 先加载训练 checkpoint 的参数，再执行配置中的 `efa_charge_proposal` transform。不是从头初始化能量模型。`data.boundary.total_charge=0.0` 是显式恒中性边界；若源文件每帧有自己的 Q，可改用 `data.boundary.total_charge=null +data.boundary.total_charge_key=net_charge` 读取对应 ASE `info` 字段，缺失值会报错。

从本方案自己的检查点恢复时，应选择 `mode=resume` 并清空 transform；保持数据、seed、dtype、划分和能量参考一致：

```bash
curator-train --config-name acvr_efa_charge \
  run_path=/path/to/new_run \
  model_path.path=/path/to/new_run/model_path/last.ckpt \
  model_path.mode=resume 'model_path.transform=[]' \
  data.datapath=/path/to/dataset.extxyz \
  data.num_train=4275 data.num_val=475 data.num_test=250 \
  data.split_file=/path/to/split.json \
  data.energy_reference_file=/path/to/energy_reference.json \
  data.boundary.total_charge=0.0
```

恢复复用序列化的动态模型结构，并由 Lightning 恢复训练权重、优化器和进度；不会再次新建 proposal。`trainer.max_epochs` 指总目标 epoch 数。默认 `save_entire_model=true` 用于保存这套动态结构，数据模块同时保存划分、元数据和训练 shuffle 的 RNG 状态。

输出沿用 `variational_prediction`：其中 `condition_energy`、`condition_forces` 和 `variable_value` 是条件对应的预测，`residuals.stationarity` 是归一化驻定残差。普通几何基线能量不会冒充条件总能量。

## 当前支持范围

- 每个原子只有一个标量电荷通道。支持全体系恒电荷、开放总电荷恒电势，以及分组恒电势加一条整体电荷约束；多条分组电荷约束及 Robin 边界明确报错。
- 支持非周期 FFF 或全周期 TTT；不支持部分周期。TTT 沿用 EFA 的原始晶格编码，不是 Ewald 静电项，也不保证任意单原子换晶胞镜像后的不变性。
- FFF 使用有限球面积分；旋转等变性有积分数值误差。默认 110 个方向与默认频率范围配套。
- 全局通信在单个结构内完成，不跨 batch 中的不同结构。默认配置没有启用额外响应性质或内层求解。

相关测试包括 `test_efa_charge_proposal.py`、`test_efa_charge_conversion.py` 和 `test_efa_fixed_charge_integration.py`，覆盖电荷约束、全局响应、迁移一致性、力的完整链式求导及 proposal-only 残差路由。代码和这些数值检查只确认实现契约，训练精度收益需要单独的配对实验验证。
