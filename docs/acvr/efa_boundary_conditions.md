# EFA proposal 的统一边界与数据约定

恒电荷和恒电势使用同一个 `VariationalJointLoss`、同一个全局 EFA proposal 和同一个内部能量模型。用户只声明边界控制量和能量标签约定，不编写新的 loss。已有恒电荷配置和 checkpoint 可继续使用；更换训练数据或边界属于新的微调任务，严格恢复原任务仍使用 `mode: resume`。

## 数据配置

在 [`acvr_efa_charge.yaml`](../../curator/configs/acvr_efa_charge.yaml) 的原生训练配置中，恒电荷写为：

```yaml
data:
  boundary:
    type: fixed_charge
    total_charge: 1.0                 # e
  energy_convention: internal        # 标签是内部能 U
```

每帧 Q 不同时，将 `total_charge` 替换为 `total_charge_key: total_charge`，从 ASE `Atoms.info` 读取。不要同时保留常量和对应的 key。

纯恒电势、允许交换总电荷时：

```yaml
data:
  boundary:
    type: fixed_potential
    potential: 0.3                   # V，与带符号电荷 q 共轭
    total_charge: null               # 开放电荷，不能填伪造的 Q=0
  energy_convention: internal
```

也可以直接使用 [`acvr_efa_potential.yaml`](../../curator/configs/acvr_efa_potential.yaml)，它继承恒电荷配置的模型、loss 和优化器，只改变边界。每帧电势使用 `potential_key`；数据已有热力学势 G 标签时设置 `energy_convention: thermodynamic`。

两个电极的电势及整体电中性：

```yaml
data:
  boundary:
    type: fixed_potential
    groups: {left: [0, 1], right: [2, 3]}
    potential: {left: -0.3, right: 0.3}
    total_charge: 0.0
  energy_convention: internal
```

`groups_key` 和 `potential_key` 可分别读取每帧的分组字典和电势字典。分组按原子索引声明，必须互不重叠并覆盖所有可变电荷原子。电势参考与符号必须和数据的能量约定一致：本接口的定义为 `G = U - sum(V_i*q_i)`，不是电子数化学势的直接输入，也不是空间外加静电势的 `+q*phi` 项。单个整体电荷约束下，无效的公共电势平移不会改变 q；能量保留对应的库功常数。可选 `gauge` 直接沿用 Curator 的 `GaugeSpec`，默认 `allow` 保留输入参考。

显式 `boundary` 必须同时声明 `energy_convention`。旧的顶层 `total_charge` / `total_charge_key` 接口继续保留原来的样本及 metadata 格式，不能与新 `boundary` 混用。

## 无电荷约束：潜变量与变分模式

显式写 `boundary.type: none` 才表示没有总电荷约束、没有电势控制。
省略 `boundary` 或写 `boundary: null` 仍使用旧接口，默认固定总电荷为零。
`none` 不接受非空的电荷、电势、分组或 gauge 参数。

仅用能量和力训练全局 q 预测头，使用 `acvr_efa_latent_proposal` 配置：

```yaml
data:
  boundary:
    type: none
    total_charge: null
  charge_mode: latent
  energy_convention: internal
```

网络返回原始 q，不做 Fukui 修正，能量仍为原来的 q-EFA 表达。
数据与模型使用已有的 `predicted_auxiliary` 变量语义。框架跳过驻定残差
求导及重复 proposal 前向，同一个联合 loss 自动将驻定项标记为不适用、
有效权重为零；用户无需修改 loss。日志中的驻定 RMS 为 `null`，不是零残差。
q 是通过能量和力学习的内部变量，其数值不被当作已验证的物理原子电荷。

保留无约束驻定训练时，选择 `acvr_efa_unconstrained`，或将上面的
`charge_mode` 改为 `variational`。这时仍不做 Fukui 修正，但训练的驻定残差是
完整的 `dU/dq`，不减去均值。它对应既定电荷化学势参考下的零电势开放体系，
不等于未指定的任意物理边界。两种 preset 继承相同的模型、loss 与优化器。

从 q-EFA 检查点创建 proposal 时继续使用默认的 `efa_charge_proposal` 转换；
从已有 proposal 检查点切换模式时使用下面的 `efa_proposal_boundary` 转换。
模式写入数据 metadata 和检查点；`mode: resume` 不允许偷偷更换模式。
已有恒电荷和恒电势配置保持 `variational` 模式；`latent` 必须显式搭配 `none`。

## 模型与训练路径

`ChargeBoundary` 仅将 ASE 配置转成已有的 `BoundarySpec` recipes。已有框架随后编译电荷约束 A、目标 Q 和电势控制，生成以下路径：

```text
几何 → MACE 完整特征 → 全局 EFA proposal → q → q-EFA 内部能 U
边界 → Q、Q/N、有效电势、开放电荷标记 ────↑
     → 实际电荷约束决定是否做 Fukui 修正
     → G = U + 边界能 → 驻定残差、力 → 同一个联合 loss
数据约定 ─────────────→ condition_energy 选择 U 或 G
```

恒 Q：`q = raw + softmax(logits) * (Q - sum(raw))`。驻定残差为 `dU/dq - mean(dU/dq)`。

纯恒 V：`q = raw`，总电荷由模型决定。驻定残差为 `dU/dq - V`。

恒 V 加整体电荷约束：保留 Fukui 修正，残差为约束切空间上的 `dG/dq`。Fukui 权重不改变驻定残差的定义。电势输入使用已有 `network_field`，去掉只改变能量常数的 gauge 方向。

能量/力损失向模型和 proposal 反传；驻定项仍仅更新 proposal。没有增加 q 标签或 Newton-CG 内循环。固定网络宽度和积分方向数时，proposal 仍采用线性 EFA attention。

**力始终来自边界规定的热力学势 G**，包含 proposal 的完整 `dq/dR`。即使能量标签声明为内部能 U，也不会错误地使用固定 V 下的 `-dU(R,q(R))/dR`。归一化、权重和残差接口在两种边界下共用；`residual_scale` 的单位仍为 V。

## 已训练模型的微调和恢复

从旧的 learned q-EFA 模型开始，沿用 `efa_charge_proposal` 转换，保留能量权重并新建 proposal。

从已经带有 proposal 的模型切换边界或数据，使用：

```yaml
model_path:
  path: /path/to/proposal_checkpoint.ckpt
  mode: model
  transform:
    - type: efa_proposal_boundary
```

该显式转换保留所有已学权重。旧 v1 proposal 新增的电势分支零初始化，旧恒 Q 的 q、能量、力和导数在数值精度范围内保持一致；v2 模型保留已经学到的电势分支。转换清除旧数据绑定，再从本次 data 配置绑定约定；新参数需要新的优化器。原生模型加载不会重新拟合原 MACE 的缩放参数。

恢复同一任务时使用 `mode: resume`，不添加转换，保留原参数、优化器和数据约定。旧完整模型不会在加载时偷偷增加参数；数据约定不匹配会明确报错。

## 支持范围与验证

本 proposal 支持全体系恒总电荷、开放总电荷恒电势，以及分组恒电势加一条整体电荷约束。多条分组电荷约束、Robin 边界和部分周期性尚不支持，会明确报错。底层变分框架更广的能力不等于该 proposal 已支持全部边界。

回归覆盖解析 U/G 关系、全路径力有限差分、gauge、混合边界 batch、同一联合 loss 反传和参数更新、Hydra 配置、原生微调入口、旧 checkpoint 严格恢复及显式升级保持预测。实现测试不代表在恒电势真实数据上已经证明精度收益。
