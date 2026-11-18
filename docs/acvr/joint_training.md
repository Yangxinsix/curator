# GNN 联合变分训练

本页对应 `acvr_joint.yaml` 和 `acvr_spice_joint.yaml`。两者通过普通
`curator-train` / `curator.commands.hydra_entrypoints.train` 启动，使用一个
Lightning task、一个 optimizer，在每个 batch 中同时训练物理模型和 proposal。
原 `acvr_oracle*`、`acvr_spice.yaml`、`acvr_spice_proposal.yaml` 保持二阶段参考实验的含义。

## 模型与参数

`NeuralNetworkPotential` 的 representation 使用普通 PaiNN。新增
`curator.layer.VariationalEnergyOutput` 接收共享的标量/等变特征；同一接口也支持
提供相应 feature spec 的 MACE。scalar 状态允许多个标量通道，vector 状态有三个
极向量分量。多种变量混合在同一状态布局中的应用仍需显式设计变量布局。

对于结构 $R$、条件 $\xi$ 和内部状态 $x$，模型计算：

$$
h = \operatorname{GNN}_{\theta_G}(R),\qquad
\widehat{x} = \operatorname{Project}_{Ax=b}\bigl(P_\phi(h,\xi)\bigr).
$$

$$
\widehat{E}(R,\xi)
= \Phi_\theta(R,\widehat{x},\xi)
= c_{\theta_c}(h) + U_{\theta_U}(h,\widehat{x},\xi)
+ E_{\mathrm{boundary}}(\widehat{x},\xi).
$$

| 参数组 | 模块 | 物理监督的梯度 | 残差的梯度 |
|---|---|---|---|
| `backbone` | GNN，排除能量 readout | 允许 | 无 |
| `baseline` | representation 的能量 readout，输出 $c$ | 允许 | 无 |
| `functional` | 内部能量模块 $U$ | 允许 | 无 |
| `proposal` | 状态预测头 $P$ | 允许 | 允许 |

能量标签监督的是 $\widehat E$，没有另外强迫 $c$ 拟合完整真实能量。
力是 $-d\widehat E/dR$，包含 proposal、GNN 和投影随坐标变化的路径。
每个结构可带多个条件，所以正式输出采用 `variational_prediction.labels` 下的
`condition_energy`、`condition_forces`、`variable_value`，不把条件维度当成结构维度。

默认 `StableInternalEnergy` 使用正二次、正四次局部能量和正图耦合，提供一个平滑、
对内部状态凸的起点。网络计算局部能量，没有构造全局稠密 $H$。这不是框架对能量形式
的限制，也不是已验证的通用物理势；可以传入实现
`forward(context, state[n_atoms, components]) -> scalar` 的 `internal_energy` 模块。
`context` 提供共享特征、位置、边、场及约束信息。默认 heads 使用去掉等式约束
gauge 分量的 `network_field`，共轭能量保留完整场，因此固定总电荷时整体平移电势
只改变正确的能量参考，不改变预测状态。自定义模块也应遵守所选变量的 gauge 约定。
替换后应重新检查对称性、稳定性、
远程相互作用和所需物理量的含义。

## 一个 batch 的 loss 和更新

$$
L = \sum_k w_k L_k^{\mathrm{supervised}}
  + w_r L_{\mathrm{stationarity}}.
$$

监督项来自实际标签。残差来自同一能量对 $x$ 的偏导，经等式约束允许方向的投影；
不需要精确状态标签。`VariationalJointLoss.residual_weights` 消费
`variational_prediction.residuals.stationarity`，复用已有条件/结构加权和单位尺度。
残差输出必须声明其来源是 `variational_energy_gradient`、梯度策略是 `proposal_only`，
并携带固定物理尺度的归一化信息。驻定残差小不等于已经证明稳定最小值。

计算图内有两条路径：

1. 监督路径完整连接全部参数及坐标。
2. 残差路径使用 `h.detach()` 重新运行 proposal，并以 detached 的能量参数和条件
   计算能量，保留对预测状态的微分。这里使用 `torch.func.functional_call`，不使用
   `torch.no_grad()` 包住能量。默认 heads 无 dropout 或 batch norm，两条路径的状态
   数值一致。

GNN 只运行一次；proposal 和内部能量模块会为残差增加一次计算。最后仍由 Lightning
自动优化执行总 loss 的一次 backward 和更新。不要手工把局部 `autograd.grad` 结果
加到 `.grad`：那需要额外正确处理 DDP 同步、AMP 缩放和梯度累计。

默认训练不调用精确 teacher、Newton、dense KKT 或完整 Hessian。残差反传和力监督
仍需要高阶自动微分相关运算；不构造 Hessian 并不意味着没有二阶导数成本。
普通训练配置 `compute_responses: false`，无需计算响应块。

## 启动与恢复

以下命令从仓库根目录运行；本工作区可用 `../.venv/bin/python`。

```bash
../.venv/bin/python -c 'from curator.commands.hydra_entrypoints import train; train()' \
  --config-name acvr_joint run_path=runs/acvr_joint \
  trainer.max_epochs=20
```

`acvr_joint` 的默认数据是可复现的解析参考标签；模型本身使用真实 PaiNN，
不是旧的专用 Oracle MLP。配置不要求任何预训练 physical checkpoint。

SPICE 的实际能量、力标签可以直接进入相同路径：

```bash
../.venv/bin/python -c 'from curator.commands.hydra_entrypoints import train; train()' \
  --config-name acvr_spice_joint run_path=runs/acvr_spice_joint \
  data.datapath=/path/to/SPICE.hdf5 trainer.max_epochs=20
```

`SpiceACVRDataModule(variational=True)` 保留来源、距离划分、单位和原始标签，把总电荷
作为等式边界，不生成原子电荷 teacher。该配置示范 E/F 联合训练；默认内部能量尚未
替换为针对 SPICE 调整的长程电荷模型，不能据此声称原电荷模型的准确性或加速收益。

断点恢复使用同一配置和数据定义：

```bash
../.venv/bin/python -c 'from curator.commands.hydra_entrypoints import train; train()' \
  --config-name acvr_joint trainer.max_epochs=30 \
  'model_path={path: runs/acvr_joint/model_path/last.ckpt, mode: resume}'
```

新配置用 `val_joint_predicted_loss` 选择 best checkpoint。
`JointModelCheckpoint` 还负责在非最佳 epoch 更新恢复用的 `last.ckpt`。
默认参考数据为 float64、trainer 为 `64-true`；使用 mixed precision 时需让数据和模型
采用匹配的 float32 基础 dtype。SPICE 配置可设置 `data.default_dtype=float32`；
内部约束和能量求导部分保留 float32/64，GNN 可以使用 autocast。

## 评估与响应

```bash
../.venv/bin/python -c 'from curator.commands.hydra_entrypoints import evaluate; evaluate()' \
  --config-name acvr_joint_evaluate \
  model_path=runs/acvr_joint/model_path/last.ckpt
```

这里比较同一能量模型在 `predicted` 与 `converged` 状态的标签误差，以及状态、
能量、力和启用的响应之间的差异。结果及 checkpoint 元信息保留状态模式和求导约定。
`converged` 是小体系 Newton 参考评估，会构造稠密 Hessian，检查收敛与局部稳定性；
不宣称任意非线性能量的全局最优。

SPICE 使用 `--config-name acvr_spice_joint_evaluate`，设置相应 `model_path` 和
`datapath.datapath`。若训练时设定 `train_limit`、`val_limit`、`test_limit`，评估时应在
`datapath` 下设成相同值，保留相同的划分元信息；该数据没有内部响应标签，默认不开启
响应监督。数据的来源、划分、limits 和 dtype 元信息都必须一致：若训练使用
`data.default_dtype=float32`，评估命令也要增加 `datapath.default_dtype=float32`。
电荷对电势的响应单位为 `e^2/eV`，SPICE 配置显式记录这个单位。

`compute_responses=True` 按需输出有效共轭场到状态的响应：预测模式对 proposal 求导，
收敛模式线性化驻定条件。后者调用
`solve_functional_linear_response(problem, state, stationarity_rhs, constraint_rhs)`，
以 HVP 和投影共轭梯度解线性系统：

$$
H\,\delta x + A^T\delta\lambda = r,\qquad A\,\delta x=d.
$$

右端需包含实际扰动对驻定条件和约束的偏导。该响应求解器不构造全状态 Hessian，
但要求约束切空间的曲率正定，并报告迭代不收敛或非正曲率。
`FunctionalProblem.energy/state_gradient/hvp` 也可以单独用于自定义能量。

预测能量的二阶坐标导数可通过 `retain_response_graph=True` 保留求力计算图。
收敛状态的力使用拉格朗日函数的包络导数，包含几何依赖约束的贡献；直接对冻结收敛
状态的力再次微分会遗漏状态响应，因此该组合会明确拒绝。收敛二阶量要走隐式响应。
当前接口没有封装晶格动力学矩阵、质量加权、波矢扫描或完整声子谱生成工具。

这一差异有独立解析验收：

$$
\Phi(u,x)=2u^2+(x-u)^2,\qquad \widehat{x}(u)=0.
$$

在 $u=0$，预测态和精确态都为零，驻定残差也为零，但预测能量曲率为 $6$。
真实最小值满足 $x^*(u)=u$，隐式响应给出 $dx^*/du=1$，松弛能量曲率为 $4$。
测试同时检查这两个答案，防止用单点小残差代替响应准确性。

## 验证入口

```bash
../.venv/bin/python -m pytest -q \
  test/test_acvr_joint_loss.py \
  test/test_acvr_joint_model.py \
  test/test_acvr_joint_distributed.py \
  test/test_acvr_joint_evaluator.py \
  test/test_acvr_joint_lifecycle.py \
  test/test_variational_functionals.py \
  test/test_variational_functional_response.py \
  test/test_acvr_spice_joint_data.py
```

验收包括实际 GNN 的四组梯度、带 AdamW 动量的残差隔离、两进程 DDP、梯度累计、
CPU bf16 mixed precision、力与外场响应有限差分、禁止训练时调用 teacher/完整 Hessian、
checkpoint 重建及恢复、同一模型的两种响应对照。旧二阶段、约束和求解器测试应继续通过。
