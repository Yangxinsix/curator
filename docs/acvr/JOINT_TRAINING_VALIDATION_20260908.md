# GNN 联合变分训练实现与验证：2026-09-08

## 修改前快照

在修改实现前，当前工作树已全部提交为：

```text
d788852 Save current ACVR training and validation work before joint training refactor
```

本次新增联合路径保留旧二阶段实验，不改变其求解/teacher 语义。
使用说明见 [joint_training.md](joint_training.md)。

## 已实现的闭环

- 普通 PaiNN/MACE 共享特征接 `VariationalEnergyOutput`；scalar 多通道与 vector 三分量
  分别使用合适的输出头。
- 预测状态经过等式投影后进入最终能量；能量监督使用完整组装能量，力保留 proposal
  对坐标的完整导数路径。
- 一次训练、一个 optimizer、四组参数：`backbone`、`baseline`、`functional`、`proposal`。
- 真实能量导出的驻定残差使用独立梯度路径，仅更新 proposal；默认不生成 teacher、
  不运行 Newton、不组装完整 Hessian 或完整响应。
- 通用标量能量接口提供偏导与 HVP；可选 field response 区分 proposal Jacobian 与
  收敛状态的隐式响应，线性响应通过 HVP/投影 CG 求解。
- 默认 heads 对固定约束的电势 gauge 平移保持正确状态不变性；边界能量保留对应参考
  平移，默认图耦合在截断处的值、一阶和二阶导数为零。
- evaluator、checkpoint 元信息及恢复流程保留预测态/收敛态和求导策略；SPICE 有独立
  联合训练、评估配置。

## 回归测试

仓库根目录执行：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ../.venv/bin/python -m pytest -q \
  test/test_acvr_*.py test/test_variational_*.py \
  test/test_spice_acvr_data.py test/test_spice_charge_model.py \
  test/test_model_loading_rescale.py test/test_model_output.py \
  test/test_direct_force_output.py test/test_direct_force_transform_order.py \
  test/test_native_neighborlist.py test/test_training_loss_consistency.py \
  --basetemp=/tmp/acvr_joint_final_20260908
```

结果：**308 passed, 6 skipped, 58 warnings，156.37 秒**。
日志位于 `runs/acvr_joint_validation/regression.log`。
跳过项是四项依赖外部 ACVR LAMMPS 的测试，以及分别需要 PySCF 多电荷和 TRECI
数据工件的两项测试；本次未将这些外部验证计为通过。
响应单位与文档复审后，另对响应、联合 loss 和 SPICE 联合数据相关测试复验，
13 项通过；`git diff --check` 和两份新增文档的数学分隔符检查通过。

关键验收不只检查 loss 数值：

| 要求 | 验收证据 |
|---|---|
| 真实共享 GNN 联合训练 | PaiNN/MACE 模型测试；真实 CLI 和 SPICE E/F batch |
| 残差不更新物理参数 | 单进程和双进程下，先填充 AdamW 非零动量，再只反传残差；物理参数 `.grad is None`，参数与 optimizer 状态逐位不变 |
| 单次自动优化的分布式正确性 | 两 rank CPU DDP 与相同数据单进程的参数/梯度等效 |
| 梯度累计 | batch 2 × 累计 2 与 batch 4 的更新等效 |
| 混合精度 | 真实 Lightning CPU bf16 mixed 训练；物理求导和约束使用 master dtype，梯度和更新有限 |
| 完整预测态力 | 非平衡 proposal 的坐标有限差分；等变旋转检查 |
| 无隐藏 teacher/Hessian | 训练测试将内部参考求解和完整 Hessian 构造替换为抛错桩，训练仍成功 |
| 可选响应 | proposal 响应有限差分、收敛 IFT 解析解和有限差分、实际 response loss 的联合反传 |
| 几何依赖约束 | 收敛态力的拉格朗日乘子项与坐标有限差分一致 |
| Gauge | 非零固定总电荷时状态和力不变，能量平移为 $-Q\,\Delta V$，响应沿 gauge 方向为零 |
| 二阶响应区别 | 独立解析例子的预测曲率 $6$ 与松弛曲率 $4$ 均得到正确结果 |
| 保存/恢复 | full 与 state-dict 模型重建逐位一致；断点恢复与连续训练的参数和 Adam 状态逐位一致 |
| 最新 checkpoint | 验证指标不改善时 best 留在 epoch 0，last 推进到 epoch 1 |
| 旧功能兼容 | 原二阶段、二次求解器、边界、普通输出、模型重建和 TorchScript 相关测试继续通过 |

## SPICE 真实数据小训练

数据：本机 `/workspace/data/datasets/spice/2.0.1/ion_pairs/SPICE.hdf5`。
沿用原固定距离划分，选取覆盖全部 28 类离子对的 56 个训练、28 个验证、28 个测试结构。
使用真实 E/F 标签；没有原子电荷监督、精确状态 teacher 或预训练物理 checkpoint。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ../.venv/bin/python \
  -c 'from curator.commands.hydra_entrypoints import train; train()' \
  --config-name acvr_spice_joint \
  run_path=runs/acvr_joint_validation/spice_verified \
  data.train_limit=56 data.val_limit=28 data.test_limit=28 data.batch_size=14 \
  trainer.max_epochs=2 trainer.num_sanity_val_steps=0 trainer.log_every_n_steps=1
```

训练完成两轮。`last.ckpt` 的 `epoch=1`、`global_step=8`，包含一个 optimizer 和上述
四个参数组；contract 记录 `energy_state=predicted`、`full_proposal_path`、
`proposal_only`，电荷响应单位显式为 `e^2/eV`。

独立 CLI 评估：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ../.venv/bin/python \
  -c 'from curator.commands.hydra_entrypoints import evaluate; evaluate()' \
  --config-name acvr_spice_joint_evaluate \
  model_path=runs/acvr_joint_validation/spice_verified/model_path/last.ckpt \
  run_path=runs/acvr_joint_validation/spice_evaluate \
  datapath.train_limit=56 datapath.val_limit=28 datapath.test_limit=28
```

结果工件：

- `runs/acvr_joint_validation/spice_verified/model_path/last.ckpt`
- `runs/acvr_joint_validation/spice_evaluate/metrics.json`
- `runs/acvr_joint_validation/spice_evaluate/results.pt`

28 个测试条件的局部参考求解全部收敛。预测态总 loss 为 2.9952422567，收敛态为
2.9699553813；这些是配置定义的加权归一化目标，不是单一物理量误差。
两种模式之间的状态、能量、力标量 RMSE 分别为 0.1315390998 e、0.0384639743 eV、
0.0002003638 eV/Angstrom。它们衡量同一模型的求解近似差异，不代表对 DFT 的对应 RMSE。

这次小训练证明配置、数据、梯度、保存和独立评估闭环可执行。它没有验证充分收敛后的
物理精度、端到端加速、跨体系泛化或完整声子谱。当前默认内部能量是可替换的稳定起点；
具体体系仍需选择适当相互作用和数据，再做精度与成本对照。
