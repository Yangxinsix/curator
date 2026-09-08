# ACVR 训练接口修复与小规模验证记录

日期：2026-09-07。代码基线：`acvr-unified-framework` / `2da307c` 加当前工作区。
本次保留工作区原有训练、指标和文档修改；新增修复与试验未提交。

本记录接续 [项目状态报告](PROJECT_STATUS_AND_VALIDATION_REPORT_20260907.md)。
它验证正式训练生命周期和学习型合成 oracle；不代表已经完成真实 DFT 数据训练、通用物理插件或生产部署。
可执行步骤见 [小规模训练说明](small_training.md)。

## 接口修复

1. 两种 ACVR 输出共享训练协议，支持 `subset`、`loss_weight` 及按阶段累计／清理。
2. train／val／test 指标隔离；sanity validation 不污染训练指标；checkpoint 不恢复过期累计量。
3. 多条件 loss 内部已经处理的几何权重和物理尺度不再被训练器重复施加。
4. 缺失可选标签和禁用项始终贡献零损失、零分母、零梯度，不再复用此前累积的非零 loss。
5. 独立 `Trainer.validate/test` 可以初始化输出；修复无 rescale 层时测试变量未定义和错误返回键。
6. `val_total_loss_exact` 按各分项自己的分母聚合，成为示例 checkpoint 的监控指标。
7. checkpoint 保存解析后的模型／数据配置；恢复兼容包含模型和配置对象的 Curator checkpoint。
8. 评估入口可实例化配置中的 Dataset；嵌套多条件 tensor 递归搬运，整数索引和 mask 保持类型。
9. 独立 variational evaluator 复用物理 loss 的单位、语义、缺失标签和几何权重合同，并保存可重算的原始记录。
10. oracle checkpoint 保存静态物理语义和完整数据 manifest；独立评估拒绝不同 seed／划分及被修改的单位或布局。

## 实际实验

输出目录：`runs/acvr_training_validation_20260907/`。

- CPU、float64、单设备、无 EMA；网络 seed 为 4，数据 seed 为 17。
- 64/16/16 个训练／验证／测试几何；每几何 5 条件，共 320/80/80 条件。
- 物理阶段：从实体间距离学习二次泛函，100 epoch、400 次优化；监督能量、力、内部状态和完整场响应。
- proposal 阶段：从物理 checkpoint 初始化，冻结物理参数，100 epoch、400 次优化；不使用测试几何生成训练 teacher。
- 测试几何位于训练距离范围内；测试场幅度也位于训练范围内，包含未见过的幅度以及共享零场基线。
- 内变量为两个实体各两个标量通道；边界为固定共轭场，无非平凡等式约束。不能据此声称跨边界或第二物理变量已验证。

物理验证总 loss 从 `0.0823453292` 降至约 `1.95e-5`。
学习型泛函的独立测试集结果（采用精确内部变量解）：

| 指标 | RMSE | 单位 |
|---|---:|---|
| 能量 | 0.0003249008 | eV |
| 力的分量 | 0.0007220040 | eV/Angstrom |
| 内部状态 | 0.002015672 | 无量纲 |
| 完整场响应的矩阵元素 | 0.004557253 | 1/eV |

数据和单位为合成 oracle 的约定；这些数值不能当作真实材料的 DFT 精度。
响应项由归一化 MSE `2.0768551214892278e-5` 换算，当前所有响应的尺度均为 `1`。

训练前后的 proposal 均针对同一学习型泛函的精确解评估，不能把该误差等同于对解析物理真值的误差：

| 指标 | 未训练 proposal | 训练后 proposal |
|---|---:|---:|
| 修正前状态 RMSE | 0.2408246671 | 0.0021534717 |
| 归一化驻定残差中位数 | 0.4711638540 | 0.0043082148 |
| 归一化驻定残差 P95 | 0.7231155336 | 0.0098328310 |
| 最大归一化驻定残差 | 0.8224163991 | 0.0102295954 |

真实调用 `solve_amortized(max_correction_steps=1, fallback=True)` 的结果：

| 残差容限 | 未训练零修正率 | 训练后零修正率 | 训练后平均修正次数 | fallback 率 |
|---|---:|---:|---:|---:|
| `1e-2` | 0% | 98.75%（79/80） | 0.0125 | 0% |
| `1e-3` | 0% | 0% | 1 | 0% |
| `1e-4` | 0% | 0% | 1 | 0% |
| `1e-8` | 0% | 0% | 1 | 0% |

各策略均无 unreliable 结果。严格容限修正后的状态 RMSE 约为 `2.58e-17`。
这证明当前 proposal 学到了更好的初猜；默认严格容限下仍需稠密修正，不能据此宣称省掉了求解或获得了端到端加速。

验收脚本另已确认：物理参数逐 tensor 完全相同，6 个 proposal 参数 tensor 均有更新，
基线 proposal 与初始化完全一致，两个 checkpoint 的数据元信息相同。
从 proposal checkpoint 独立重新评估得到的物理指标，与物理 checkpoint 的结果完全一致。
两个最终 checkpoint 还分别在新建的 `Trainer` 中完成独立 `test`；
物理阶段 `test_total_loss_exact` 为 `2.5458333766e-5`，
proposal 阶段为 `6.9035972956e-6`，所有返回指标均有限。

本次产物：

- `runs/acvr_training_validation_20260907/physical/model_path/last.ckpt`
- `runs/acvr_training_validation_20260907/proposal/model_path/last.ckpt`
- `runs/acvr_training_validation_20260907/physical_evaluation/metrics.json`
- `runs/acvr_training_validation_20260907/proposal_evaluation/metrics.json`
- `runs/acvr_training_validation_20260907/test_stage_metrics.json`：两个最终 checkpoint 的独立 `Trainer.test` 指标。
- `runs/acvr_training_validation_20260907/validation/summary.json`：完整验收结果、checkpoint hash、数据划分和容限扫描。
- `runs/acvr_training_validation_20260907/validation/physical_holdout/results.pt`：可重算的原始记录。

复现本次运行使用：

```bash
cd /workspace/work/agent-curator/curator

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
../.venv/bin/curator-train --config-name acvr_oracle \
  run_path=runs/acvr_training_validation_20260907/physical

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
../.venv/bin/curator-train --config-name acvr_oracle_proposal \
  run_path=runs/acvr_training_validation_20260907/proposal \
  model_path.path=runs/acvr_training_validation_20260907/physical/model_path/last.ckpt

../.venv/bin/python example/acvr/check_oracle_checkpoint.py \
  --physical runs/acvr_training_validation_20260907/physical/model_path/last.ckpt \
  --proposal runs/acvr_training_validation_20260907/proposal/model_path/last.ckpt \
  --output runs/acvr_training_validation_20260907/validation
```

已有结果目录应保留；重跑时可统一替换为新目录。

## 自动验证

最终定向回归：`214 passed, 72 warnings in 57.79s`，无跳过。
日志：`runs/acvr_training_validation_20260907/regression.txt`。
范围包括已有 ACVR、QEq、Ewald、SPICE 测试，真实本地 SPICE／PySCF／TRECI 读取，
LAMMPS KSpace 算符对照，以及新增训练、数据、评估、checkpoint 回归。
这不是全仓库测试，也不是 GPU／MPI 性能测试。

CLI 生命周期回归使用 8/2/2 几何、每几何 3 条件，实际验证：

- 正式入口训练 2 epoch，保存模型、输出、优化器和数据随机数状态。
- 去除完整模型对象后，仅通过保存的配置与 state dict 重建，逐 tensor 与原模型相同。
- 恢复到第 3 epoch，与连续训练 3 epoch 的模型和 Adam 状态逐 tensor 完全相等。
- proposal 阶段确实更新 proposal，物理参数逐 tensor 完全不变。
- 通过独立评估 CLI 加载 checkpoint，在测试集产生指标和原始记录。
- 分阶段指标、缺失标签、单位／manifest 篡改拒绝，以及标签污染不改变预测。

## 适用范围

本次完成了可学习合成模型的正式训练、保存、重建、恢复及独立评估流程。
当前 proposal 认证和修正仍使用稠密参考路径，未测量端到端加速。
真实 DFT 数据的科学验证、非平凡约束下学习的完整示例、其他物理插件、EMA 恢复、MPI／GPU 和 MD 均不属于本次验收。
