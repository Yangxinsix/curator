# ACVR 小规模完整训练

本示例通过现有 `curator-train`、`LitNNP` 和 `curator-evaluate` 运行。
它先从结构学习二次泛函，再冻结物理参数、训练独立 proposal。
数据为可重建的解析合成数据，不是 SPICE、TRECI 或其他 DFT 训练结果。
实际跑数与验收范围见 [2026-09-07 验证记录](TRAINING_VALIDATION_20260907.md)。

## 示例范围

- 64/16/16 个训练、验证、测试几何；每个几何对应 5 个场条件。
- 两个实体、每个实体两个内部标量；输入结构特征是实体间距离及其平方。
- 网络学习 `H(R)`、`g(R)`、`c(R)`；以正定构造保证泛函稳定。
- 物理监督包含能量、力、内部状态和完整 4×4 场响应，标签由独立解析公式生成。
- proposal 网络仅使用几何特征和边界场，teacher 来自当前学习型泛函的精确解。
- 验证／测试几何未参与训练，场幅度位于训练范围内，部分场数值也未见过。
- 边界类型为固定共轭场；本例没有非平凡等式约束，不构成跨边界类型泛化或第二物理插件验证。

默认使用 CPU、float64、单设备，不启用 EMA。数据划分、单位、gauge、变量布局随 checkpoint 保存。
独立评估会拒绝与训练配置不一致的数据元信息。相同几何的全部条件始终属于同一数据划分。

## 运行物理模型训练

以下命令使用当前工作区已有环境。其他环境应替换 Python/CLI 路径。

```bash
cd /workspace/work/agent-curator/curator

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
../.venv/bin/curator-train --config-name acvr_oracle
```

配置：`curator/configs/acvr_oracle.yaml`。默认运行 100 个 epoch，输出到
`runs/acvr_oracle/physical/`，包含配置、训练日志、CSV 指标、最佳及最后一个 checkpoint。
checkpoint 以 `val_total_loss_exact` 选择，该指标逐项累加分子／分母后求值，
不会把大小不同或标签覆盖不同的 batch loss 简单平均。

只检查完整生命周期时，可以缩小数据和 epoch 数：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
../.venv/bin/curator-train --config-name acvr_oracle \
  run_path=runs/acvr_tiny/physical trainer.max_epochs=2 \
  data.num_train=8 data.num_val=2 data.num_test=2 data.batch_size=4
```

## 训练 proposal

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
../.venv/bin/curator-train --config-name acvr_oracle_proposal
```

此配置默认读取 `runs/acvr_oracle/physical/model_path/last.ckpt` 的模型权重，
重新建立 proposal 阶段优化器，冻结物理参数，运行 100 个 epoch。
它不接续物理阶段的 Adam 状态，因为优化对象已经改变。
如物理训练更改了数据规模、条件数或 seed，proposal 配置必须使用相同数据参数。

自定义物理 checkpoint 时：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
../.venv/bin/curator-train --config-name acvr_oracle_proposal \
  model_path.path=/absolute/path/to/physical/model_path/last.ckpt \
  run_path=runs/my_proposal
```

## 恢复中断的训练

恢复同一阶段时使用 `mode: resume`；`trainer.max_epochs` 是恢复后的总 epoch 上限。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
../.venv/bin/curator-train --config-name acvr_oracle \
  'model_path={path: runs/acvr_oracle/physical/model_path/last.ckpt, mode: resume}' \
  run_path=runs/acvr_oracle/physical_resumed trainer.max_epochs=102
```

proposal 阶段恢复时，将 `--config-name` 改为 `acvr_oracle_proposal`，
并指定 proposal checkpoint。数据和网络配置须保持一致。
恢复会载入模型、优化器、训练进度和训练数据随机数状态；阶段指标重新累计。

## 独立评估与验收

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
../.venv/bin/curator-evaluate --config-name acvr_oracle_evaluate

../.venv/bin/python example/acvr/check_oracle_checkpoint.py \
  --physical runs/acvr_oracle/physical/model_path/last.ckpt \
  --proposal runs/acvr_oracle/proposal/model_path/last.ckpt \
  --output runs/acvr_oracle/validation
```

独立评估读取保存的模型，输出 `metrics.json` 和保留标签、预测、mask、指针及边界信息的 `results.pt`。
当训练使用自定义数据参数时，评估的 `datapath.num_train`、`num_val`、`num_test`、`num_conditions`、`seed`
也要匹配；只修改测试集 seed 会被拒绝。

验收脚本从 checkpoint 的配置和 state dict 重建模型，不依赖保存的完整模型对象。
它自动复用训练数据配置，验证 proposal 阶段的物理参数完全未变，
并比较未训练／训练后的 proposal 在测试集上的修正前误差、残差分布及不同容限下的接受率、修正次数和回退率。
结果保存在 `validation/summary.json`。

零修正接受率不等于端到端加速。当前认证与修正仍包含稠密参考运算；
本示例没有验证真实 DFT 精度、大规模性能、MPI／GPU 或 MD。

## 自动回归

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
../.venv/bin/python -m pytest -q \
  test/test_acvr_training_lifecycle.py \
  test/test_acvr_output_lifecycle.py \
  test/test_acvr_evaluator.py \
  test/test_variational_oracle.py
```

其中 CLI 集成测试验证两轮物理训练、checkpoint 配置独立重建、恢复到第三轮、
与连续三轮训练的模型及 Adam 状态逐 tensor 相等、proposal 阶段参数隔离，以及独立测试集评估。
这是本示例和当前无 EMA 配置的验证，不是所有 Curator 训练模式的验收。
