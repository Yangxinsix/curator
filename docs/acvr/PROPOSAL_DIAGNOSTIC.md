# 固定泛函 proposal 诊断

[诊断脚本](../../scripts/acvr_proposal_diagnostic.py)只训练 proposal，冻结骨干、基线和物理泛函。历史逐原子 MLP 使用 `--communication baseline`；新接口支持 `pointwise`、`mp`、`attention`、`ewald`。这是独立的求解能力诊断，正式训练仍使用原有联合训练入口。

2026-09-11 的七项正式对照均已完成 600 步，使用同任务匹配的物理参数、公共初始化和数据顺序。所有候选尚未通过完整响应门槛；结果、失败分母与初始/最终响应核验见 [实验报告](/workspace/data/work/agent-curator/experiments/acvr_proposal_global_20260911/RESULTS.md)。

无 `--model-path` 时，脚本要求周期电荷配置，使用数值数据的固定电负性、硬度、常数及相同 Ewald 核，基线置零。有 `--model-path` 时，复用检查点中的骨干、基线和泛函，仅重新初始化 proposal，并创建新的 Adam；不恢复旧优化器，也不声称是完整训练恢复。

从仓库根目录启动一个候选：

```bash
../.venv/bin/python scripts/acvr_proposal_diagnostic.py \
  --communication attention --output runs/diagnostic_attention \
  --steps 600 --seed 17 --train-size 128 --val-size 32
```

输出目录必须尚不存在，避免覆盖旧实验。用 `--config /path/config.yaml --model-path /path/model.ckpt` 切换到检查点对应的数据和泛函；例如非周期 QM/MM 只比较 baseline、pointwise、mp、attention。脚本读取配置中的数据种子和划分，`--seed` 单独控制 proposal 初始化、子集采样和训练顺序。可重复 `--override data.num_train=...` 等 Hydra 参数；改变数据生成配置会产生不同诊断任务，不能作为同一任务的性能比较。

CPU 两步接口检查示例：

```bash
../.venv/bin/python scripts/acvr_proposal_diagnostic.py \
  --communication pointwise --output runs/diagnostic_smoke \
  --steps 2 --train-size 1 --val-size 1 --batch-size 1 \
  --hidden-features 4 --layers 1 --heads 1 --jacobian-geometries 1 \
  --override data.num_train=1 --override data.num_val=1 \
  --override data.num_test=1 --override data.plane_side=2 \
  --override model.representation.num_features=4 \
  --override model.representation.num_interactions=1 \
  --override model.output_modules.0.proposal.hidden_features=4 \
  --override model.output_modules.0.proposal.num_heads=1
```

默认 float64、Adam 学习率 0.001、梯度裁剪 10、残差权重 0.1。每个几何内部先平均所有条件的残差，再平均几何。精确状态只用于验证，训练步骤不调用参考求解器。默认总时间预算 3600 秒，可用 `--time-limit` 修改；耗尽预算和非有限损失会保留未完成状态。

新 proposal 继承配置/检查点中的 `state_scale` 和 `field_scale`；当前周期配置为 0.05 e 和 1 V。旧 MLP 检查点缺少这两个字段时，新 proposal 默认使用 1 和 1，可用 `--state-scale`、`--field-scale` 显式指定。实际数值写入迁移记录；不同分支比较时保持一致。历史 baseline 保留原 MLP 形式，不应用新接口的尺度变换。

脚本缓存冻结的几何特征与物理核，加速这一特定诊断；缓存结果会与完整模型前向核对。`training_step_seconds_mean` 因此只包括缓存之后的 proposal、驻定梯度和反向更新，不能作为端到端加速结论。报告另外记录缓存构建时间、单个验证几何的完整模型前向时间、参考求解和 Jacobian 时间；完整前向的物性导数已关闭。

输出包括：

- `config.yaml`、`arguments.json`、源码快照和哈希、`data_manifest.json`：保存配置、种子、选中几何与条件输入。周期数值任务只构建 train/val；不生成 test 标签。公共数据读取器可能载入同一源文件，但脚本不取 test 样本、不评估 test。
- `initial.json`、`result.json`：训练前后原始驻定 RMS、绝对残差 95 分位/最大值、归一化残差、约束误差、同泛函参考状态误差及能量差、完整局域场 Jacobian 误差。conq/conp 分开统计；Jacobian 同时报告自身、邻居和明确不相交分组之间的误差。
- `initial_states.pt`、`final_states.pt`：逐条件原始状态、残差、参考解和所选 Jacobian，可复算指标。参考与 Jacobian 失败、预算跳过均保留在条件总数中，单独列出原因。
- `migration.json`、`training.jsonl`、`training_history.json`、`checkpoint.pt`：记录替换范围、冻结参数哈希、每步几何顺序和实际耗时，以及诊断状态字典。该 checkpoint 是诊断审计产物，不能直接当作完整生产训练恢复文件。

训练前后都会检查冻结参数哈希。有限步数 smoke 通过只证明接口、求导和记录流程可执行，不证明候选已经满足残差、物性或效率门槛。
