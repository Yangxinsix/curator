# ELoRA Addon Refactor Report

## Workspace

- source branch: `kme`
- working branch: `lora`
- worktree: `/home/xinyang/curator-lora`

## Summary

这次实现把 ELoRA / cueq / cueq+ELoRA 从“模型构造参数”改成了“顶层 addon + runtime wrapper patch”。

结果是：

- 默认模型训练和推理仍然走原始 e3nn 路径
- `MACE` / `Nequip` 不再需要显式暴露 ELoRA 配置字段
- `cueq+elora` 可以通过本地 wrapper 组合实现，不需要改外部 `cuequivariance` 源码

## Main Code Changes

### 1. Top-level addon config surface

新增：

- `curator/configs/addon/elora.yaml`
- `curator/configs/addon/cueq.yaml`
- `curator/configs/addon/cueq_elora.yaml`

修改：

- `curator/configs/train.yaml`

效果：

- 用 `addon=...` 显式开启 wrapper stack
- 不再把 addon 配置塞进 `model.representation.*`

### 2. Wrapper runtime and patch factory

新增目录：

- `curator/layer/wrappers/`

核心文件：

- `curator/layer/wrappers/config.py`
- `curator/layer/wrappers/patch.py`
- `curator/layer/wrappers/elora.py`
- `curator/layer/wrappers/cueq_elora.py`
- `curator/layer/wrappers/registry.py`

实现内容：

- `WrapperConfig` 和 stack 规范化
- `apply_wrappers()` runtime patch 工厂
- addon metadata 导出与恢复
- LoRA merge helper
- ELoRA / cueq / cueq+ELoRA backend 分发

### 3. Compatibility facade

修改：

- `curator/layer/_cuequivariance_wrapper.py`

效果：

- 保留旧 import 面
- 实际 backend 决策转发到 `wrappers/`
- wrapper config 改为上下文作用域，而不是普通全局单例

### 4. Representation/model rebuild hooks

修改：

- `curator/model/base.py`
- `curator/model/mace.py`
- `curator/model/nequip.py`

新增能力：

- `Representation.export_init_kwargs()`
- `NeuralNetworkPotential.clone_with_representation(...)`

效果：

- wrapper patch 不再依赖把 ELoRA 字段塞回 representation constructor
- model rebuild 更显式

### 5. Load / checkpoint path

修改：

- `curator/utils.py`
- `curator/cli.py`
- `curator/model/lit_module.py`

效果：

- checkpoint 保存 `wrapper_params`
- `load_trained_model()` 会在 full-model / state_dict 路径上都恢复 addon metadata
- CLI 会把 `cfg.addon` 和 checkpoint wrapper metadata 合并后应用

### 6. Optimizer-group fix

修改：

- `curator/model/lit_module.py`

修复：

- LoRA 参数不再同时出现在 base parameter groups 和 `elora` group 里
- 避免 optimizer 初始化时报 duplicate parameter error

### 7. cueq internal-weight fix

修改：

- `curator/layer/wrappers/cueq_elora.py`

修复：

- `cuequivariance_torch` 的 internal-weight operator 不允许高层 `forward(weight=...)`
- wrapper 现在会直接走 cueq 底层 `f(...)`，用 `base.weight + delta` 作为 effective weight

这一步是让 `cueq+elora` 真正能跑训练的关键修复。

### 8. Tests migrated to addon API

修改：

- `test/test_read_user_config.py`
- `test/test_elora_wrappers.py`

新增：

- `test/test_elora_addon.py`

测试重点：

- addon config 不污染 representation schema
- `apply_wrappers()` metadata round-trip 正确
- `elora` parameter group 可见
- `cueq+elora` 结构级 smoke 正常

## Validation

### Unit tests

执行：

```bash
PYTHONPATH=/home/xinyang/curator-lora pytest -q \
  test/test_elora_addon.py \
  test/test_elora_wrappers.py \
  test/test_read_user_config.py
```

结果：

- `12 passed`

### Smoke training

使用：

- dataset: `/home/xinyang/curator/test/LiFePO4.traj`
- base model: `/home/xinyang/curator-lora/tmp/mace-128-L1-curator.pt`

#### Baseline

执行：

- no addon
- `num_train=2`
- `num_val=1`
- `limit_train_batches=1`

结果：

- 训练成功完成 1 epoch
- run path: `/home/xinyang/curator-lora/runs/smoke-addon-baseline`

#### ELoRA

执行：

- `addon=elora`
- `num_train=4`
- `num_val=2`
- `limit_train_batches=2`

结果：

- 训练成功完成 1 epoch
- run path: `/home/xinyang/curator-lora/runs/smoke-addon-elora`

#### cueq + ELoRA

执行：

- `addon=cueq_elora`
- CUDA + `cuequivariance_torch`
- `num_train=4`
- `num_val=2`
- `limit_train_batches=2`

结果：

- 训练成功完成 1 epoch
- run path: `/home/xinyang/curator-lora/runs/smoke-addon-cueq-elora`

备注：

- 首个 cueq+ELoRA batch 很慢，主要是 cueq kernel/graph 的首轮开销
- 修复 `cueq_elora.py` 之后，训练可以正常跑通

## Behavior Impact

### What changes when addon is absent

没有变化。

当前 baseline smoke 仍然能正常训练，说明默认路径没有被 wrapper 逻辑显式改变。

### What changes when addon is present

- `addon=elora`: e3nn backend + LoRA adapters
- `addon=cueq`: cueq backend
- `addon=cueq_elora`: cueq backend + LoRA adapters

## Remaining Follow-ups

- `curator/utils.py` 里的 e3nn/cueq 权重迁移还可以继续拆进 `wrappers/adapters/`
- 旧的 legacy wrapper kwargs 兼容分支还在 `MACE` / `Nequip` 中，可在后续完全迁移后删除
