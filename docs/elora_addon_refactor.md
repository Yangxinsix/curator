# ELoRA Addon Refactor Design

## 1. Goal

把 ELoRA、cueq、cueq+ELoRA 做成真正的 addon，而不是 representation schema 的一部分。

这次重构的目标是：

- 默认 `e3nn` 路径完全不变。
- 不在 `MACE` / `Nequip` 的公开配置结构里显式增加 `use_elora`、`wrapper_stack` 这类字段。
- 所有 backend/LoRA 选择都收敛到 `curator/layer/wrappers/`。
- 训练、加载、checkpoint 升级都能通过统一的 wrapper metadata 恢复 addon 状态。

## 2. Final Architecture

### 2.1 Config surface

新增顶层 Hydra 组：

- `addon=cueq`
- `addon=elora`
- `addon=cueq_elora`

入口在：

- `curator/configs/train.yaml`
- `curator/configs/addon/*.yaml`

约束：

- 默认 `defaults` 里只声明 `optional addon: null`。
- 未显式指定 `addon` 时，模型仍然走纯 `e3nn`。
- `model/representation/*.yaml` 不再携带 ELoRA/cueq 配置字段。

### 2.2 Wrapper runtime

所有后端构造入口统一走：

- `curator/layer/_cuequivariance_wrapper.py`
- `curator/layer/wrappers/registry.py`

真实实现收敛在：

- `curator/layer/wrappers/config.py`
- `curator/layer/wrappers/elora.py`
- `curator/layer/wrappers/cueq_elora.py`
- `curator/layer/wrappers/patch.py`

分工：

- `config.py`: `WrapperConfig`、stack 规范化、上下文作用域 wrapper config。
- `registry.py`: 根据 `wrapper_stack` 决定构造 e3nn / cueq / elora / cueq+elora backend。
- `patch.py`: `apply_wrappers()`、metadata 导出、LoRA merge、addon parameter groups。
- `cueq_elora.py`: cueq internal-weight 模块的 LoRA 适配。

### 2.3 Patch flow

`apply_wrappers(model, wrapper_cfg)` 的工作流：

1. 解析目标 `WrapperConfig`
2. 导出当前 representation 的 init kwargs
3. 在临时 wrapper context 内重建 representation
4. 用新 representation clone 出新的 `NeuralNetworkPotential`
5. 迁移 e3nn/cueq 之间需要特殊处理的权重
6. 复制 shape-compatible state
7. 在 model 和 representation 上附加 `_wrapper_config`

关键点：

- wrapper patch 改的是“运行时构造出的 operator backend”，不是上层模型语义结构。
- `NeuralNetworkPotential` 本身没有新增 ELoRA 专用构造参数。

## 3. Invariants

### 3.1 Default behavior isolation

以下条件同时成立时，行为必须与原来一致：

- 没有 `addon`
- checkpoint 中没有 `wrapper_params`
- 没有显式调用 `apply_wrappers()`

对应保证：

- `train.yaml` 只把 addon 作为可选顶层组。
- `get_model_wrapper_config()` 默认解析成 `e3nn`。
- `load_trained_model()` 对未带 addon metadata 的模型不会强行 patch。

### 3.2 Addon metadata is top-level, not representation schema

wrapper metadata 只应该存在于：

- `cfg.addon`
- checkpoint `wrapper_params`
- runtime attached `_wrapper_config`

而不应该再依赖：

- `cfg.model.representation.use_elora`
- `cfg.model.representation.use_cueq`
- `cfg.model.representation.wrapper_stack`

### 3.3 One owner for LoRA params

LoRA 参数只能由 addon parameter group 持有一次。

实现上：

- `collect_addon_parameter_groups()` 单独收集 `elora` 参数组
- `LitNNP._optimizer_parameter_groups()` 会先把 base groups 里的 LoRA 参数剔除，再追加 `elora`

否则会在 optimizer 初始化时报：

- `ValueError: some parameters appear in more than one parameter group`

## 4. cueq + ELoRA Combination Rule

### 4.1 What uses cueq

优先使用 cueq 的是：

- channelwise tensor product
- cueq 原生支持且不依赖内部可学习权重注入的算子

### 4.2 What uses LoRA

优先走 LoRA adapter 的是：

- `Linear`
- `FullyConnectedTensorProduct`
- `SymmetricContraction`
- MACE/NequIP 内部 radial/readout MLP

### 4.3 Why cueq needs special patching

cueq 的 `Linear` / `TensorProduct` 在 `internal_weights=True` 时不允许高层 `forward(weight=...)`。

所以 `curator/layer/wrappers/cueq_elora.py` 里需要：

- 对 internal-weight 分支直接调用 cueq 底层 `f(...)`
- 先把 `base.weight + adapter.delta()` 作为 effective weight 传进去

这也是 cueq+ELoRA 唯一真正需要“特殊适配”的位置。

## 5. Checkpoint / Loading Plan

### 5.1 Save

checkpoint 额外保存：

- `wrapper_params`

来源：

- `curator/model/lit_module.py`
- `export_wrapper_config(self.model)`

### 5.2 Load

加载优先级：

1. `cfg.addon`
2. checkpoint `wrapper_params`
3. stored model attached metadata
4. legacy checkpoint 中可推断的 wrapper 字段

实现位置：

- `curator/utils.py`
- `resolve_wrapper_config_payload(...)`
- `load_trained_model(...)`
- `curator/cli.py`

### 5.3 Compatibility

旧 checkpoint 即使没有 `wrapper_params` 也能加载。

如果用户显式提供 `addon=...`，runtime 会在加载后 patch 到目标 stack。

## 6. Representation / Model Rebuild Hooks

为了减少 `inspect.signature` 式的弱耦合重建，这次新增：

- `Representation.export_init_kwargs()`
- `NeuralNetworkPotential.clone_with_representation(...)`

当前已在：

- `curator/model/mace.py`
- `curator/model/nequip.py`
- `curator/model/base.py`

这样 `apply_wrappers()` 不需要再把 wrapper 字段塞回 representation constructor。

## 7. Validation Plan

### 7.1 Unit tests

要覆盖三类行为：

- addon config 不污染 representation schema
- `apply_wrappers()` 能产出正确 metadata 和 LoRA 参数组
- cueq+ELoRA 在 CUDA + cueq 环境下至少能完成结构级 smoke

对应测试：

- `test/test_read_user_config.py`
- `test/test_elora_addon.py`
- `test/test_elora_wrappers.py`

### 7.2 CLI smoke

用以下输入做 smoke：

- dataset: `curator/test/LiFePO4.traj`
- base model: `tmp/mace-128-L1-curator.pt`

至少验证：

- baseline
- `addon=elora`
- `addon=cueq_elora`

## 8. Remaining Follow-ups

这次已经把 addon 行为从主模型结构里抽离出来，但还有两点后续可继续收紧：

- `curator.utils` 里仍保留了 e3nn/cueq 权重迁移逻辑，后续可以再挪进 `wrappers/adapters/`
- `MACE` / `Nequip` 里仍保留了 legacy wrapper kwargs 的兼容告警，后续可以在完全迁移后删除
