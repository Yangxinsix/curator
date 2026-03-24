# CURATOR Wrapper Stack 方案：`cueq`、`elora` 与 `cueq+elora`

## 1. 背景

当前 `curator` 的等变算子切换主要依赖单个文件：

- `curator/layer/_cuequivariance_wrapper.py`

它承担了两件事：

1. 在 `e3nn` 与 `cuequivariance_torch` 之间切换后端。
2. 为 `MACE` / `NequIP` 等上层 layer 提供统一构造入口。

现在如果要引入 `ELoRA`，并且后续支持：

- 仅 `cueq`
- 仅 `elora`
- `cueq + elora`

继续把所有逻辑堆在一个 `_cuequivariance_wrapper.py` 文件里会越来越难维护。更合理的方向是把 wrapper 体系拆成一个可组合的子目录，并把现有 `cueq` 逻辑迁移进去。


## 2. 目标

本方案的目标是把 wrapper 体系整理成一个清晰、可扩展的结构，使得后续能够逐步支持：

1. 默认 `e3nn` 路径，不改变当前训练和推理行为。
2. `cueq` 加速路径，保持当前语义。
3. `elora` 路径，在 `e3nn` 后端上实现 LoRA/ELoRA 微调。
4. `cueq+elora` 路径，在 `cueq` 后端上叠加 LoRA/ELoRA。

非目标：

- 第一阶段不追求一次性让所有表示学习模型都支持 `cueq+elora`。
- 第一阶段不追求修改外部 `cuequivariance_torch` 或 `e3nn` 安装包源码。
- 第一阶段不追求把所有 call site 都重写；优先通过兼容 facade 降低改动面。


## 3. 设计原则

### 3.1 默认行为不变

在没有显式开启 wrapper 的情况下：

- 现有训练行为不变
- 现有推理行为不变
- 现有 checkpoint 加载行为不变

### 3.2 逻辑上允许“堆叠”，实现上优先“组合后端”

用户配置上可以表达：

- `wrapper_stack: []`
- `wrapper_stack: [cueq]`
- `wrapper_stack: [elora]`
- `wrapper_stack: [cueq, elora]`

但代码实现上，不建议真的在每个模块对象外面套多层 Python decorator/wrapper。

更稳妥的实现方式是：

- 用 `wrapper_stack` 描述用户意图
- 在 registry/factory 层把它解析成一个“组合后端”

例如：

```python
() -> E3NNBackend
("cueq",) -> CueqBackend
("elora",) -> ELoRABackend(base="e3nn")
("cueq", "elora") -> CueqELoRABackend(base="cueq")
```

这样做的好处是：

- 上层 layer 不需要知道 wrapper 是怎么叠的
- `cueq+elora` 可以按需要写成专门实现，而不是强行做任意顺序的通用嵌套
- 后续如果发现某些算子不适合双重封装，也可以局部特判

### 3.3 保留一个兼容 facade

为了降低重构风险，建议保留：

- `curator/layer/_cuequivariance_wrapper.py`

但把它变成一个兼容 facade，而不是继续承载全部实现。这样：

- 现有 import 基本不用立刻改
- 新逻辑都进入 `curator/layer/wrappers/`
- 迁移可以分阶段完成


## 4. 建议目录结构

建议新增目录：

```text
curator/layer/wrappers/
  __init__.py
  config.py
  registry.py
  base.py
  cueq.py
  elora.py
  cueq_elora.py
  mlp.py
  utils.py
```

各文件职责建议如下。

### 4.1 `config.py`

保存 wrapper 相关配置对象与全局上下文，例如：

- `WrapperConfig`
- `ELoRAConfig`
- `WrapperContext`

建议这里不要放大量实现，只放：

- 配置标准化
- wrapper stack 归一化
- 兼容旧配置字段，例如 `use_cueq`

### 4.2 `registry.py`

负责把 `wrapper_stack` 解析成具体 backend。

核心职责：

- 维护 backend registry
- 根据 `wrapper_stack` 返回 backend 实例
- 提供统一构造函数，例如：
  - `build_linear(...)`
  - `build_tensor_product(...)`
  - `build_fully_connected_tensor_product(...)`
  - `build_symmetric_contraction(...)`
  - `build_scalar_mlp(...)`

### 4.3 `base.py`

定义 backend 协议和默认 `e3nn` 实现。

建议定义一个抽象接口，例如：

```python
class OperatorBackend:
    def make_linear(...)
    def make_tensor_product(...)
    def make_fully_connected_tensor_product(...)
    def make_symmetric_contraction(...)
    def make_scalar_mlp(...)
```

`E3NNBackend` 作为默认实现，直接返回当前 `e3nn` / 本地模块版本。

### 4.4 `cueq.py`

实现纯 `cueq` 后端。

职责：

- 封装当前 `_cuequivariance_wrapper.py` 中已有的 `cueq` 逻辑
- 返回：
  - `cuet.Linear`
  - `cuet.ChannelWiseTensorProduct`
  - `cuet.FullyConnectedTensorProduct`
  - `CuetSymmetricContractionWrapper`

这一层的目标是和当前行为完全等价。

### 4.5 `elora.py`

实现纯 `e3nn + ELoRA` 后端。

职责：

- 提供 `ELoRALinear`
- 提供 `ELoRAFullyConnectedTensorProduct`
- 如有必要，提供 `ELoRATensorProduct`
- 提供 `ELoRAFullyConnectedNet` 或其构造工厂
- 提供本地 `SymmetricContraction` 的 ELoRA 版本
- 提供 `merge_elora_()`、`iter_elora_parameters()` 等辅助函数

### 4.6 `cueq_elora.py`

实现组合后端：`cueq + ELoRA`

职责：

- 提供 `CueqLoRALinear`
- 提供 `CueqLoRAFCTP`
- 提供 `CueqLoRASymmetricContraction`
- 如确有必要，再补 `CueqLoRAChannelWiseTensorProduct`

这里的重点是：

- 不要求改 `cuequivariance_torch` 源码
- 而是在 `curator` 里对子类/包装类进行扩展

### 4.7 `mlp.py`

统一管理 “标量 MLP / radial MLP / readout MLP” 的构造。

原因是当前 `MACE` 的一部分关键参数不走 `Linear/FCTP` 工厂，而是直接用：

- `e3nn.nn.FullyConnectedNet`

如果不把这条路径纳入 wrapper 系统，`MACE` 的 ELoRA 就会不完整。

### 4.8 `utils.py`

放 wrapper 共用辅助函数，例如：

- LoRA 权重初始化
- merge/unmerge
- 权重形状校验
- checkpoint key 兼容工具
- backend capability 检查


## 5. 与现有文件的关系

### 5.1 `_cuequivariance_wrapper.py` 的角色

建议保留这个文件，但改成 facade：

```python
from curator.layer.wrappers.registry import (
    build_linear,
    build_tensor_product,
    build_fully_connected_tensor_product,
    build_symmetric_contraction,
)
```

然后继续对外暴露原来的名字：

- `Linear`
- `TensorProduct`
- `FullyConnectedTensorProduct`
- `SymmetricContractionWrapper`

这样上层调用点暂时不需要大改。

### 5.2 `curator/layer/__init__.py`

后续如果 wrapper 目录稳定，可以按需导出一些工具，例如：

- `merge_model_elora_`
- `set_wrapper_context`

但第一阶段不必在 `__init__.py` 暴露太多内容。


## 6. wrapper stack 的配置模型

建议新增统一配置，而不是继续只靠 `use_cueq: bool`。

示意：

```yaml
representation:
  wrapper_stack: []
  wrapper:
    elora:
      enabled: false
      rank: 16
      alpha: 16.0
      train_radial: true
      train_symmetric: true
      merge_for_export: false
```

兼容策略：

- 如果配置里只有 `use_cueq: true`
  - 自动转换为 `wrapper_stack: [cueq]`
- 如果 `wrapper_stack` 明确给出
  - 以 `wrapper_stack` 为准

这样现有配置仍可工作，同时为后续扩展留下空间。


## 7. MACE 的特殊处理

`MACE` 不能只依赖统一算子工厂，还需要额外处理两类本地模块。

### 7.1 `conv_tp_weights` / radial MLP

`MACE` 的主消息传递里，`conv_tp` 的权重是由外部 MLP 生成的，而不是 `TensorProduct` 自己持有的内部参数。

对应位置：

- `curator/layer/_mace_interaction.py`

这意味着：

- `ELoRA` 不能只 patch `TensorProduct`
- 还必须把 `FullyConnectedNet` 这一条路径也纳入 wrapper/MLP 工厂

建议：

- 新增 `build_scalar_mlp(...)`
- 在 `_mace_interaction.py` 中改为通过 wrapper factory 构建 `conv_tp_weights`

### 7.2 `SymmetricContraction`

`MACE` 的 many-body 部分使用本地拷贝的：

- `curator/layer/_symmetric_contraction.py`

这部分不属于 `e3nn`，因此：

- 纯 `ELoRA` 版本需要 patch 本地 `SymmetricContraction`
- `cueq+elora` 版本需要 patch `cuet.SymmetricContraction` 的本地子类/包装类

建议把 “对称收缩 + LoRA” 明确作为单独任务处理，不要混在一般 `Linear` wrapper 里。

### 7.3 `TensorProduct` 的优先级

对于 `MACE` 来说，`TensorProduct` 本身并不是第一优先级。

原因：

- 当前 `conv_tp` 使用的是 `internal_weights=False`
- 实际可训练权重来自 `conv_tp_weights`

因此在第一版 `MACE + ELoRA` 中：

- 可以优先覆盖 `Linear`
- 覆盖 `FullyConnectedTensorProduct`
- 覆盖 `SymmetricContraction`
- 覆盖 `FullyConnectedNet`

`ChannelWiseTensorProduct` 的 `cueq+elora` 支持可以放到后续阶段，除非 benchmark 证明它是主要瓶颈。


## 8. 推荐的 backend 责任边界

### 8.1 `E3NNBackend`

负责：

- 默认 `o3.Linear`
- 默认 `o3.TensorProduct`
- 默认 `o3.FullyConnectedTensorProduct`
- 默认本地 `SymmetricContraction`
- 默认标量 MLP

### 8.2 `CueqBackend`

负责：

- `cuet.Linear`
- `cuet.ChannelWiseTensorProduct`
- `cuet.FullyConnectedTensorProduct`
- `cuet.SymmetricContraction`

不负责：

- LoRA 参数
- merge 行为

### 8.3 `ELoRABackend`

负责：

- `e3nn` 路径上的 LoRA 增强
- 本地 MACE 对称收缩的 LoRA 增强
- MLP/radial 路径上的 LoRA 增强

### 8.4 `CueqELoRABackend`

负责：

- `cueq` 路径上的显式权重算子 + LoRA
- `cueq` 版 `SymmetricContraction` + LoRA
- 同时继续管理 MLP/radial 路径上的 LoRA

这个 backend 是“组合后端”，不是简单把 `CueqBackend` 和 `ELoRABackend` 机械相加。


## 9. 参数组与微调模式

当前 `curator` 已经有成熟的参数组机制和 fine-tune preset：

- `curator/configs/finetune/full.yaml`
- `curator/configs/finetune/head_only.yaml`
- `curator/configs/finetune/multi_domain.yaml`

引入 ELoRA 后，建议新增以下参数组：

- `elora`
- `radial`
- `symmetric_contractions`

可选地再拆：

- `elora_readout`
- `elora_representation`

推荐策略：

1. 不要照搬 upstream MACE 的“每 step 动态改 `requires_grad`”。
2. 直接让 `MACE.parameter_groups()` 返回清晰的 LoRA 相关组。
3. fine-tune preset 通过 `optimizer_groups` 和 `freeze_schedule` 控制哪些组训练。

例如：

- `full.yaml`
  - 全参数训练，不启用 `elora`
- `head_only.yaml`
  - 仅 readout / domain head
- `elora.yaml`
  - 仅 `elora + radial + symmetric_contractions + readout`
- `multi_domain.yaml`
  - `readout_domains + output_domains + 可选 elora`

对于 multi-domain，需要明确一条原则：

- `ELoRA` 参数默认属于共享表示层，不属于 domain-specific readout
- `readout_domains` / `output_domains` 仍然单独控制


## 10. 默认行为与兼容性要求

### 10.1 默认关闭

如果没有显式配置 wrapper：

- 行为应与当前主分支一致

### 10.2 旧配置兼容

已有配置里的：

- `use_cueq: false/true`

应该仍然有效，并在内部被翻译成新的 wrapper stack 语义。

### 10.3 旧 checkpoint 兼容

要求：

- `use_elora=false` 时，旧 checkpoint 应可直接加载
- `use_elora=true` 时，加载旧 checkpoint 允许新增 LoRA 参数缺失

建议：

- 在 `update_model()` 中加入 wrapper 相关配置升级
- 或在新模块加载时统一使用兼容逻辑

### 10.4 导出行为

建议区分两类模型状态：

1. 训练态：保留 LoRA 参数
2. 导出态：执行 `merge_elora_()` 后导出纯权重模型

这有利于：

- 保持训练灵活性
- 保持部署推理路径简单


## 11. 推荐实施顺序

### 阶段 1：仅重构 wrapper 基础设施，不改行为

目标：

- 建立 `curator/layer/wrappers/`
- 把 `_cuequivariance_wrapper.py` 变成 facade
- 确保 `wrapper_stack=[]` 与 `wrapper_stack=[cueq]` 行为与当前一致

阶段产物：

- 无功能变化
- 只是代码组织变化

### 阶段 2：支持 `e3nn + elora`

目标：

- 完成 `ELoRABackend`
- 覆盖 `Linear`、`FCTP`、本地 `SymmetricContraction`
- 把 `MACE` 的 `FullyConnectedNet` 接入 MLP factory

阶段产物：

- `MACE + ELoRA`
- 默认先不支持 `cueq+elora`

### 阶段 3：补全参数组和 fine-tune 预设

目标：

- 新增 `elora` / `radial` / `symmetric_contractions` 参数组
- 新增 `configs/finetune/elora.yaml`
- 让 `multi_domain` 可以和 `elora` 组合

### 阶段 4：支持 `cueq+elora`

目标：

- 实现 `CueqELoRABackend`
- 先覆盖：
  - `Linear`
  - `FullyConnectedTensorProduct`
  - `SymmetricContraction`
- `ChannelWiseTensorProduct` 根据实际需要决定是否纳入第一版

### 阶段 5：导出、merge 与测试补齐

目标：

- `merge_elora_()`
- checkpoint round-trip
- 推理数值一致性
- regression tests


## 12. 测试建议

至少需要覆盖以下测试。

### 12.1 无 wrapper 回归测试

- `wrapper_stack=[]` 与当前输出一致

### 12.2 `cueq` 回归测试

- `wrapper_stack=[cueq]` 与当前 `use_cueq=True` 行为一致

### 12.3 `elora` 零增量一致性测试

- LoRA 初始化为零或 merge 前后
- 输出应与 base 模型一致或在容差内一致

### 12.4 `merge` 一致性测试

- merge 前后推理结果一致

### 12.5 `MACE` 专项测试

- `conv_tp_weights` 参与训练
- `symmetric_contractions` 参与训练
- `BesselBasis(trainable=True)` 的径向参数可训练

### 12.6 multi-domain 集成测试

- `readout_domains` 与 `elora` 参数组能同时存在
- 冻结计划不会误冻结 LoRA 参数


## 13. 最终建议

这套方案里，“双重 wrapper” 建议保留为用户和配置层的表达方式，但代码实现上应当落成“组合 backend”。

推荐结论如下：

1. 在 `curator/layer/` 下新增 `wrappers/` 子目录是合理的。
2. 现有 `_cuequivariance_wrapper.py` 不要立即删除，而应改成兼容 facade。
3. `ELoRA` 应先在 `e3nn` 路径上做通，再扩展到 `cueq`。
4. `cueq+elora` 不需要直接修改 `cuequivariance_torch` 源码，但需要在 `curator` 本地实现对应子类/包装类。
5. `MACE` 的完整支持一定要包括：
   - 算子 wrapper
   - `FullyConnectedNet` / radial MLP
   - `SymmetricContraction`
   - 参数组与 fine-tune preset

如果按风险和收益排序，建议先落地：

1. wrapper 基础设施重构
2. `e3nn + elora`
3. `MACE` 参数组与 preset
4. `cueq+elora`

