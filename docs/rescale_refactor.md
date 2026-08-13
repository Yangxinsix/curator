# Rescale 重构设计

## 目标

本次重构只解决输出的 scale/shift 边界，不改变 direct-force head、Hessian
计算公式或数据集标签格式。

核心原则：

1. `GlobalRescaleShift` 是 scale/shift 配置和数值的唯一所有者。
2. `NeuralNetworkPotential`、force output 和 `DistillOutput` 不推断 scale 来源。
3. 模型严格按照 checkpoint 或配置中的 module 顺序执行，不做全局重排。
4. 只要求同一个 loss 中 prediction 和 target 使用相同单位；不要求所有输出在模型内部使用同一种归一化空间。
5. eval/deploy/offline-label 的公开输出仍使用物理单位。

## 1. `scale_forces=True` 的定义

保持当前公开行为不变。令训练集 force RMS 为

```text
s_F = RMS(F_train)
```

能量 shift 仍由当前 energy 统计或 per-species reference 逻辑得到，记为
`b_E`。启用 `scale_forces=True` 后，rescale 配置为：

| Property | Scale | Shift |
| --- | --- | --- |
| energy | `s_F` | `b_E` |
| forces | `s_F` | none |

对应关系为：

```text
E_physical = s_F * E_internal + b_E
F_physical = s_F * F_internal
```

这里 `scale_forces=True` 包含两个明确动作：

1. energy 的 scale 统计量改用 force RMS，但不改变 energy shift 的计算。
2. rescale layer 注册 forces scale，数值同样初始化为 force RMS；不注册
   forces shift。

这两个 transform 都由 `GlobalRescaleShift.setup_from_datamodule()` 创建和初始化。
`NeuralNetworkPotential` 不参与。

### Force derivatives

不为 Hessian 创建新的 `HeadConfig` 或 Hessian rescale head。

Hessian、sampled Hessian 和 projected Hessian 都是 force 对未缩放坐标的导数，
因此只继承 forces 的乘法 scale：

```text
H_physical        = s_F * H_internal
H_sampled_phys    = s_F * H_sampled_internal
H_projected_phys  = s_F * H_projected_internal
```

它们不继承 forces shift，也不需要自己的统计量。

`GlobalRescaleShift` 内部维护 forces scale 的 scale-only dependent keys：

```python
FORCE_SCALE_DEPENDENTS = (
    properties.energy_hessian,
    properties.energy_hessian_sampled,
    properties.energy_hessian_projected,
)
```

当 forces `ScaleTransform` 执行 `scale()` 或 `unscale()` 时，它对当前 data 中
存在的 dependent keys 使用同一个实时 scale tensor。这样在
`scale_trainable=True` 时也不会出现 forces 和 Hessian scale 不同步的问题。

如果一个模型没有注册 forces scale，以上 Hessian key 就保持 identity；不得隐式
fallback 到 energy scale，也不得根据 force producer 类型推断。

## 2. 官方 MACE 转 direct force

### 官方计算图

普通官方 `MACE` 直接对 total energy 求导。官方 `ScaleShiftMACE` 则先对
interaction atomic energy 执行 `scale_shift`，再对缩放后的 interaction energy
求导：

- [普通 MACE force/Hessian](https://github.com/ACEsuit/mace/blob/22f0809735bd4dd1deba80cf8e16f89913a35ff4/mace/modules/models.py#L397-L414)
- [ScaleShiftMACE force/Hessian](https://github.com/ACEsuit/mace/blob/22f0809735bd4dd1deba80cf8e16f89913a35ff4/mace/modules/models.py#L576-L596)

因此官方 `ScaleShiftMACE` 的 force tensor 没有经过一个 post-force rescale
module，但 force 自然继承了 interaction-energy 的乘法 scale。energy shift 的导数
为零。

### 决策：不创建额外 ForceRescale

官方 `ScaleShiftMACE` 转换为 direct force 时：

1. 保留 checkpoint 原有 energy rescale/shift 及其位置。
2. 用 `DirectForceOutput` 替换 gradient force producer。
3. direct-force head 直接学习物理单位 forces。
4. 从 direct forces 计算的 Hessian 也直接使用物理单位。
5. 不添加 forces scale，不添加 Hessian scale，不创建 `ForceRescale` 类或第二个
   `GlobalRescaleShift`。

结果可以是内部混合单位：

```text
energy loss:  normalized energy
force loss:   physical force
Hessian loss: physical Hessian
```

这是合法的。loss weight 本来就负责平衡不同量纲和数值范围。真正需要保证的是：

```text
prediction unit == target unit
```

以及 eval/deploy 输出为物理单位，而不是强迫 energy、forces 和 Hessian 必须共享
一个内部 scale。

### 新模型与加载模型的区别

- 从零构建的新模型：`scale_forces=True` 可以按第 1 节同时初始化 energy 和
  forces scale。
- `model_path.mode=model` 加载的 checkpoint：checkpoint 中已有的 rescale heads、
  数值和 module 顺序是权威来源。datamodule 不得因为 `scale_forces=True` 静默新增
  forces transform 或覆盖 checkpoint rescale。
- 如果用户确实希望 loaded direct-force 模型使用 normalized force loss，必须通过
  显式配置请求；这不是默认转换行为。

因此 direct-force transform 不应把整个模型设置成 `_initialized=False`。新增 direct
head 本身不需要数据统计初始化，已有 rescale layer 也不应被 datamodule 重建。

## 3. Hessian module 的最小插入逻辑

当前训练入口的 `apply_configured_derivative_outputs()` 同时承担了：

- 扫描 Hydra output 配置；
- 删除已有 Hessian module；
- 清理旧 `model_outputs`；
- 重建整个 `output_modules`；
- 修改 PairwiseDistance；
- 重新注册 callbacks；
- 调用全局 output pipeline 重排。

这些复杂度主要是为了配合全局重排和隐式配置同步。删除这两个目标后，不需要这套
逻辑。

### 唯一需要的 helper

在 model conversion/output utility 中保留一个小函数：

```python
def add_hessian_output(model, module):
    if any(isinstance(m, EnergyHessianOutput) for m in model.output_modules):
        raise ValueError("Model already has an EnergyHessianOutput")

    pairwise = next(m for m in model.input_modules if isinstance(m, PairwiseDistance))
    pairwise.compute_distance_from_R = True
    pairwise.compute_forces = True

    force_index = next(
        i for i, m in enumerate(model.output_modules)
        if isinstance(m, ForceOutput) and m.produces_forces
    )
    model.output_modules.insert(force_index + 1, module)
    return model
```

它只做三件事：

1. 让 positions 可微；
2. 找到真正产生 atomic forces 的 module；
3. 在其后插入 Hessian module。

它不重建 ModuleList、不删除 outputs、不处理 rescale、不改变其他 module 的相对顺序。
`CallbackModuleList.insert()` 已经负责登记新 module 的 `model_outputs`。

### 训练入口

训练入口只需要：

```python
hessian_cfg = OmegaConf.select(
    config,
    "model.output_modules.energy_hessian_output",
)
if hessian_cfg is not None and loaded_from_model_checkpoint:
    add_hessian_output(model, instantiate(hessian_cfg))
```

如果 checkpoint 已经包含 Hessian output，默认明确报错，不做隐式替换。需要改变
Hessian 采样参数时，应通过显式 model transform 或重新构建模型完成。

offline teacher preparation 也复用同一个 `add_hessian_output()`，而不是维护第二套
插入和排序逻辑。

## 4. `DistillOutput` 的边界

`DistillOutput` 只接收 student 的 rescale layers，然后调用公开接口：

```python
normalized = target.copy()
normalized[self.student_property] = teacher_value
for layer in self._student_rescale_layers:
    normalized = layer.unscale(normalized, force_process=True)
teacher_value = normalized[self.student_property]
```

它不包含：

- `_curvature_scale()`；
- `property_scale_sources`；
- energy/force producer 类型判断；
- Hessian 专用归一化分支。

如果 forces/Hessian 使用 scale，`GlobalRescaleShift.unscale()` 会处理；如果 loaded
direct-force 模型没有 forces scale，它们保持物理单位。

## 5. Module 顺序策略

不再定义全局合法顺序。以下顺序都可能正确：

```text
# 官方 ScaleShiftMACE gradient
PairRepulsion -> GlobalRescaleShift -> GradientOutput -> EnergyHessianOutput

# Curator gradient，post-force rescale
GradientOutput -> EnergyHessianOutput -> GlobalRescaleShift

# 从零构建并启用 force normalization 的 direct model
DirectForceOutput -> EnergyHessianOutput -> GlobalRescaleShift

# 官方 ScaleShiftMACE 转 direct，forces/Hessian 使用物理单位
PairRepulsion -> GlobalRescaleShift -> DirectForceOutput -> EnergyHessianOutput
```

构造器、loader 和 checkpoint upgrade 都必须保留原有 module 相对顺序。只有显式
model transform 可以对自己负责的 module 做局部替换或插入。

## 6. 需要撤回和保留的代码

### 撤回

- `ForceOutput.normalization_source`；
- `NeuralNetworkPotential.property_scale_sources`；
- `NeuralNetworkPotential._ordered_output_modules()`；
- rescale layer 的 `bind_property_scale_sources()`；
- `DistillOutput._curvature_scale()`；
- 训练入口中的 `apply_configured_derivative_outputs()` 整体重建逻辑；
- direct transform 对整个模型设置 `_initialized=False`。

### 保留

- `ForceOutput` 作为 force producer marker；
- `DirectForceOutput` 和 representation feature specs；
- 单一 force producer 检查；
- PaiNN vector bias 修复；
- Hydra force-output 稳定槽；
- legacy config/checkpoint 兼容；
- `EnergyHessianOutput` 对 forces 求导的实现。

## 7. 验收测试

1. `scale_forces=True` 时 energy scale 和 forces scale 都等于 force RMS。
2. energy shift 保持 energy 统计值，forces 没有 shift。
3. full/sampled/projected Hessian 使用 forces transform 的同一个实时 scale。
4. forces scale trainable 时，Hessian scale 自动同步。
5. 没有 forces scale 的模型保持 forces/Hessian identity。
6. 普通官方 MACE 转换前后 energy/forces 一致。
7. 官方 ScaleShiftMACE 转换前后 energy/forces 一致，并保留 module 顺序。
8. ScaleShiftMACE 转 direct 后只保留原 energy rescale，不新增 rescale 类或模块。
9. loaded direct model 的 force/Hessian loss 在物理单位中比较。
10. `add_hessian_output()` 只在 force producer 后插入一个 module。
11. online/offline teacher 都只通过 student rescale layer 转换一次。
12. multi-domain rescale 对每个 domain 使用相同的 force-dependent Hessian 规则。
