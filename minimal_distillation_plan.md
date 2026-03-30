# Minimal Energy/Forces Distillation Plan

## Goal

先做一个最简单的蒸馏版本，只支持：

- `energy` distill
- `forces` distill
- 不做 feature distill
- 尽量不改 `curator/model/lit_module.py`

当前状态：

- 已经有一个最小 online `DistillOutput` 原型
- 推荐的下一步方向改为 offline teacher labels
- online 路径保留为 fallback，不作为默认主路径

## Short Answer

基本可以先不改 `lit_module.py`，但不能只改现有 `ModelOutput` 的几行逻辑就结束。

可行的最小路径是：

1. 保留现有 `LitNNP` 的训练循环不动。
2. 新增一个蒸馏专用的 `ModelOutput` 子类。
3. 让这个新 output 在 `calculate_loss()` 里自行完成：
   - 加载/缓存 teacher
   - 运行 teacher forward
   - 取 teacher 的 `energy` / `forces`
   - 和 student `pred` 计算 distill loss
4. 继续保留原来的 hard-label `energy` / `forces` loss。

这样 `LitNNP.loss_fn()` 仍然只是在遍历 outputs；蒸馏被封装在 output 内部。

## Why This Works Without Touching `lit_module.py`

现有训练路径里，`LitNNP` 已经会对每个 output 调用：

```python
output.calculate_loss(pred, batch, True)
```

这给了 output 两个关键信息：

- `pred`: student 当前 batch 的预测
- `batch`: 当前 batch 的原始输入和标签

因此 distill output 可以在自己的 `calculate_loss()` 里直接：

1. 从 `batch` 里拿输入
2. 跑一次 frozen teacher
3. 用 teacher 输出作为 soft target
4. 返回一个普通 loss

这意味着蒸馏不一定非要由 `LitNNP` 显式感知。

## Minimal Design

### 1. 保留现有 hard loss

继续使用已有的：

- `energy` hard loss
- `forces` hard loss

不改现有 `ModelOutput` 的默认行为。

### 2. 新增 distill output

新增一个子类，例如：

- `DistillOutput(ModelOutput)`

建议配置字段：

- `name`: 例如 `energy_distill` / `forces_distill`
- `student_property`: 真正从 `pred` 里读取的 key，例如 `energy`
- `teacher_property`: teacher 输出里读取的 key，例如 `energy`
- `teacher_model_path`
- `teacher_labels_path` 不直接放在 output 上，而是走 dataset/datamodule
- `teacher_cfg`: 可选，teacher checkpoint 没有完整模型时兜底
- `loss_fn`: 先只支持 `MSELoss`
- `loss_weight`
- `only_train`: 默认 `True`
- `cache_key`: 用于 batch 内缓存 teacher 结果

### 3. online teacher 放在 output 内部懒加载

为了不改 `lit_module.py`，teacher 由 `DistillOutput` 自己管理。

建议实现为：

- 第一次进入 `calculate_loss()` 时才加载 teacher
- teacher 全部参数 `requires_grad_(False)`
- teacher 始终 `eval()`
- 每次 forward 前检查 device / dtype，必要时迁移到和 student `pred` 一致

## Important Constraint

如果 teacher 作为普通 `nn.Module` 子模块直接挂在 output 上，会有两个副作用：

1. 它会被 optimizer 参数分组扫描到
2. 它会被 checkpoint 的 `outputs` 一起序列化

这两个都不是当前最小方案想要的。

因此建议：

- 不把 teacher 注册成普通子模块
- 只在 output 内保存：
  - `teacher_model_path`
  - 一个运行时 lazy cache

也就是说，teacher 更像“运行时资源”，不是训练参数的一部分。

## Batch-Level Cache

因为同一个 batch 会先算 `energy_distill`，再算 `forces_distill`，如果每个 output 都各跑一遍 teacher，会重复前向。

最小方案建议把 teacher 结果缓存到当前 batch 上，例如保留字段：

- `__teacher_pred__`

流程：

1. `energy_distill` 第一次发现 batch 没有 cache
2. 运行 teacher
3. 把 teacher 输出写入 batch cache
4. `forces_distill` 直接复用

这样可以继续不改 `lit_module.py`，同时避免同 batch 双重 teacher forward。

## Forces Distillation Caveat

`forces` 是通过 `GradientOutput` 用 autograd 从 `energy` 导出来的。

因此 teacher forward 不能简单地放进完全禁用梯度的上下文里，否则拿不到 `forces`。

实现上需要满足：

- teacher 参数本身不参与训练
- 但 teacher 对输入图的求导链条仍然可用

建议做法：

- 冻结 teacher 参数
- `teacher.eval()`
- 不对整个 teacher forward 使用 `torch.no_grad()`

这样 teacher 不会更新，但仍然能产出 `forces`。

## Updated Direction

结论更新：

- `DistillOutput` 保留 online 逻辑
- 但推荐主路径改成 offline distillation
- offline 模式下，`teacher_model_path` 可以为 `None`
- 此时 `DistillOutput` 不再运行 teacher，而是直接从 batch 中读取 teacher labels

推荐的双模式语义：

1. `teacher_model_path != None`
   - online 模式
   - `DistillOutput` 自己加载 teacher 并推理
2. `teacher_model_path == None`
   - offline 模式
   - `DistillOutput` 直接使用 batch 里的 `teacher_energy` / `teacher_forces`

这样可以保留当前原型，同时把真正高效的训练路径切到 offline。

## Why Offline Is Preferred

online 原型只能解决“同一个 batch 内多个 distill losses 不要重复跑 teacher”。

它解决不了：

- 每个 epoch 整个训练集都要重新跑 teacher
- teacher 很大时的总推理开销
- `energy_distill` 和 `forces_distill` 只是局部共享，不是全局消除 teacher 代价

offline 的优势更明确：

- teacher 只预计算一次
- 训练时只跑 student
- 多个 distill losses 共享同一份预计算标签
- 不需要 teacher model 常驻 GPU
- 不需要 step 级 teacher prediction cache

## Offline Data Strategy

不建议把 teacher labels 直接写回原始 ASE / traj 数据：

- 原始数据会明显变重，尤其是 `forces`
- 源数据和派生标签混在一起，不利于版本管理
- 以后切换 teacher checkpoint 时会污染原始数据

更推荐：

- 保持原始结构数据不动
- 单独维护一个 sidecar teacher label store
- 训练时把原始 dataset 和 sidecar labels overlay 到一起

## Preferred Offline Store: SQLite3 Sidecar

在当前 repo 中，优先推荐 sqlite3，而不是 HDF5。

原因：

1. 仓库里已经有现成的 sqlite 数据访问代码
   - `curator/data/sql_database.py`
   - 已经处理了多 worker 下每进程单独连接的问题
2. teacher labels 是按 structure 随机访问的
   - `energy`: 标量
   - `forces`: 变长 `(N_atoms, 3)`
   - sqlite 一行一个 structure 更自然
3. HDF5 在当前 repo 没有现成 dataset 封装
   - 需要自己维护 offsets 或 variable-length layout
   - DataLoader 多 worker 下文件句柄/并发处理更麻烦

推荐形态：

- 一个独立 sidecar 文件，例如 `train.teacher_labels.sqlite`
- 不重复存结构本体
- 只存与蒸馏相关的 teacher labels

建议 schema：

- `id`
- `teacher_energy`
- `teacher_forces`
- `n_atoms` 或一个简单 sanity-check 字段
- 可选 metadata：
  - `teacher_model_path`
  - `source_dataset`
  - `num_structures`
  - `created_at`

## Dataset Integration

推荐增加一个 overlay dataset，而不是改原始 reader：

- `base_dataset = AseDataset(...)` 或其他现有 dataset
- `dataset = TeacherLabelOverlayDataset(base_dataset, teacher_label_store)`

`TeacherLabelOverlayDataset.__getitem__(idx)`：

1. 从 `base_dataset[idx]` 读取原始样本
2. 从 sidecar label store 读取 `teacher_energy` / `teacher_forces`
3. 把它们 merge 到 sample 里
4. 返回给现有 collate 逻辑

这样有几个好处：

- 不需要改 `AseDataReader`
- 不污染原始数据
- 未来切换 teacher 只需要换 sidecar 路径
- 只给 train split 增加 teacher labels 也很自然

## CLI Integration

offline 模式不是“先创建一个 offline datamodule”，而是：

1. 训练开始前检查 sidecar labels 是否存在
2. 如果不存在，且提供了 `teacher_model_path`，则先生成一次 sidecar
3. 把 `teacher_labels_path` 注入 `config.data`
4. 再执行：

```python
datamodule = instantiate(config.data)
```

推荐在 `cli.py` 中做双判断：

- `teacher_labels_path` 优先
- `teacher_model_path` 仅作为 sidecar 缺失时的生成来源

伪代码：

```python
if offline_distill_enabled:
    if config.data.teacher_labels_path exists:
        use it
    elif config.task.teacher_model_path exists:
        build teacher labels once
        set config.data.teacher_labels_path
    else:
        error

datamodule = instantiate(config.data)
```

关键点：

- 不要因为有 `teacher_model_path` 就每次训练都重算 labels
- 真正训练消费的是 `teacher_labels_path`
- `teacher_model_path` 更像“生成 labels 的来源”

## DistillOutput Changes For Offline

`DistillOutput` 建议收敛成下面的行为：

1. online:
   - `teacher_model_path != None`
   - 用当前原型逻辑推理 teacher
2. offline:
   - `teacher_model_path == None`
   - 直接读取 batch 中的 `teacher_property`

推荐的 teacher label 命名：

- `teacher_energy`
- `teacher_forces`

示意：

- `student_property = energy`
- `teacher_property = teacher_energy`

以及：

- `student_property = forces`
- `teacher_property = teacher_forces`

这样 offline 时 `DistillOutput` 就真正只剩下 loss 计算职责。

## Proposed File Changes

### 必改

- `curator/train/model_output.py`

建议内容：

- 新增 `DistillOutput`
- 增加 teacher lazy loading
- 增加 batch teacher cache
- 支持 `teacher_model_path=None` 时直接读取 batch teacher labels

### 新增配置

- `curator/configs/task/outputs/components/energy_distill.yaml`
- `curator/configs/task/outputs/components/forces_distill.yaml`
- `curator/configs/task/outputs/energy_force_distill.yaml`

### offline 路径新增

- `cli.py` 中的 sidecar 准备逻辑
- 一个 `TeacherLabelOverlayDataset`
- 一个 label-only sqlite sidecar builder

### 可能要补的导出

- `curator/train/__init__.py`

如果希望 Hydra / import 路径更整洁，可以把 `DistillOutput` 导出来。

## What I Do Not Want to Change in Phase 1

第一阶段先不动：

- `curator/model/lit_module.py`
- `curator/data/*`
- `curator/model/*`

只有当下面任一问题真的挡路时，才再回头改：

- teacher 生命周期管理太别扭
- 多卡/DDP 下 teacher 懒加载行为不稳定
- checkpoint / resume 语义不清晰

## Minimal Training Config Shape

目标是把 task outputs 组织成 4 个 loss：

1. `energy`
2. `forces`
3. `energy_distill`
4. `forces_distill`

总 loss 仍然由 `LitNNP.loss_fn()` 自动求和。

示意：

```yaml
defaults:
  - _self_
  - /task/outputs/components@energy: energy
  - /task/outputs/components@forces: forces
  - /task/outputs/components@energy_distill: energy_distill
  - /task/outputs/components@forces_distill: forces_distill
```

其中：

- hard loss 继续对 GT
- distill loss 对 teacher prediction

offline 时建议写成：

```yaml
defaults:
  - _self_
  - /task/outputs/components@energy: energy
  - /task/outputs/components@forces: forces
  - /task/outputs/components@energy_distill: energy_distill
  - /task/outputs/components@forces_distill: forces_distill

energy_distill:
  student_property: energy
  teacher_property: teacher_energy
  teacher_model_path: null

forces_distill:
  student_property: forces
  teacher_property: teacher_forces
  teacher_model_path: null
```

## Implementation Steps

### Phase 1: Add the distill output class

在 `curator/train/model_output.py` 中新增蒸馏输出类，能力包括：

- 读取 student `pred`
- 懒加载 teacher
- 运行 teacher forward
- 从 batch cache 复用 teacher 结果
- 仅在 train 模式下返回 distill loss
- 在 val/test 下返回零损失

### Phase 2: Add config components

新增：

- `energy_distill.yaml`
- `forces_distill.yaml`
- `energy_force_distill.yaml`

配置项至少包含：

- teacher 路径
- distill 权重
- student/teacher property 对应关系

### Phase 3: Smoke test

做一个最小 smoke test，验证：

1. `LitNNP` 无修改时可以正常训练一步
2. 同 batch 只跑一次 teacher
3. `energy_distill` 和 `forces_distill` 都能参与总 loss
4. teacher 参数没有梯度

### Phase 4: Offline Distillation Path

在 online 原型跑通后，进入真正推荐的训练路径：

1. 新增 sqlite sidecar builder
2. 新增 `TeacherLabelOverlayDataset`
3. `cli.py` 支持：
   - `teacher_labels_path`
   - `teacher_model_path`
   - missing-then-build
4. `DistillOutput` 支持 offline 直接读 batch teacher labels

offline 路径完成后，训练时默认不再运行 teacher model

## Test Checklist

建议至少补 3 个测试：

### Test 1: `DistillOutput` 能计算 energy loss

- 输入 student `pred`
- 用 batch 驱动 teacher
- 返回有限的 loss

### Test 2: `DistillOutput` 能计算 forces loss

- teacher 带 `GradientOutput`
- 结果能正确产出 `forces`
- loss 有限

### Test 3: 同 batch teacher 只前向一次

- 两个 distill outputs 共用一个 batch
- 断言 teacher forward 次数为 1

### Test 4: offline teacher labels 可直接参与蒸馏

- batch 中直接包含 `teacher_energy`
- `DistillOutput` 在 `teacher_model_path=None` 时可正常工作

### Test 5: offline overlay dataset 正确 merge sidecar labels

- 原始 sample 不变
- merge 后 sample 含有 `teacher_energy` / `teacher_forces`
- shape 与 structure / atom counts 一致

## Risks

### 1. output 内部管理 teacher 比较“偏门”

这是为了换取“不改 `lit_module.py`”。

短期可行，长期不一定是最干净的结构。

### 2. checkpoint 语义需要刻意控制

如果不小心把 teacher 注册成子模块，checkpoint 会膨胀，而且 resume 逻辑会变复杂。

### 3. DDP / 多进程下的懒加载要验证

第一阶段先以单卡或本地 smoke test 为主。

### 4. sidecar label 与原始 dataset 的索引必须稳定对齐

offline 方案依赖“按 sample index 对齐”。

因此需要至少一个 sanity check：

- `idx`
- `n_atoms`
- 或一个结构摘要字段

## Exit Criteria for Phase 1

完成下面几项就算第一阶段达标：

1. 不修改 `curator/model/lit_module.py`
2. 一个配置就能同时启用 hard `energy/forces` 和 distill `energy/forces`
3. 单步训练可跑通
4. teacher 不进入 optimizer
5. 同一 batch 不重复 teacher forward

## Recommendation

建议现在的总体策略是：

1. 保留 online `DistillOutput` 原型
   - 作为 fallback
   - 也方便快速验证 loss 逻辑
2. 默认主路径转向 offline
   - sqlite sidecar teacher labels
   - `TeacherLabelOverlayDataset`
   - `DistillOutput` 在 `teacher_model_path=None` 时直接读 batch labels

换句话说：

- online 用来验证机制
- offline 用来真正训练

在这个 repo 里，offline + sqlite3 sidecar 是当前最合理、改动也最可控的方案。
