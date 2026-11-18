# Curator 长程对标方法：实现与使用入口

这次新增的是四种可调用的方法接口，其中前三种可以在**同一个 Curator
短程主干上训练**。它们和已有 ACVR 分开封装，不改变正在运行的 ACVR
能量、proposal、HVP 或训练任务。

实现完成、梯度测试通过和模型预测效果更好是三个不同结论。本目录记录前两项的
接口与验证方式；正式训练的能量／力误差需要后续独立实验给出。
本轮没有提交这些新增方法的正式训练。

## 四种方法分别做什么

| 方法／实验名称 | 怎样得到总能量 | 当前范围 | 详细文档 |
|---|---|---|---|
| MACE + LES | 短程神经能量，加上由网络预测 latent charges 后计算的官方 Gaussian LES 长程能量 | TTT 周期结构与 FFF 孤立结构；可以混合批处理 | [LES](LATENT_EWALD.md) |
| MACE + polarizable LES（rank 1） | 上述长程能量进一步包含永久偶极及一次诱导偶极响应 | 官方扩展 LES 的 rank-1 子集；无四极、无迭代自洽 PQEq | [LES](LATENT_EWALD.md)、[极化方法边界](POLARIZABLE_BASELINES.md) |
| MACE + EFA adapter | 短程神经能量，加上 Euclidean Fast Attention 产生的全局能量修正 | 所有输入 irreps 参与 Q/K，V 为标量；FFF 球面网格或 TTT 官方晶格分支 | [EFA adapter](EFA_ADAPTER.md) |
| AIMNet2 adapter | 直接调用完整 AIMNet2 模型的能量，包含它自己的神经项、Coulomb 与 D3 | 孤立分子；显式总电荷输入；已有作者 checkpoint 用作参考 | [AIMNet2](AIMNET2_ADAPTER.md) |

前三种方法都可以复用同一份 pure MACE 初始化、同一 train/validation/test
划分、同一能量参考值、同一 EF loss 和优化器。主干与新增模块共同训练。
接口本身基于 Curator 的 response feature specification，而非硬编码某种主干；
其他主干仍需各自完成集成验证，不能仅凭接口兼容宣称所有模型已经实测。

EFA adapter 是论文 EFA 方程的一种明确配置：等变 Q/K、标量 V、不使用
tensor-product integration。它不是作者完整模型及训练方案的复现，
也不是普通 softmax Transformer。FFF 使用 Lebedev 球面网格，TTT 使用作者
公开周期分支的三个原始晶格矢量及 1/3 权重，可以装配 AuMgO 周期模型。
部分周期条件（如 TTF）仍不支持。这是晶格方向编码，不是 Ewald 求和；
单原子跨晶胞换像、任意晶胞基变换和复制晶胞不变性未由该公式保证，
应保持一致的坐标及晶胞约定。具体适用范围与检查见 EFA 文档。

Polarizable LES 使用官方 PyTorch 算子。当前诱导偶极由永久源场计算一次，
不做迭代自洽求解；其可训练极化率上界是实验超参数，不是全局稳定性定理。
真正的 ReaxNet/PQEq 核壳模型尚未实现，不能把这项实验命名为 PQEq 复现。

## 公平比较与 AIMNet2 的区别

前三组使用 pure MACE 初始化，是在同一个主干上比较不同长程模块。
新增参数量、训练耗时和预测耗时仍应单独报告，不能因主干相同就宣称
各组总参数量或计算量完全相等。

AIMNet2 是另外一种完整网络，不是 MACE 上的能量修正头。目前下载的 ACDB
作者 checkpoint 曾采用电荷监督，而且可能已见过我们重新划分的测试结构。
因此它的预测可以用于代码对照和作者模型参考，**不能作为我们划分下的
独立、公平、EF-only 测试成绩**。在该 checkpoint 上只使用 EF 做微调，
也不能消除其已有训练历史或测试集泄露。

要将 AIMNet2 纳入严格的 EF-only 排名，需要使用新初始化的完整官方架构，
只训练我们的 frozen train split，并保持测试集完全独立。当前 adapter
支持接收完整可微的 PyTorch 模型，但没有把作者预训练模型冒称为新初始化。

所有模型都不使用 Bader/DDEC 等原子电荷标签作为本轮训练目标。AIMNet2
输入的分子总电荷是边界条件，与原子电荷标签不同。当前 LES latent charges
没有被强制为物理原子电荷，也不投影到固定总电荷；EFA adapter 只读取
几何与主干特征。这些边界处理差异必须随结果一并注明。

## 代码结构

- `curator/model/longrange_benchmark.py`：
  `augment_energy_model(base_model, adapter)`，复制基础模型并装配新增输出模块。
- `curator/layer/_latent_ewald.py`：
  `LatentEwaldEnergy` 与 `PolarizableLatentEwaldEnergy`。
- `curator/layer/_efa_adapter.py`：`EFAAdapter`。
- `curator/model/aimnet2_adapter.py`：`AIMNet2Adapter`，包装完整模型作为 representation。
- `scripts/prepare_longrange_baseline.py`：保存独立初始化 checkpoint、
  adapter 配置、来源 manifest，并可生成训练配置。
- `curator/train/ef_monitor.py`：`EFPhysicalMonitor`，在物理单位下汇总 EF 误差。
- `curator/configs/model/output_modules/{les,polarizable_les,efa_adapter,efa_adapter_periodic}.yaml`：
  前三种模块的可选配置。
- `curator/configs/aimnet2_ef_finetune.yaml`：
  作者 AIMNet2 checkpoint 的 Curator EF 微调配置；参考模型限制依然适用。

这些模块使用 Curator 现有数据、模型、Lightning 训练、优化器和 checkpoint
接口。没有另写一套训练器，也没有把模型经 NumPy/ASE 接口转换后假装可训练。
所有新增模块都应在创建优化器之前绑定特征规格并完成参数注册。

## 为什么要调整力的计算路径

装配顺序为：

    短程主干 -> 物理单位的短程能量
             -> 加入 LES／EFA 的全局能量
             -> 对原子坐标求导 -> 总力

总力必须是 F = -dE_total/dR。这同时包含长程能量对几何的直接依赖，
以及主干特征、latent charges 等对几何的间接依赖。
如果只对短程邻居的 edge difference 求导，会漏掉全局项的直接坐标贡献。
公共装配器负责恢复坐标梯度并使用完整的位置导数。

EF 微调还需要能量关于坐标和网络参数的混合导数，所以“能做一次前向预测”
不足以证明可以训练。测试必须实际对 force loss 反向传播到新增模块和主干。

LES 与 EFA 不是 ACVR 的驻定求解器，因此它们没有需要优化的 ACVR residual
loss，不能给它们添一个无物理意义的零残差并用于排名。
最终比较应同时报告：

1. 在相同独立测试集上的能量／力误差，以及长程扫描的表现。
2. 各模型训练与推理的实际计算成本。
3. ACVR 另外必须满足的驻定残差门槛；能量／力表现改善不能替代这个条件。

## 运行环境与准备示例

canonical Git 源码：

    /data1/home/hzxy05/work/yangxin/curator-acvr

现有运行镜像保持不变：

    /data1/home/hzxy05/images/curator-dev.sif

本任务专用 launcher：

    /data1/home/hzxy05/work/yangxin/acvr/runtime/longrange-baselines-20261010/run.sh

该 launcher 在现有运行环境上加入隔离的官方 LES Python 源码路径。
依赖、Git 源码、数据和实验输出保持分开存放。

以下命令只准备一个新的模型和配置，**不启动训练**。
输出目录必须是尚不存在的新目录：

```bash
cd /data1/home/hzxy05/work/yangxin/curator-acvr

baseline_run_dir=/data1/home/hzxy05/work/yangxin/acvr/experiments/longrange_baselines_20261010/prepared/nacl_les_seed3407

/data1/home/hzxy05/work/yangxin/acvr/runtime/longrange-baselines-20261010/run.sh cpu \
  python scripts/prepare_longrange_baseline.py \
  --base-model /data1/home/hzxy05/work/yangxin/acvr/experiments/acvr_longrange_20261009/initialization/nacl/pure_mace_initial.pt \
  --adapter-config curator/configs/model/output_modules/les.yaml \
  --training-template /data1/home/hzxy05/work/yangxin/acvr/experiments/acvr_longrange_20261009/configs/nacl_pure_mace.yaml \
  --output-dir "$baseline_run_dir" \
  --seed 3407
```

将 adapter 配置替换为 `polarizable_les.yaml` 或 `efa_adapter.yaml`，
并选择各自的新输出目录，即可准备另外两组。EFA 数据必须显式提供
FFF 或 TTT 边界标志；TTT 还需要每个结构对应的有效晶胞。
AuMgO 的官方周期频率设置为 max_frequency=pi/4、max_length=1.0，
可用 `efa_adapter_periodic.yaml` 独立配置准备；该配置不创建 Lebedev
网格，只接受 TTT，不能将分子直径参数解释直接套到晶格投影。
不同数据集必须选其各自的 pure MACE 初始化与训练模板，以保留正确的
元素能量参考、数据划分、边界条件及预处理。

准备结果包含 `initial_model.pt`、`adapter.yaml`、`manifest.json`
和可选的 `train.yaml`。正式 GPU 训练应由 Slurm 使用专用 launcher 执行，
并继续采用确定性运行配置。不要复用或覆盖正在运行的 ACVR 输出目录。

## 验证记录

具体实现的单元测试见各方法文档。总集成审计保存在：

    /data1/home/hzxy05/work/yangxin/acvr/experiments/longrange_baselines_20261010

应以该目录的测试日志、集成结果 JSON、checkpoint 回读和 native Curator
优化器步骤记录核验完成情况。最终通过数量由本轮汇总报告给出；
这里不把尚未结束的检查写成通过。

CPU float32 的已有 MACE TorchScript 子模块可能在首次 JIT 优化后出现
小幅浮点运算差异。序列化检查应统一预热条件或关闭 JIT 优化后比较，
并记录这一条件；不能通过随意放宽阈值掩盖权重或计算图丢失。
相关诊断记录为 `les_serialization_debug.json`。

## 论文、代码来源与许可

- LES：[原始论文](https://www.nature.com/articles/s41524-025-01577-7)、
  [官方 PyTorch 库](https://github.com/ChengUCB/les)，固定 commit
  `35b971a650cf30adb84fef7138761e81aa47b3a2`；CC BY-NC 4.0。
  官方依赖保留在独立 runtime 中，没有复制进 Curator 源包。
- Polarizable LES：[扩展方法论文](https://arxiv.org/abs/2605.05746)、
  [作者训练示例](https://github.com/ChengUCB/extended_les_fit)；
  当前接口的 rank-1 范围见上文。
- EFA：[论文](https://doi.org/10.1038/s42256-026-01195-y)、
  [官方 JAX 代码](https://github.com/thorben-frank/euclidean_fast_attention)，
  官方项目声明 MIT。本任务是独立的 PyTorch adapter；
  Lebedev 数值表来源及 Apache-2.0 的 e3x 交叉核对见 EFA 文档。
- AIMNet2：[论文](https://doi.org/10.1039/D4SC08572H)、
  [ACDB 作者模型](https://github.com/elmjonas/ACDB/tree/454a2e452b0228243781363393d8505798f8b353/Articles/neefjes26_long_range_NN)、
  [维护中的官方包](https://github.com/isayevlab/aimnetcentral)。
  ACDB Zenodo record 11422835 声明 CC BY 4.0；对应 GitHub tree 未提供单独
  LICENSE，因此不推断额外的代码许可。具体 artifact SHA256 与训练历史见
  AIMNet2 文档。

