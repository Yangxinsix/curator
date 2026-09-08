# SPICE 真实数据实验与持久化 GPU 队列

本批实验日期：2026-09-07。目标是检验真实双原子分离曲线上的物理拟合、距离外推和 proposal 误差。
它不验证恒电势电极、复杂分子、长轨迹 MD 或生产加速。

已于 2026-09-07 08:07:12 UTC 启动，09:30:37 UTC 完成全部 31 个任务。
结果和保存缺陷见 [第二阶段结果核对](SPICE_STAGE2_RESULTS_20260908.md)。
特别注意：本批 `last.ckpt` 实际停留在最优轮，不能当作训练最后一轮的恢复点。

## 实验矩阵

数据为本地 SPICE Ion Pairs，源文件 SHA-256：
`631137e6ee16f6974721a5d6653b9f730fb57bd3dbea7e129e8c8a9f95e51ebd`。
28 个元素对，1428 个构型；总电荷取 `-2/0/+2 e`。
按距离固定划分：训练 `r <= 5.5 Angstrom` 共 868 个；验证 `5.5 < r <= 6.5` 共 280 个；
测试 `r > 6.5` 共 280 个。测试距离实际为 6.6–7.5 Angstrom。

| 实验名 | 模型和监督 | 网络种子 | 优化预算 |
|---|---|---|---|
| `short_range` | 相同局域能量网络；能量、力监督；6 Angstrom 外没有距离依赖 | 4、17、42 | 各 200 epoch |
| `acvr_ef` | 局域能量、电负性、正硬度，加完整孤立 Gaussian Coulomb 核；能量、力监督 | 4、17、42 | 各 200 epoch |
| `acvr_efd` | 同一 ACVR 模型，增加偶极监督 | 4、17、42 | 各 200 epoch |
| 上述两种 ACVR 的 proposal | 从对应验证集最优物理 checkpoint 初始化，冻结物理参数 | 同对应物理实验 | 各 100 epoch |

全部使用 GPU float64、batch size 32、Adam 固定学习率 0.003，无 EMA。
物理项是以 1 eV、1 eV/Angstrom、1 e*Angstrom 为尺度的 MSE，启用项权重均为 1。
物理 checkpoint 按全验证集精确聚合的 `val_spice_loss_exact` 选择；
proposal 按 `val_total_loss_exact` 选择。测试误差不参与选择或学习率调整。

原始数据中的部分数组以 float32 存储；float64 计算不会恢复源文件中不存在的精度。
保留源坐标，带电偶极使用源坐标原点；不对构型做未记录的平移。
短程基线的均分电荷及其偶极只是未学习的参考，不参与训练。

## 任务依赖与 GPU 分配

共 31 个任务：9 次物理训练、9 次物理评估、6 次 proposal 训练、6 次 proposal 评估、1 次汇总。

```text
物理训练 → 独立物理评估 → proposal 训练 → 独立 proposal 评估
                    全部评估完成 → summary.json / metrics_by_seed.csv
```

调度器扫描 `nvidia-smi` 的 GPU UUID、显存、利用率和计算进程，已有计算任务的卡不会被分配。
每个任务独占一张被分配的卡，最多并发 6 个任务；`CUDA_VISIBLE_DEVICES` 使用 UUID，
训练进程内部使用单个 `cuda:0`。当前批次允许 GPU 0、2、3、4、5、6、7，避开已有任务的 GPU 1。

独立评估先检查数据 manifest、有限数值和 checkpoint 一致性。
proposal 评估逐参数检查物理参数冻结，并对照未训练 proposal；
输出投影前后状态误差、认证接受率、修正次数、fallback，以及修正后状态和能量误差。
该认证扫描使用 CPU 稠密参考求解，不将其耗时当成 GPU 吞吐或实际加速证据。
近似解的力误差和大体系性能需要后续专门实验。

## 持久化目录

本批目录：`runs/acvr_spice_stage2_20260907/`。

- `manifest.json`：完整任务 argv、依赖、随机种子和运行目录。
- `source/`、`source_hashes.json`：启动时的代码与配置快照及逐文件 hash；后台训练从此快照导入。
- `environment.json`、`git_head.txt`、`git_diff.patch`、`git_status.txt`：环境与工作区证据。
- `state/state.json`：任务状态、进程身份、GPU 分配和心跳。
- `state/events.jsonl` 及任务 attempt 目录：调度事件、独立退出状态和标准输出日志。
- `experiments/<实验>/seed<种子>/`：Hydra 配置、训练日志、逐轮指标、最优和最后 checkpoint。
- 各 `evaluation/`：`metrics.json`、验证／测试的原始预测与分离曲线 CSV。
- `summary.json`、`metrics_by_seed.csv`：全部任务结束后生成的汇总。

退出聊天或终端后任务继续运行。调度器异常退出后，可以用同一个 `run` 命令恢复；
它会接管仍存活的任务，避免重复启动。训练任务异常结束且有 `last.ckpt` 时，最多自动续训重试一次。
失败任务会阻塞依赖，不能伪装成完成。
机器重启后需重新启动调度器；本机用户 systemd 不可用，未安装开机服务。

## 查看、恢复与停止

在仓库目录执行：

```bash
cd /workspace/work/agent-curator/curator

../.venv/bin/python scripts/acvr_gpu_campaign.py status \
  --state-dir runs/acvr_spice_stage2_20260907/state
```

调度器退出后的后台恢复命令：

```bash
nohup ../.venv/bin/python \
  runs/acvr_spice_stage2_20260907/source/scripts/acvr_gpu_campaign.py run \
  --manifest runs/acvr_spice_stage2_20260907/manifest.json \
  --state-dir runs/acvr_spice_stage2_20260907/state \
  --max-parallel 6 --gpus 0,2,3,4,5,6,7 --poll-seconds 10 --min-free-mb 8192 \
  >> runs/acvr_spice_stage2_20260907/scheduler.log 2>&1 < /dev/null &
```

单实例锁会拒绝同时启动第二个调度器。
明确取消本批任务（只停止经过身份核对的本批进程，不影响其他实验）：

```bash
../.venv/bin/python scripts/acvr_gpu_campaign.py stop \
  --state-dir runs/acvr_spice_stage2_20260907/state
```

`stop` 是取消操作；取消的任务不会被相同 `run` 自动重新执行。
另起新批次时使用新目录准备 manifest，再启动其调度器：

```bash
../.venv/bin/python example/acvr/prepare_spice_campaign.py \
  --output runs/acvr_spice_new_campaign
```

## 启动前检查与已知限制

预检目录：`runs/acvr_spice_preflight_20260907/`。
覆盖真实数据 GPU 小训练、独立 checkpoint 重建与评估、proposal 物理参数冻结、断点续训，
以及数据、数学导数、输出协议和调度器的回归测试。
最终定向回归为 `256 passed, 77 warnings in 70.95s`，无跳过。
使用最终配置从第 3 轮恢复到第 4 轮，与连续 4 轮的模型参数和 Adam 状态逐项相同；
独立 GPU 评估再次确认 proposal 的物理参数和物理预测完全不变。
启动验证副本保存在本批 `preflight/` 和 `launch_verification.json`。

预检发现当前 Lightning 的 plateau 学习率调度状态在 epoch 更新前写入 checkpoint，
会导致恢复后调度计数落后一次；本批使用固定学习率，相关旧预检记录保留供追溯。
训练终端中旧的 batch 平均 loss 仅供观察；物理模型选择使用新增精确验证指标，
物理误差以独立评估和全数据 RMSE 为准。

该双原子模型不是通用 MLIP。能量与力拟合良好并不证明响应已被充分识别；
E/F 与 E/F/偶极的对照正是用于检查这一点。
