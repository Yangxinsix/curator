# ACVR proposal：消息传递、全局注意力与 Ewald 实施计划

日期：2026-09-11。代码核对：HEAD d788852a8c2a640c525258ccb31042c57d1ad01a，工作区有既存未提交修改；实施时必须记录实际源码快照，不能只记录 HEAD。

状态更新：共同 pointwise、MP、attention、Ewald proposal 及训练/诊断入口已实现，原 MLP 保留。七项固定泛函对照均完成 600 步；七项联合 pilot 及匹配评价均完成，周期主对照为 20 个完整 epochs / 640 steps，QM/MM 为预算内最大共同的 8 个完整 epochs / 880 steps。所有候选均未通过固定泛函完整晋级条件，因此未触发多 seed 扩大训练。实际接口见 [实现说明](PROPOSAL_IMPLEMENTATION_20260911.md)，结果、终点差异与证据见 [实验报告](/workspace/data/work/agent-curator/experiments/acvr_proposal_global_20260911/RESULTS.md)。本文保留原方案及预设验收标准；完成实验不等于实现预设科学目标。

## 1. 本轮决定与要验证的目标

目标：使 proposal 能依据几何、物性和边界条件表达跨原子的平衡响应，降低网络预测状态与同一泛函精确平衡状态的差距，同时保持物性预测质量和摊销效率。

保留用户指定的三个方向：

| 分支 | 网络中的空间通信 | 本轮适用状态 |
|---|---|---|
| mp | 边界输入先进入特征，再沿近邻图进行等变消息传递 | 标量电荷、三维内部向量 |
| attention | 同一结构、同一条件内的全局等变注意力 | 标量电荷、三维内部向量 |
| ewald | 使用现有电荷 Ewald 核作用于网络的试探电荷，并把电势返回网络 | 三维周期体系中的标量电荷 |

增加一个 pointwise 对照：使用相同的新输入适配和逐点更新，但关闭空间通信。历史默认 head 只作历史复现参考，不代替公共初始化、训练数据和步数匹配的新对照。通信模块会增加参数，固定宽度和块数不代表参数总量完全相同。

mp 是局部传播分支，不预先宣称它具备任意距离的全局求解能力。约束处理本身产生的全局通信，在所有分支保持相同。第一轮不加入电极 token、额外多尺度模块、学习型 Ewald 修正、PPPM 或偶极 Ewald，避免三个方向变成无法解释的组合。

核心比较：同一个数据任务、同一个物理泛函定义、同一个边界定义、相同数据顺序和训练目标，仅替换 proposal 的空间通信。固定泛函诊断中物理参数也完全相同；联合训练中各分支从相同物理参数初始化，然后各自学习，最终参数可能不同。特别是周期电荷任务中，所有分支的物理能量均包含同一个 Ewald 项；不能仅给 ewald 分支更换能量模型。

## 2. 已核对的事实与可复用资产

本节保留实施前的核对记录；后续特征接口及已实现支持范围以 [实现说明](PROPOSAL_IMPLEMENTATION_20260911.md) 为准。

- 当前默认 proposal 位于 [VariationalStateHead](../../curator/layer/_variational_output.py)，是逐原子 MLP；向量版利用 GNN 向量、局部有效外场和原点组成状态。外场没有经过 proposal 内部的空间传播。
- [VariationalEnergyOutput](../../curator/layer/_variational_output.py) 已负责边界编译、预测后投影、物性导数和 residual 梯度分流。继续复用，不新增另一套训练器。
- [features.py](../../curator/model/features.py) 已有 ScalarVectorFeatureSpec 和 IrrepsFeatureSpec。当前 head 对 irreps 只抽取 0e 和 1o，不能把这叫作完整保留高阶特征。
- PaiNN、MACE 提供 direct_force_feature_spec；当前 NequIP 未覆盖基类该方法，且最终能量特征是标量，需暴露适当的中间等变特征。
- [EwaldChargeOperator](../../curator/variational/backends/ewald.py) 和 [EwaldSummation](../../curator/layer/_ewald.py) 已存在。Torch 适配器目前建立每个结构的稠密电荷核；不是 matrix-free 或 PPPM 生产实现。
- 当前 Ewald 是三维周期、导体外边界约定，支持非中性均匀补偿背景。热力学 conq/conp 与空间周期约定是两组不同条件。
- 当前真实 QM/MM 数据为非周期的 6/9 原子结构；首条记录无周期、晶胞为零。状态为每原子三维向量。不能为了比较而给它随意补晶胞，再把标量电荷 Ewald 套在向量分量上。
- QM/MM 固定划分：1760 train / 220 val / 220 test。沿用 [旧配置](/workspace/data/work/agent-curator/experiments/acvr_public_response_20260909/ethanol_methanol_qmmm/config.yaml) 和原始 splits.json。
- 既有诊断：局部外场干预下，当前 proposal 的其他原子状态响应为零，而同一泛函的精确响应非零；冻结后继续训练旧 head 能显著降残差，因此结构缺口不是已证明的唯一误差来源。见 [诊断报告](/workspace/data/work/agent-curator/experiments/acvr_proposal_diagnosis_20260909/REPORT.md)。
- 既有全测试集审计显示：当前泛函的精确平衡物性明显差于预测状态物性。因此仅降低固定泛函的残差不能作为最终成功。见 [平衡审计](/workspace/data/work/agent-curator/experiments/acvr_alignment_audit_20260909/equilibrium_full_test.json)。

首轮结论只覆盖数值求解能力及指定数据上的联合训练；不会因为一个新层能运行就宣称跨体系、跨任意边界或大规模加速成立。

## 3. 共同接口：骨干、边界和训练图

### 3.1 骨干无关的特征接口

拟增加 response_feature_spec()，由几何骨干声明节点/边特征的位置及其旋转类型。旧 direct_force_feature_spec() 保留；为已有模型提供兼容适配，不破坏直接力输出。

- proposal 不导入或实例化特定 PaiNN 内部层。
- 标量骨干可以接入；向量特征可由相对位置及向量外场在 proposal 内构造。
- 标量/向量骨干按类型接入。
- 高阶骨干保留配置需要的全部 irreps，通道映射只混合同一旋转类型；不能默认丢弃高阶输入。
- 第一轮所有三分支使用同一特征适配器和相同类型/宽度配置。首个历史对照使用 PaiNN，接口验收同时覆盖 MACE；补齐 NequIP 的中间特征导出后纳入验收。
- 更换骨干是更换配置，不等于同一个 checkpoint 可以自动变成另一种骨干。尚未提供可微特征及类型声明的外部模型不能声称已接通。

新 proposal 支持先由配置构造，再由 bind(feature_spec, state_spec) 确定维度。当前 head 把构造维度硬传给默认网络的逻辑要兼容此流程。

### 3.2 共同边界输入

延续行约束约定：

$$
Ax=b.
$$

o 是最小范数可行原点，P 是切空间投影，f 是物理能量中的完整线性驱动：

$$
o=A^{\mathsf T}(AA^{\mathsf T})^{-1}b,
\qquad
P=I-A^{\mathsf T}(AA^{\mathsf T})^{-1}A,
\qquad
f_{\mathrm{net}}=Pf.
$$

运行时仍使用 solve 和投影作用，不建立完整 P。无约束时 o=0、P=I。

每个分支获得相同的：

1. 完整几何特征；
2. 有效外场、可行原点；
3. 约束矩阵和目标、有限刚度环境矩阵与刚度；
4. positions、cell、pbc，以及包含周期镜像的 edge_diff/edge_distance；
5. 对明确具有局域物性系数的电荷泛函，提供相同的当前电负性和硬度特征；residual 分支将这些物性输入 detach。

约束和环境不是任意的类别编号。共同边界编码允许在与状态同类型的特征通道上使用 (I-P) 和环境刚度作用。对标量电荷，它们可以通过约束行运算实现；向量状态必须按真实状态分量布局作用，不能随意投影高阶张量通道。

边界规范必须一致：proposal 使用去除冗余规范方向的驱动；物理能量保留完整驱动及相关常数，使物性导数正确。相同规范处理用于三个分支和 pointwise 对照。

### 3.3 明确的共同计算骨架

建议试验起点为 2 个通信块、48 个标量通道；向量任务另保留 48 个极向量通道。高阶通道按骨干特征配置，不以 48 或两层作为能力上限。深度/宽度扩展属于后续验证失败后的定向试验。

h 表示已经注入边界条件的隐藏特征；Readout 是与状态类型一致的可学习读出。每一层都可获得同样的可行试探状态：

$$
x^{(t)}=o+P\,\operatorname{Readout}^{(t)}(h^{(t)}).
$$

三分支只替换通信消息 m。共同的等变逐点更新接收旧特征、消息、试探状态、外场、原点及共同边界编码，再产生下一层特征。最后统一读出并投影：

$$
\widehat x=o+P\,\operatorname{Readout}^{\mathrm{out}}(h^{(L)}).
$$

pointwise 的空间消息为零，但边界处理、试探状态和逐点更新相同。记录各分支有效参数量、训练 FLOP/耗时和峰值显存；相同通道数不等于参数量或成本完全相同，不给物理核添加无用参数来凑数。

### 3.4 保持单次联合训练和正确梯度归属

- 物性分支保留 geometry → proposal → energy 的完整求导图。
- residual 分支 detach 共享几何特征、物性参数和几何/边界张量，重新评估同一个 proposal；所有新适配器、通信层和读出属于 proposal 参数组。
- 不允许 residual 通过可调用对象捕获未 detach 的泛函参数或几何图。
- 首版 context 保持 tensor-only；静态特征规格由模块持有。Ewald 核作为显式张量进入两个分支，便于清楚控制 detach。
- 不使用 dropout 或训练态 BatchNorm；两个分支数值输出必须一致。注意力首版使用支持高阶求导的明确矩阵运算，不假设 fused/Flash 内核支持所有响应导数。
- 未改变架构的 checkpoint 才能按 resume 恢复优化器。改架构后的参数迁移必须记录成功/遗漏项，并新建优化器，不能把 strict=False 加载包装成完整恢复。

## 4. 三个通信模块的具体实现

### 4.1 MPCommunication

几何边来自统一邻居表，但按 proposal cutoff 取实际通信边。对每条边，使用隐藏特征、试探状态、距离和方向构造等变消息：

$$
m_i^{(t)}
=
\sum_{j\in\mathcal N(i)}
M_t(h_i^{(t)},h_j^{(t)},x_i^{(t)},x_j^{(t)},\mathbf r_{ij}).
$$

标量使用不变量 MLP；向量/高阶特征采用相容的等变线性映射、张量积和标量门控。实现依赖通用类型/等变算子，不依赖 PaiNN 类。

验收：外场在读出之前进入邻居消息；无约束测试中，连通路径内的跨原子响应可非零，超出有限层感受野的响应按设计受限。对不连通电极保持这一限制，用于对照，不能暗中增加全局池化。

### 4.2 AttentionCommunication

对同一结构、同一条件的所有原子做全局注意力。a 是标量注意力权重：

$$
a_{ij}^{(t)}
=
\operatorname{softmax}_j
\bigl(
\operatorname{score}_t(h_i^{(t)},h_j^{(t)},\text{pair geometry}_{ij})
\bigr),
$$

$$
m_i^{(t)}
=
\sum_j a_{ij}^{(t)}\,V_t(h_j^{(t)},\text{pair geometry}_{ij}).
$$

首版 4 个头、2 个块，标量 queries/keys 来自不变量，values 和方向耦合保留特征类型。禁止直接用普通 Dense 混合笛卡尔方向和不同 irreps。

必须有逐结构、逐条件 mask；一个 batch 中不能跨 geometry 通信，也不能让同一 geometry 的不同边界互相读到答案。可以逐条件前向先保证正确，再优化 batching。

周期电荷任务的 pair geometry 必须具有周期平移一致性。初版采用明确的周期相对几何编码，验证镜像平移及坐标导数；最小镜像切换处的不可微问题须在加入力/应力训练前处理。仅世界坐标拼接不算合格实现。

### 4.3 EwaldCommunication

首版只接受 scalar、components=1 的真实电荷状态，以及有效三维周期晶胞。使用原有物理核 G，在每个块中计算试探电荷产生的电势：

$$
\psi^{(t)}=G_{\mathrm{Ewald}}\,q^{(t)}.
$$

把每个原子的该电势输入共同的可学习逐点更新。Ewald 核不训练；前后的编码/读出/更新训练。第一轮不将任意隐藏向量分量解释为电荷，不做潜在多通道 Ewald。

- G 与能量中使用的核完全相同，包括单位、自作用、背景及周期约定。
- 一个 geometry 的核构建一次，在多个条件、通信块和能量计算中复用。需要对坐标求导时不能跨 optimizer step 缓存已 detach 的核。
- 实空间邻居表必须覆盖 Ewald 实空间截断；不能直接复用半径 5 Å 的 GNN 邻居表而漏项。可建立更大的共同邻居表并分别筛选，或维护专用列表。
- 各块只有 kernel matvec，没有调用完整 KKT、CG、Newton 或 reference solve。共同的硬约束投影仍可使用小型约束求解。
- 第一轮采用全体系电中性；非中性补偿背景和电子开放体系留作独立扩展。
- 记录核构建、每块 matvec、投影、训练导数的全部成本。当前稠密实现不作为大规模线性复杂度证据。

## 5. 代码修改清单与交付顺序

| 顺序 | 位置 | 实际改动 |
|---|---|---|
| 1 | curator/model/features.py、base.py、painn.py、mace.py、nequip.py | 新增通用响应特征接口、兼容旧接口、保留中间高阶特征，允许标量骨干接入 |
| 2 | curator/layer/_variational_output.py | 新 proposal 的 bind、完整特征和几何张量传递、共同边界输入、明确梯度分流和 checkpoint contract |
| 3 | 拟新增 curator/layer/_variational_proposal.py | BoundaryConditionedProposal、共同输入/更新/读出、pointwise/mp/attention/ewald 插件 |
| 4 | curator/variational/backends/ewald.py；必要时新增 charge 内部能量适配器 | 核构建与使用分离；接入通用 joint 输出；沿用既有 Ewald 数学实现 |
| 5 | 拟新增 curator/data/periodic_charge_oracle.py | 可复现周期电荷数值数据；按几何划分、多边界条件、物性与精确状态参考 |
| 6 | curator/data/public_response.py、variational_data.py；必要时 variational_losses.py | 条件级局域场增广、缺失物性标签 mask、保持按几何平衡聚合 |
| 7 | curator/variational/observables.py、curator/evaluate/variational.py、joint_callbacks.py、拟新增诊断脚本 | 控制量/约束目标的热力学导数；保存原始 proposal、同泛函参考解、跨位置响应、各边界分项、端到端计时 |
| 8 | curator/configs 与 test | 三分支/pointwise 配置、模型迁移合同、针对性回归及运行说明 |

训练入口以当前源码 curator.commands.train.run_train_config 为准，复用现有 Hydra/Lightning 流程；技能参考中的历史入口仅作导航，不作为实际可调用性的依据。

旧工作区修改和旧实验目录不覆盖。新运行拟写入：

~~~
/workspace/data/work/agent-curator/experiments/acvr_proposal_global_20260911/
    qmmm/<phase>/<arm>/seed_<seed>/
    periodic_charge/<phase>/<arm>/seed_<seed>/
~~~

第一批实现完成标准是共享接口、pointwise/mp/attention 前向与训练图通过；第二批接入 Ewald 和周期电荷数据；第三批运行结构诊断和新联合训练。

## 6. 数据与训练安排

### 6.1 任务 A：已有 QM/MM 真实物性数据

分支：pointwise、mp、attention。Ewald 不参加这组比较。

- 使用既有原始数据、物理单位及 1760/220/220 划分，保持 6/9 原子混合，不额外按标签筛选。
- 结构诊断复用旧 checkpoint 的几何/泛函，并复用既有 manifest 中的 128 train / 64 val。新的三个 head 使用对应 seed 的相同共同模块初始化；历史旧 head 结果另列。
- 局域场增广只能使用训练集统计定义幅度。建议初始增广标准差为训练场分量 RMS 的 0.1 倍；RMS、采样 seed、每个 geometry 的条件数写入 manifest。
- 新外场下没有新的 DFT 物性标签。增广条件只贡献驻定 loss；不能沿用原条件能量、偶极或极化率标签。
- 当前 local_external_field 是按 geometry 存储的。若要同一 geometry 多个局域场条件，必须增加条件级字段/索引，不能仅复制 boundary_spec 而让实际输入仍是同一个场。
- 同一 batch 含带标签原条件及 residual-only 新条件时，required_labels 在后者设 eligible=false、present=false；保留原条件的标签完整性检查。
- 各 geometry 的 residual 先在其原条件和增广条件之间平均，避免仅因增加条件数量就隐式提高 loss 权重。
- finite-difference 响应评估沿用 1e-3、1e-4 V/Å 双步长。参考状态 Jacobian 只在固定小验证子集计算，不在每个训练样本建立完整 Jacobian。

结构诊断使用固定泛函，是独立定位试验，不是正式生产模型的两阶段训练。它只回答新网络能否更准确地求解同一泛函。

正式物性训练仍从头联合学习几何骨干、基线、泛函和 proposal。所有新分支使用同一份骨干/泛函初始权重；避免从已发生非驻定补偿的旧泛函开始后，把历史状态当作架构效果。

### 6.2 任务 B：周期电荷上的三方向公平比较

分支：pointwise、mp、attention、ewald。

首轮新建有明确解的周期电荷数值任务；它证明求解与边界能力，不冒充 DFT 金属电极数据。所有分支共享：

$$
F_\theta(R,q)
=
F_{0,\theta}(R)
+
\chi_\theta(R)^{\mathsf T}q
+
\frac12q^{\mathsf T}
\left[
\operatorname{diag}(\eta_\theta(R))
+
G_{\mathrm{Ewald}}(R,\mathrm{cell})
\right]q.
$$

其中 chi 为电负性、eta 为局域硬度；此处固有系数只依赖几何。边界驱动、硬约束和有限环境由现有编译器加入，不能再次计入 F。若增加正四次项，三个分支及教师都要同步定义，不能只修某一分支。

建议最小几何族：32 个站点组成两个各 16 点的周期平面，平面内部有近邻、两组间距大于 MP 截断，并加入小幅位置扰动。晶胞可从 11.2 × 11.2 × 16 Å 起步；这是三维周期重复的两组电荷站点，不是已带 slab 修正的孤立双电极。

- 第一轮 128 train / 32 val / 64 test 个几何，geometry 先划分，再生成条件。
- conq：两组总电荷 Q、-Q，初始训练范围 Q ∈ [-0.5,0.5] e。
- conp：两组电势 V、-V，另加总电荷零约束，初始训练范围 V ∈ [-1,1] V。
- 两种边界均进入同一模型的训练；验证/测试使用未见几何和控制值/组合。这个对照不等于“只训练 conq 也保证任意 conp”。
- 为避免只有一个电势差就能猜出全部条件，增加组内零和的局部驱动，例如某两个站点分别 +delta、-delta；初始幅度 0.1 V。它不会只通过改变组总量来制造简单全局信号。
- 每个训练 geometry 初始 2 个 conq、2 个 conp 条件，另各附带 1 个组内零和驱动条件，共 6 个条件；所有分支使用同一条件 manifest。
- 稳定性检查针对完整受约束 Hessian。只保证 eta 为正不够。先在至多 20 个训练校准几何上确定固定教师系数、Ewald 截断及非病态范围，然后冻结数据生成参数；不能按测试表现反调教师。
- 任务有效性检查：固定左组局部输入、对右组施加组内零和驱动，参考解左组应有可解析检测的响应。若远组响应小到只剩数值噪声，不能用该任务证明全局模块收益。
- 64 原子新尺寸作为后续固定验证，不与首轮架构/深度/多个物理变化同时展开。

先做固定教师的 residual-only 求解对照，再做从头联合训练：提供教师平衡能量及共轭可观测量用于物性监督，精确逐原子状态用于评估而非默认训练 loss。conp 可用组电荷作为共轭响应；conq 可用电势差/乘子差；标签必须与所报告的热力学势一致。

这需要明确补齐可微条件输入：控制值和约束目标作为 live tensor 进入编译后的问题及 proposal，不能先转成 Python float 再希望对它们求导。conp 的负电势导数和 conq 的电荷目标导数，均从预测状态热力学势沿完整 proposal 路径计算。尚未驻定时，简单汇总预测电荷或读取估计乘子，不自动等于该完整导数；参考平衡点才可使用相应包络关系。加入这部分导数及有限差分测试后，再开启 B 的共轭物性训练。

固定教师检查的是求解能力；联合训练检查在学习物理泛函时是否仍能保持驻定和正确物性。两者结果分别报告。

### 6.3 已有其他数据的处理

已发现 TRECI 236 几何 × 8 条件，只有 15 个几何在所有条件下同时具有电子数、实际电势和 Bader 标签；其 manifest 明确标记自由电荷映射及巨势语义待审计。PySCF 周期 H2 当前只有 9 条状态记录，能支持跨总电荷，不能验证电极恒势或电容。

因此它们不替代首轮周期数值对照，也不立即进入大规模新训练。后续真实 conq/conp 科学验证先补齐电势参考、热力学能量、电子数和状态映射审计。数值任务通过不等于这一证据已完成。

### 6.4 共同训练设置和推进上限

第一轮固定结构以外的训练因素：

- 浮点：float64；Adam lr=1e-3；gradient clip=10。
- QM/MM 原条件能量、偶极、极化率权重各 1，沿用训练集 RMS 标度；stationarity 权重 0.1、现有 residual_scale=1 V/Å。
- 周期电荷使用自己的物理单位和训练集物性标度，stationarity 初始 0.1、residual_scale=1 V；不把两任务的原始 loss 数值横向比较。
- 新增 proposal 模块全部收到 residual 梯度；不在本轮同时调 residual 权重、骨干学习率比例或 teacher loss。
- 第一轮 seed=17；初步通过后以 23、31 验证重复性。旧冻结诊断抽样 seed 仍使用 manifest 中的 20260909。
- 全部完整训练保持一个 optimizer、一次联合 backward；固定泛函诊断不是生产训练的强制第一阶段。

| 阶段 | 拟执行量 | 推进条件/预算上限 |
|---|---|---|
| 接口与求导验收 | 小结构、2 个优化步、旧合同回归 | 每项 CPU/GPU smoke 最多 10 分钟；错误先修复，不重复提交完整作业 |
| 固定泛函结构诊断 | A 三分支、B 四分支；各 600 步，固定数据顺序 | 每分支最多 1 GPU 小时；达到上限保留未完成记录，不与满步结果混比 |
| 联合训练 pilot | A 和 B 同任务内各分支匹配，seed=17，最多 20 epochs / 2000 steps，先到为止 | 每分支最多 2 GPU 小时，pilot 阶段总计最多 14 GPU 小时 |
| 扩大训练 | pointwise 与通过的候选，seeds=17/23/31，最多 100 epochs，早停 patience=20 | 先依据 pilot 实测步时给出成本；本计划未提交完整训练队列 |

上述是待执行预算，不是已经花费或已获得结果。GPU 编号不写死；启动时读取实时占用，不中断其他任务。每个实际 run 开始前保存源码快照、完整配置、数据哈希、seed、条件列表和性能计时设置。

## 7. 验收：学会非局部响应还不等于完成 ACVR

### 7.1 必要的工程测试

1. 三个状态输出均满足原硬约束；复杂约束保留通用投影，不把逆硬度投影当作欧氏投影的等价替换。
2. scalar/vector、PaiNN/MACE 以及新 NequIP 适配后的特征可以前向和反传；高阶通道真实影响可配置输出，未被静默丢弃。
3. 旋转/反射/平移、原子排列、电极组重命名及周期镜像平移的一致性。
4. 多结构和同几何多条件之间没有 attention、cache 或增广字段串扰。
5. residual-only backward 只更新 proposal 所属参数；物性监督仍能到达应更新的各参数组。
6. 预测能量的力、外场一阶/二阶导数通过小结构有限差分或 gradcheck；attention 的双分支值一致。
7. Ewald 核能量、charge gradient、坐标导数一致；更换 cell/positions 后不会错误复用缓存；额外邻居范围覆盖完整实空间项。
8. 训练预测路径不调用参考状态求解器。通过拦截 solve_dense/solve_functional_reference 等入口检查；不禁止小约束投影使用 solve。
9. checkpoint 架构/单位/边界约定恢复检查、短程训练恢复和后续步的一致性。
10. required_labels 对原条件仍严格生效；新增 residual-only 条件不会错误继承原物性标签。

### 7.2 每次评估同时输出

- 约束误差最大值与失败数。
- 原始 proposal 的物理单位驻定 RMS、95 分位、最大值；同时报告归一化 residual。
- 到同一泛函参考平衡解的状态 RMSE/相对误差与能量差。
- 局域驱动到本原子、近邻、远组的状态响应误差；不能只以“出现非零响应”作为通过。
- 预测状态物性与精确平衡物性分别对独立标签的误差。
- conq/conp 及新几何、新控制组合、新尺寸分别统计。
- 前向、边界处理、Ewald 构建/应用、物性导数和参考求解的分项成本；完整耗时包含实际必要步骤。
- 参考失败、非稳定样本、NaN、预算超限保留在分母中并单列原因，不能静默删除。

候选晋级的预设初始标准：

- 无结构性测试失败、无参考不稳定问题；
- 固定泛函验证驻定 RMS 相比同预算训练的 pointwise 至少降低 2 倍；
- 跨原子响应误差相比 pointwise 至少降低 2 倍；参考响应显著非零的样本，相对范数误差目标不高于 20%；
- 联合训练独立物性验证 RMSE 不超过新 pointwise 对照的 1.05 倍，并且精确平衡物性与预测物性的差距缩小；
- 加速只在相同目标精度下报告。若候选更准但更慢，报告精度/成本折中，不宣布摊销加速。

这些是预先设置的工程/候选筛选目标，不是已证明能达到的结果或论文精度标准。若所有分支均未通过，优先定位输入缺失、增广未到达模型、梯度路由或移动目标；不直接启动更大的超参数网格。

### 7.3 模型选择和避免测试集回看

固定验证子集每 5 epochs 求一次参考平衡解。普通训练仍记录原来的 joint predicted loss；另外保存符合物性门槛时状态/驻定最好的 checkpoint，不能只按预测物性选模型。

架构、层数和学习率调整只使用 train/val。旧 QM/MM test 的历史结果已经查看过，明确记录这一事实；本轮不反复用 test 调参。新周期任务 test 在方案和 checkpoint 选择完成后才评估。

## 8. 本轮完成的基础核验与下一步

2026-09-11 在本工作区 CPU 单线程执行：

~~~bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
../.venv/bin/python -m pytest -q \
test/test_ewald_theory.py test/test_acvr_charge_runtime.py \
-k 'kernel_reproduces_energy_and_charge_gradient or cell_shapes_and_ewald_parameter_independence or real_ewald_runtime_relaxed_force_matches_finite_difference'
~~~

结果：5 passed, 15 deselected，6.92 秒。范围：两类晶胞的 Ewald 参数一致性、中性/非中性核与能量/电荷梯度一致性、已有 charge runtime 的平衡力有限差分。这只是可复用基础的工程核验，不是三个新 proposal 的可行性或性能结果。

下一步实施顺序：

1. 先实现共同特征/边界接口、pointwise/mp/attention 及必要梯度测试。
2. 接入 scalar Ewald 通信、共同 Ewald 电荷泛函及周期数值任务。
3. 运行固定泛函诊断，确认信号路径及真实响应误差。
4. 用相同物性/驻定目标启动有限规模联合训练，依据验证结果决定扩大哪些候选。

本计划不要求把模型拆成先训练物性、再训练 proposal 的生产流程；结构诊断和正式联合训练是不同目的的实验。

参考： [等变张量积接口](https://docs.e3nn.org/en/stable/api/o3/o3_tp.html)、[注意力原论文](https://arxiv.org/html/1706.03762v7)、[LAMMPS Ewald/PPPM 文档](https://docs.lammps.org/kspace_style.html)、[电极 conq/conp 文档](https://docs.lammps.org/fix_electrode.html)。
