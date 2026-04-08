# Curator-QEQ LAMMPS 接口实现方案

## 1. 能量分解原理

### 1.1 完整的 Ewald 求和分解

总静电能量 (Ewald) = 实空间部分 + 倒空间部分 + 自能校正

$$E_{Ewald} = E_{real} + E_{recip} + E_{self}$$

其中：
- $E_{real}$ = 短程实空间部分 `erfc(α·r)/r` → 由 `pair_style coul/long` 计算
- $E_{recip}$ = 长程倒空间部分 (k-space sum) → 由 `kspace_style ewald/pppm` 计算
- $E_{self}$ = 自能校正 `-α/√π · Σq²` → 由 `kspace_style ewald/pppm` 计算

### 1.2 Curator QEQ 模型的能量组成

训练时的总能量：
```
E_total = E_ML_short + (E_ewald + E_residual) * ewald_weight
        = E_ML_short + E_real + E_recip + E_self + E_residual
```

其中：
- `E_ML_short`: MACE 短程 ML 能量
- `E_ewald`: 完整 Ewald 能量 (real + recip + self)
- `E_residual`: χ·q + η·q² (电负性和硬度项)

### 1.3 LAMMPS 接口的能量分配

| 能量部分 | Curator 输出 | LAMMPS 计算 |
|----------|--------------|-------------|
| ML 短程能量 | ✅ `short_energy` | `pair_mliap` |
| 残差能量 (χ·q + η·q²) | ✅ `residual_energy` | `pair_mliap` |
| Ewald 实空间 | ❌ | `pair_style coul/long` |
| Ewald 倒空间 | ❌ | `kspace_style ewald/pppm` |
| Ewald 自能 | ❌ | `kspace_style ewald/pppm` |
| 原子电荷 | ✅ `atomic_charge` | 写入 `atom->q` |

## 2. 实现架构

### 2.1 架构设计

```
┌──────────────────────────────────────────────────────────────┐
│                    LAMMPS Verlet Integrator                  │
│                                                              │
│  ┌─────────────────────┐     ┌─────────────────────────┐    │
│  │   pair_mliap        │     │   kspace_style ewald    │    │
│  │  (短程 ML 力)       │     │   (长程 Coulomb 力)     │    │
│  │                     │     │   使用 ML 预测的电荷    │    │
│  └──────────┬──────────┘     └───────────┬─────────────┘    │
│             │                            │                   │
│             └────────────┬───────────────┘                   │
│                          ▼                                   │
│                f[i] = f_ML_short + f_kspace_long             │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 需要修改的文件

#### 2.2.1 扩展 mliap_unified_couple.pyx
```python
# 新增属性用于访问原子电荷
@property
def charges(self):
    """Get atom charges array (atom->q)"""
    if self.data.atom_q is NULL:
        return None
    return np.asarray(<double[:self.ntotal]> &self.data.atom_q[0])

@write_only_property
def charges(self, value):
    """Set atom charges from ML predictions"""
    cdef double[:] charges_view = <double[:self.nlocal]> &self.data.atom_q[0]
    cdef double[:] value_view = value
    charges_view[:] = value_view
```

#### 2.2.2 修改 mliap_data.h
```cpp
class MLIAPData : protected Pointers {
  // ... existing members ...
  
  // 新增: 暴露 atom->q 给 Python
  double *atom_q;  // pointer to atom->q
  
  void init() {
    // ...
    atom_q = atom->q;  // 初始化时获取电荷指针
  }
};
```

#### 2.2.3 新建 curator/simulate/lammps_mliap_qeq.py

```python
class LAMMPS_MLIAP_QEQ(MLIAPUnified):
    """Curator-QEQ integration for LAMMPS using ML-IAP + kspace"""
    
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model  # Curator MACE-QEQ model
        # ... initialization ...
        
    def compute_forces(self, data):
        """
        Main compute function called by LAMMPS.
        
        Steps:
        1. Forward pass: get atomic energies, edge forces, and predicted charges
        2. Update atom->q with predicted charges
        3. Return short-range forces to LAMMPS
        4. LAMMPS kspace will automatically compute long-range forces
        """
        # 1. Prepare batch
        batch = self._prepare_batch(data)
        
        # 2. Model forward - get short-range energy/forces AND charges
        out = self.model(batch)
        atom_energies = out['atomic_energy']
        edge_forces = out['edge_forces']  
        predicted_charges = out['charges']  # ML predicted charges
        
        # 3. Update LAMMPS atom charges for kspace
        # This requires extended MLIAPData
        data.charges = predicted_charges.cpu().numpy()
        
        # 4. Update short-range energy and forces
        data.eatoms = atom_energies.cpu().numpy()
        data.energy = atom_energies.sum().item()
        data.update_pair_forces_gpu(edge_forces)
        
        # LAMMPS will call kspace->compute() after pair->compute()
        # to calculate long-range Coulomb forces using the charges we set
```

### 2.3 LAMMPS 输入文件示例

```lammps
# LAMMPS input script for Curator-QEQ
units           metal
atom_style      charge  # 需要支持电荷

# ... 结构定义 ...

# Pair style: ML-IAP with Python unified interface
pair_style      mliap unified curator_qeq.pt

# Kspace style: Ewald or PPPM for long-range Coulomb
kspace_style    pppm 1e-4

# ML-IAP 计算短程力并更新电荷
# kspace 使用这些电荷计算长程力
# 两者自动叠加
```

## 3. 实现步骤

### Phase 1: 验证概念
1. [ ] 确认 LAMMPS 可以使用动态更新的电荷
2. [ ] 测试 pair_mliap + kspace 的组合是否工作

### Phase 2: 扩展 ML-IAP
3. [ ] 修改 `mliap_data.cpp/h` 暴露 `atom->q`
4. [ ] 修改 `mliap_unified_couple.pyx` 添加 charges 属性
5. [ ] 重新编译 LAMMPS

### Phase 3: 实现接口
6. [ ] 创建 `lammps_mliap_qeq.py`
7. [ ] 修改 Curator 模型输出电荷
8. [ ] 添加多 GPU 支持 (LAMMPS_MP)

### Phase 4: 测试验证
9. [ ] 单元测试: 电荷更新
10. [ ] 集成测试: 能量/力与纯 Python Curator 对比
11. [ ] 性能测试: 多 GPU 扩展性

## 4. 技术细节

### 4.1 电荷通信
当使用多 GPU 时，预测的电荷需要通过 LAMMPS 的通信机制同步到 ghost 原子：
```cpp
// kspace 计算前需要确保 ghost 原子的电荷是最新的
comm->forward_comm();  // 这会同步 atom->q 到 ghost
```

### 4.2 与 Curator Ewald 的对比

| 特性 | Curator Ewald | LAMMPS kspace |
|------|---------------|---------------|
| 实空间截断 | 需要包含所有原子 | 只需 cutoff 内 |
| 倒空间计算 | 全局 FFT | 分布式 FFT |
| MPI 并行 | 需要 gather | 内置支持 |
| GPU 支持 | PyTorch | Kokkos PPPM |

### 4.3 精度验证
确保 LAMMPS kspace 与 Curator Ewald 使用相同参数：
- α (g_ewald) = 0.4 (Curator 默认)
- accuracy = 1e-4 (LAMMPS 默认)
- k_cutoff 需要匹配

## 5. 注意事项

1. **Ewald α 参数一致性**: 如果 Curator 训练时使用特定的 α，LAMMPS kspace 应该使用相同的值

2. **实空间短程力**: Curator 的短程力已经包含了实空间 Coulomb 的 erfc 部分，所以我们需要：
   - 选项 A: LAMMPS kspace 只计算倒空间部分
   - 选项 B: Curator 只计算 ML 力，实空间 Coulomb 也由 LAMMPS pair_coul 计算

3. **电荷守恒**: ML 预测的电荷可能不精确守恒，需要归一化处理

4. **原子类型映射**: 确保 Curator 的元素顺序与 LAMMPS 类型映射一致
