# RayTraceDicom Python 绑定（pybind11）

本工程新增了一个 `pybind11` 绑定模块 `raytracedicom_py`，用于从 Python 侧直接调用 `subsecondWrapper()`（即 `src/core/raytracedicom_wrapper.cu` 内的 GPU 剂量计算总入口）。

> 注意：该模块依赖 CUDA Runtime（`libcudart.so` 等）以及可用 GPU。请在带 CUDA 的环境中编译/运行。

## 1. 生成 Python 模块

```bash
mkdir -p build && cd build
cmake .. -DRTD_BUILD_PYTHON_BINDINGS=ON
cmake --build . -j
```

生成物默认输出到：

- `build/python/raytracedicom_py*.so`

## 2. Python 侧调用接口

模块导出函数：

- `raytracedicom_py.raytracedicom_wrapper(...)`

返回：`np.ndarray`（float32），形状为 `(Z, Y, X)`。

### 函数签名（简化说明）

```python
raytracedicom_wrapper(
    ct: np.ndarray,
    ct_dims: (int,int,int),
    ct_resolution: (float,float,float),
    ct_corner: (float,float,float),
    dose_dims: (int,int,int),
    dose_resolution: (float,float,float),
    dose_corner: (float,float,float),
    beam: dict,
    energy: dict,
    gpu_id: int = 0,
    nuclear_correction: bool = False,
    fine_timing: bool = False,
    tables_dir: str = "tables/",
) -> np.ndarray
```

### beam dict 必要字段

- `energies`: list/np.ndarray，长度 = 能量层数
- `spot_sigmas`: Nx2（每层 spot sigmaX/sigmaY）
- `steps`: int，ray tracing 步数
- `subspot_data`: shape = `(num_layers, max_subspots_per_layer, 5)`
  - 5 个分量依次为：`deltaX, deltaY, weight, sigmaX, sigmaY`
- `max_subspots_per_layer`: 可不传（由 `subspot_data` 推断），但建议保持一致

CarbonPBS 几何字段（必须给）：

- `beam_direction`: (3,)
- `beam_xdir`: (3,)
- `beam_ydir`: (3,)
- `sad`: float
- `source_position`: (3,)

可选字段：

- `ray_spacing`: (2,)（不传则 wrapper 内会回退）
- `source_dist`: (2,)（不传则 wrapper 内回退为 SAD）
- `spot_offset`: (3,)
- `spot_delta`: (3,)
- `ref_plane_z`: float

### energy dict 必要字段

- `n_energies`: int
- `n_energy_samples`: int
- `energies_per_u`: list/np.ndarray
- `peak_depths`: list/np.ndarray
- `scale_facts`: list/np.ndarray
- `cidd_matrix`: shape = `(n_energies, n_energy_samples)`

密度/阻止功率/辐射长度 LUT（可选）：

- 如果 energy dict 不提供以下字段，模块会从 `tables_dir` 读取参考表，仅用于填充 LUT：
  - `density_vector`, `density_scale_fact`, `n_density_samples`
  - `sp_vector`, `sp_scale_fact`, `n_sp_samples`
  - `rrl_vector`, `rrl_scale_fact`, `n_rrl_samples`

