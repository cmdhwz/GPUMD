# RPMD J_A 任务断点（2026-10-04，用户要求暂停）

## 恢复后的当前状态（2026-10-04）

已由一个 gpt-6-luna/high 子代理完成必要源码接入，主代理已复核公式、布局、缓存状态及生成—准备—读取—HAC 链路。仍未提交或推送。子代理上下文使用 fork none，只传必要任务，约 250K token 上限。

新增 NEP_Charge 参考解析站点 JVP 和能量梯度；raw v2 使用解析 V、差分解析能量梯度 K、差分原生九分量 virial C。原有能量差分保留为诊断，不再决定解析路径的验收。解析梯度/native force 和 JVP/梯度的绝对 RMS 检查仍为 1e-4，K/C 收敛检查仍为 0.05。参考生成仍可能因真正的离散势/力不一致或不稳定参考而拒绝，服务器问题没有完成数值验证。

prepare、reader 和 pre_run 支持旧 raw1/reference，并传递新来源标记 `native_reference_transport;analytic_site_gradient_v1`。参考生成移除不再调用的四路 batch scratch，固定 R0 先完成 V，再逐帧求 K/C；两阶段共约 10 次进度打印。生产测量仍不调用新增导数接口，不增加每个采样点的矩阵动作。

通过：现有 CPU `rpmd_ja_qnep_prepare_test.py`、静态 `rpmd_ja_measurement_test.py`、静态 `test_qnep_ja_station_operator_contract.py`、`git diff --check`。本机仍无 nvcc；新增 PPPM CUDA fixture、raw2 prepare CUDA fixture、正常 make -j8、小体系/服务器模型运行及性能实测均未执行。

5896 原子的 raw 矩阵存储估算约 17.87 GiB；QEvaluator 已跟踪 GPU 缓冲约 1.06 MiB，排除 NEP/PPPM 私有缓存及主机缓冲，不代表生成总峰值。当前串行生成估算 70814 次原生势求值（4 类型各 3 个采样坐标），17700 次站点 JVP；每次 JVP 一对 FFT，gradient-only 无额外 FFT。实际耗时由运行日志报告。

以下为上一次暂停的历史断点，凡与当前状态冲突，以本节为准。

状态：已停止；不再启动下一轮。源码修改由一个 gpt-6-luna/high 子代理落实，主代理给公式和审核。没有提交、合并或推送。

工作树：`D:\计算文件\各种软件\GPUMD-4.7\GPUMD-5.3\GPUMD-branch_new_wpe-worktree`

分支：`codex/rpmd-ja-heat-current`；起点 HEAD：`86f71183fdcc17fa55d17c125feca0fa1f3fd15f`（本轮开始时洁净）。

## 已保存的本轮代码

- `src/force/pppm.cuh`、`src/force/pppm.cu`：新增参考专用 `PPPM::compute_reference_energy_tangent`，支持站点能量 JVP，以及不做 FFT 的 gradient-only 调用。独立 δQ/δΦ 缓冲区，复用当前单帧的 G、Φ 和 ik 场；需要有效 force-frame ID，位置先展开至同一主周期晶胞。切线一对 FFT，梯度只做场插值；尺寸不变时不重新分配显存。
- `tests/pppm_reference_tangent_cuda_test.cu`：新增实际调用 PPPM 的 CUDA fixture，包含逐原子能量差分、h/h2、非零中性 q/dq、PBC 跨界、零/重复 JVP、gradient-only、错误 frame ID 及 adjoint 恒等式检查。测试代码已保存，尚未执行。

本轮代码还没有接入 NEP_Charge、QEvaluator、raw 矩阵生成或 diagnose。当前参考生成仍是原有有限差分路径，不能宣称服务器报错已经解决。

## 本轮定位与审核

5896 原子服务器诊断：网格由约 30³ 加密到 60³，没有明显改善能量梯度误差。完整 qNEP 的 h=0.001/0.01/0.1 Å 能量梯度绝对 RMS 约为 0.164/0.0387/0.00753；短程对应约 0.0104/0.000912/0.00960。没有通过原有 1e-4 eV/Å 检查。不能据此单独判定 PPPM 原生力错误，也不能提高阈值绕过。

当前能量输出数组为 double，但局域 ANN、charge、网格内部仍有 float 运算；标量能量差分有舍入噪声。另一个必须独立测量的问题是 PPPM 原生 ik 力与离散插值能量梯度的区别。

主代理已交回并由子代理修正：FFT 状态成员名、可选指针组合的空指针风险、无条件 GPU_Vector::resize、每线程多余 125 个 double 临时数组，以及 fixture 使用全局平均能量而非逐原子分配的问题。最终完整差异的复核和可执行 CUDA 验收仍待恢复后进行。

## 冻结的公式与后续工作

PPPM 原生逐原子倒空间能量是 U_i = K_C_SP q_i Σ_g W_ig Φ_g，无额外 1/2；Φ=IFFT[G FFT(Q)]，使用实际 G 和原生 FFT 归一。

δQ=Σ_i(δq_i W_i+q_i δW_i)，δΦ=IFFT[G FFT(δQ)]。

δU_i=K_C_SP[δq_i ΣW_iΦ+q_i ΣδW_iΦ+q_i ΣW_iδΦ]。

显式空间梯度 g_exp,i=2 K_C_SP q_i Σ(∇W_i)Φ。组合总能量梯度的符号为：grad_E = -F_native + F_pppm_ik + g_exp。不能修改传播用的物理力。

恢复后先复核并构建当前 PPPM 阶段，再派同一子代理实施 NEP_Charge 的局域能量/电荷 JVP（δq=δq_raw-mean(δq_raw)）、mode-1 实空间/自能项、ZBL 和小盒固定周期像。D3 仍按既有接口明确拒绝，不扩展支持。

之后才接入参考生成：V 用解析站点导数，K 用解析总能量梯度的差分，C 保留原生完整九分量 virial 的 h/h2 差分；几何项使用 F_E=-grad_E。先固定 R0 完成 V 列，避免每列重新评价 R0。诊断必须分别报告 JVP/解析梯度恒等式和解析梯度/native force 差异，不能把复制的解析 V 伪称为通过有限差分收敛。

raw/参考文件的导数来源标记与兼容格式尚未冻结；不能静默改变既有诊断字段语义。原 J_A 定义、完整量子核、NEP 路径、RPMD 积分器、bead 物理状态及生产测量开销均不在本轮修改范围。

## 验证状态

`git diff --check` 通过，仅有 Git LF/CRLF 提示。当前环境未发现 nvcc 或可用 C++ 编译器；新增生产 CUDA 代码未编译，fixture 未运行，服务器模型未验证，性能没有实测。

待执行（Linux/CUDA，仓库根目录）：

```bash
nvcc -std=c++17 -arch=sm_89 -Isrc tests/pppm_reference_tangent_cuda_test.cu \
  src/force/pppm.cu src/model/box.cu src/utilities/error.cu \
  -lcufft -o pppm_reference_tangent_cuda_test
./pppm_reference_tangent_cuda_test
```

正常 GPUMD 构建仍使用 `cd src && make -j8`；上面是额外检查程序的待验证命令，不是生产运行必须单独编译的模块。
