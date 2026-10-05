# 有限温度 J_A 参考：主代理计算链路审核

工作目录：`GPUMD-branch_new_wpe-worktree`；分支：`codex/rpmd-ja-heat-current`；基线 HEAD：`c61d2a6c4d27eeece634edf4a30351e5f0c0610e`。

本轮实现由 gpt-6-luna/high 子代理负责，主代理提供方案、反馈和源码/数值审核，没有直接修改生产代码或测试。没有提交或推送。

## 1. 保持的计算目标

\[
J_{A,\alpha}=J^{\rm native}_{c,\alpha}[U]+x_c^T\Delta H_\alpha v_c.
\]

修改的是辅助参考模型和离线系数生成。真实势 U、RPMD 传播和原生质心机械流保留。稳定辅助矩阵不能证明强非谐量子热输运精度；有效频率也不能直接当作材料真实谱峰。

## 2. 输入与收集

入口：`tools/rpmd_ja_fit_reference.py collect`。输入是完整小体系在同 Hamiltonian、同固定晶胞、同温度和同 P 下的平衡 bead 位置及物理力。力必须与采样能量相容，不含环弹簧，不能替换为 F(Rc)。不允许从大体系任意裁切若干原子后把它们当成完整小体系正则样本。

每个 bead 的帧时标、原子序和晶胞必须相容。相邻 bead 链连续对齐，拒绝非零环 winding 和半胞歧义；进一步比较链质心与 GPUMD 的 bead0-MIC 质心，拒绝两者不一致的周期分支。时间方向连续展开并去除整体质心漂移；采样稀疏造成的整胞运动混叠仍需要数据质量检查，程序没有凭空设定 dt 门限。

实际 dump 的 Lattice 只写 8 位小数，因此真实数据应提供 `--cell-model` 指向原始完整精度固定晶胞输入。用原晶胞对齐并生成 samples/mean，而不是放宽生产晶胞匹配门限。

输出：`samples.npz` 与训练段平均结构。前 2/3 连续帧用于拟合和确定 R0，后 1/3用于独立核验。轨迹仍需独立块重采样及相关时间分析，合成精确协方差测试不能代替这些统计检查。

## 3. 基准导数与拟合

先用训练平均结构和原 qNEP/PPPM 配置调用 `rpmd_ja generate_raw`，得到与 R0 绑定的 raw v1/v2/v3 导数文件。新增命令复用已有 writer；能量—力、差分精度、平移和 Hessian 对称等基准诊断仍按 raw 版本独立检查。

拟合入口：`tools/rpmd_ja_fit_reference.py fit`。先检查 N、类型分区、质量、晶胞、温度、训练均值与 raw R0；内部维数限制在大矩阵读取之前生效。samples 相对 raw R0 的连续位移须处于生产既有 fractional 0.45 分支。包温度统一写 raw 温度，避免拟合与准备器浮点表达不一致。

\[
q=Z^TM^{1/2}(R_c-R_0),\qquad
f=Z^TM^{-1/2}\frac1P\sum_s F(R_s).
\]

参数是每个原子固定局域星的对称 Bi，包含邻居之间和 Cartesian 方向之间的耦合。当前图对每对不同原子只选一个 fractional MIC 周期像，不枚举多像或自像，因此这是受限参考空间，不是完整 qNEP 作用图。

用局域到全局刚度映射的代数核确定最小 Frobenius 分配规范；样本设计秩不足时拒绝，而不是用 ridge 掩盖。拟合不要求每个 Bi 正定，只要求总内部 D0 满足闭约束 D0 >= epsilon I。NumPy SVD 白化最小二乘、累积负方向切平面及对偶 QP 用完整谱和可行性/KKT复核。

独立核验比较 kBT D0^-1 与质心内部协方差；使用预测协方差白化后的最坏特征方向误差，避免平均误差掩盖硬模偏差。另核验力—位移积分分部关系与训练/验证力残差。受约束或截断拟合并不自动等于逆协方差响应。

## 4. 同一局域势同时生成 K 和 H

\[
U_{0,i}=U_i^t+\ell_i^TS_ix+\tfrac12(S_ix)^TB_i(S_ix),
\quad K_0=K_t+\sum_i S_i^TB_iS_i.
\]

ell 通过连通图线性平衡抵消 raw 的总梯度；该参考一次项在同一原生局域能量/virial/运输约定下对二次 H 的净贡献为零。准备器仍显式检查梯度抵消。

\[
(H_{{\rm add},\alpha}^Tx)_{j\mu}
=-\sum_{ie:j_e=j}d^0_{ie,\alpha}
\sum_{e'\nu}B_i[e\mu,e'\nu](S_ix)[e'\nu].
\]

公开 Python 组装器返回物理 Cartesian H：行是位移，列是速度。H一般非对称，不能对称化。GPJAADD1包存原始局域参数、固定像、P/T、数值裕量、响应核验和 raw 来源指纹，避免拼接来自不同参考的预计算 K/H。

## 5. prepare、稳定性证书与文件方向

入口：GPUMD `rpmd_ja prepare raw outfile kernel [pack]`，或 CPU `tools/rpmd_ja_qnep_prepare.py`。旧输入保持有效。

基准质量加权反对称及投影变化、fine/coarse误差在添加增量前独立检查，不能被大的精确 Kadd/Hadd稀释。C++ Kadd使用与基准相同的质量加权、Householder变换和平移投影。Hadd以转置方向和两端质量因子加入已有运输矩阵。

最终 v3 存储：D=M^-1/2 K0 M^-1/2，`block_site_transpose`=M^-1/2 H0^T M^-1/2。不改变现有 v3 二进制布局。CPU增量准备对D/H均写无损块；C++准备对D写无损块，H沿用既有逐tile SVD和重建误差门限1e-8，不能把两条路径都描述为所有矩阵无损。

对内部 D0-epsilon/2 I做 Cholesky；清理 column-major因子未用上三角后用LL^T计算完整Frobenius误差eta，要求eta<epsilon/2。C++另从最终实际存储D重新变换并复核重建；写入D始终未移位。传统三个探针只作附加诊断。新 policy与v4 sidecar严格配套，不能借用旧证书；P在生产阶段与实际bead数比较。

## 6. 映射与生产每帧计算

在内部质量加权本征基E上，A=E^T M^-1/2 H0 M^-1/2 E。既有谐波映射给出Atilde和DeltaH。缓存读取的是At=A^T；令ta=tau^2 lambda_a，缓存算子元素为

\[
K^{cache}_{ab}=t_a t_b P_{ab}A^T_{ab}+t_b Q_{ab}A^T_{ba}.
\]

运行时y=E^T sqrt(M)xc，u=E^T sqrt(M)vc，收缩u^T Kcache y，等于xc^T DeltaH vc。谱必须位于kernel覆盖范围。该缓存构造、真实质心流和积分器没有改写。

运行前核对模型/机械配置指纹、质量、类型、晶胞、温度、P与稳定性文件；运行中逐步更新连续质心并检查固定参考分支及有限值。生产结果保存Jcent、DeltaJ、JA，检查JA=Jcent+DeltaJ。

`hac_rpmd_ja.out`输出方向上的CAA及积分；四项关联需由`heat_current_rpmd_ja.out`离线重建：CAA=Ccc+CcDelta+CDeltac+CDeltaDelta。只看DeltaJ自相关不足以检验最终输运。

## 7. 审核状态与边界

主代理已独立复跑以下五个脚本，均 exit 0：

- `tests/rpmd_ja_fit_reference_test.py`：拟合、QP、最坏响应/IBP、图规范、raw诊断、收集/分支/精度、失败清理。
- `tests/rpmd_ja_additive_prepare_test.py`：真实CPU prepare/读回、一般K/H、不等质量、raw版本、Householder、因子方向和清理。
- `tests/rpmd_ja_additive_workflow_test.py`：真实subprocess collect/fit/CPU prepare；独立局域能量/virial差分；零附加修正与无pack旧路径比较；非零恢复；实际v3 D/H/P/Q读回及独立Fock权重比较；同K不同局域分配的DeltaJ敏感性；逐lag四关联重建；温度canonical和坏输入/响应拒绝。
- `tests/rpmd_ja_qnep_prepare_test.py`：旧准备器数值回归。
- `tests/rpmd_ja_measurement_test.py`：既有测量数值/源码契约。

Python工具及上述测试的`py_compile`通过。独立C++14静态复核和主代理链路审核未留下已知明确的新正确性缺陷；这不是所有输入上无bug的证明。`git diff --check`通过，Git仅提示平台换行转换。

工作流的raw由合成fixture写入；真实GPUMD/qNEP raw writer调用需要CUDA环境，尚未执行。合成bead坐标/力采用完整精度以严查谐波零恢复，晶胞仍模拟8位dump写入；实际dump的坐标/力量化、平衡数据和独立训练块的参考估计误差仍需验证。真实采集必须提供`--cell-model`；不提供的旧调用只适用于已有充分精度晶胞的XYZ，不能靠放宽生产门限恢复已丢失的精度。

环境缺少nvcc及C++编译器；CUDA fixture尚未编译或运行。不能宣称GPU构建、运行数值、内存峰值、速度或材料热导率已验证。

当前拟合明确限制internal dimension<=384、局域参数<=1600、主要dense对象<=6,000,000元素、NPZ载荷<=256MiB；prepare增量路径另限制d*d<=4,000,000及包<=64MiB。完整bead轨迹仍读入内存。不能据此直接给约17,685内部自由度的大体系生成参考。

科学验收还需要真实平衡分支、势能—物理力一致性、独立块统计、图截断/尺寸/P收敛、参考分配敏感性以及小量子体系和材料输运验证。若分支、可识别性、响应或资源条件不满足，应停止并报告具体原因。

## 8. 2026-10-05 未提交改动的追加审查

主代理重新沿收集、拟合、准备、文件读取、缓存与JA/HAC输出检查了未提交修改。以下修复均由新的gpt-6-luna/high子代理落实，主代理只提供复现与反馈、审核源码、修改本记录及复跑测试：

1. **中心梯度范数溢出。** Python的普通平方范数会把有限的1e200梯度算成Inf；Inf与Inf比较可放过未抵消梯度，主代理实际复现了错误写出参考。fit净平移梯度及prepare抵消检查改用稳定hypot范数并检查运算结果有限；正确取消的1e200梯度仍可通过。
2. **运算结果非有限仍可序列化。** 有限的正负1e308坐标相减会使边几何及Hadd溢出，旧CPU路径仍会写出v3。共享组装器检查K/H/gradient结果，准备器检查质量加权/增量组合结果，失败不会留下final、sidecar或work。旧大矩阵的新增有限扫描按tile进行，避免整块布尔临时阵。
3. **两个R0。** mean容差检查通过后，旧拟合仍围绕训练均值，而prepare/runtime围绕raw positions。微小合法偏差使日志残差近零、实际参考残差却非零。现在训练均值只用于输入核验，后续graph、位移与残差统一使用raw R0；回归独立核对真实中心下的残差。
4. **证书对象。** CPU symmetric tile writer会镜像上三角，实际存储D应以dcomp为准。主代理通过真实多tile writer构造了写前重建eta约0.953<epsilon/2=1、写后内部最小特征值为-0.09的反例。证书现在从dcomp重算内部矩阵和平移残差；测试包含多tile差异和真实prepare的存储矩阵拒绝/清理路径。
5. **Step-only接口。** collector支持整数Step字段，但旧输出steps为int64，被自己的fit浮点dtype约定拒绝。现在保存时canonical为float，并实际测试Step-only collect到fit成功；Time流程继续通过。

主代理在全部修复后独立复跑五个Python脚本，均exit0：fit_reference、additive_prepare、additive_workflow、旧qnep_prepare与measurement。修改的Python文件语法编译和git diff检查也通过。新溢出负例中的NumPy告警是刻意触发拒绝条件，未产生可用成品。

本轮没有更改真实势、积分器或生产JA测量公式；C++/CUDA新路径的静态检查没有发现需要追加修复的明确问题。编译器缺失、GPU运行与材料验证边界仍与第7节一致。所有改动仍未提交或推送。
