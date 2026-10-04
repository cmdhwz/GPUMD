# 原 J_A 的 qNEP 原生完整 virial：参考算子推导与实现契约

日期：2026-10-03 起草，2026-10-04 补充完整链式项。推导与审核责任：主代理。实现责任：两个 gpt-6-luna/high 子代理；推导期间暂停，没有在本文件形成前派发新的长程实现任务。

工作树为 `GPUMD-branch_new_wpe-worktree`，分支为 `codex/rpmd-ja-heat-current`，本轮起点为 `24425b4ce9a145bddcad5b716998adaf98e81aab`。本文是推导和实施设计，不是 qNEP J_A 已实现、CUDA 已运行或 5896 原子性能已验收的声明。

## 1. 目标与已经确定的物理选择

用户选择每个采样点用同一 qNEP 在质心重新求值，取原生局域能量和完整九分量 virial；保留普通 virial 已有的电荷链式贡献。电荷取当前质心的 q(R_c)，不冻结 q，也不额外加入 `hac_current qnep_full_a` 的电荷传播热流修正。

目标外形保持为

\[
J_{A,\alpha}=J^{\rm native}_{c,\alpha}+x_c^T\Delta H_\alpha v_c,
\quad x_c=R_c-R_0,
\]

\[
J^{\rm native}_{c,\alpha}
=\sum_i\left[(\tfrac12m_i|v_i^c|^2+U_i(R_c))v_{i\alpha}^c
+\sum_\mu W^{\rm native}_{i,\alpha\mu}(R_c)v_{i\mu}^c\right].
\]

它是广延总流，不除以 bead 数或体积。BEC 是电极化响应观测量；下文需要的 dq/dR 和参考二阶导数是机械势链式微分的组成，不等于必须计算 BEC 张量。生产热流仍只在 HAC 采样点求值、保存历史，生产段结束后输出。

## 2. 实际源码路径：先纠正均分分支的误读

`src/force/pppm.cu:2262` 的开关为

```cpp
calculate_peratom_virial = need_peratom_virial || request_peratom_virial;
```

当质心观测调用 `request_peratom_virial_for_next_force()` 时，走 `find_force_virial_potential_from_field`（约 869、2401 行）。该分支按每个原子的电荷与场插值输出能量和 virial，不是将总量均分。没有请求逐原子 virial 的另一分支才调用 `find_potential_and_virial`（约 965、2450 行）均分总量。

因此，均分站点 virial 的小模型只能证明“任意站点分配不能自动套用第 44 章”，不能证明当前质心 PPPM 分支不匹配或有物理错误。前一轮据均分分支提出的困难须按实际场插值路径重新核实。

原生机械流需要九分量张量，这也与 [GPUMD 热流文档](https://gpumd.org/theory/heat_current.html)一致。但该文档没有证明当前 qNEP/PPPM 的所有站点分配都满足下面的固定边恒等式。

## 3. 第 44 章已有恒等式及其假设

固定参考周期边 d_ie^0，局域能量边梯度 a_ie=∂U_i/∂d_ie。若物理 virial 满足

\[
W^{\rm can}_{j,\alpha\mu}(R)=-\sum_{i,e:\,j_e=j}d_{ie,\alpha}(R)a_{ie,\mu}(R),
\]

则固定边流 W_can^0(R)=-Σ d_ie^0 a_ie(R)，其导数为

\[
H_{{\rm can},\alpha}^T=-\sum_iP_{i\alpha}^T\mathcal K_iS_i.
\]

令 V_i,jμ=∂U_i/∂R_jμ，(L_α x)_i=x_iα，(Ψ_α x)_jμ=x_jα F_jμ。对当前边求导，有

\[
\partial_x W^{\rm can}_{j,\alpha\mu}
=-\sum_{ie}(x_{j\alpha}-x_{i\alpha})a_{ie,\mu}
-\sum_{ie}d^0_{ie,\alpha}(\mathcal K_iS_ix)_{e\mu}.
\]

第一项来自边长随位置变化，第二项才是固定边的 H_can^T x。由 Σ_i V_i,jμ=-F_jμ，可化为

\[
\boxed{H_{{\rm can},\alpha}^T=C_{{\rm can},\alpha}-V^TL_\alpha-\Psi_\alpha},
\quad C_\alpha=\partial_R W_\alpha|_0.
\]

这是第 44 章式 44.15；它不能不加条件地用于任意 W_native。若 W_native=W_can+Γ，则

\[
C_{{\rm native},\alpha}-V^TL_\alpha-\Psi_\alpha
=H_{{\rm can},\alpha}^T+\partial_R\Gamma_\alpha|_0.
\]

Γ 的总应力之和为零也不保证 Σ_j Γ_j v_j=0。不能用“总力/总能量相同”代替站点流规范一致性检查。

## 4. 与原生分配匹配的参考运输定义

用户已授权推导原生分配的长程参考算子。一个明确、可微、回到普通 NEP 原定义的扩展是：令 z=R-R0，并**定义**

\[
\boxed{W_{{\rm ref},\alpha}^{\rm native}(R)
=W_\alpha^{\rm native}(R)-V(R)^TL_\alpha z-\Psi_\alpha(R)z.}
\tag{N1}
\]

所有 U_i、W_native 和 V 使用同一势、分配、晶胞和展开分支。这里 Ψ 使用 F_E=-∇Σ_iU_i 的能量导数力；只有证实原生数值力与 F_E 一致后才能用原生力代替。该定义在 z=0 时保留原生 W 的值，并除去位置运输的几何项。

逐项对 N1 求方向导数：

\[
\partial_x[V(R)^TL_\alpha z]
=(\partial_xV)^TL_\alpha z+V^TL_\alpha x,
\]

\[
\partial_x[\Psi(R)z]=(\partial_x\Psi)z+\Psi x.
\]

在 R0 处 z=0，前述乘积中的第一项消失，因此

\[
\boxed{H_{{\rm native},\alpha}^Tx
=C_{{\rm native},\alpha}x-V_0^TL_\alpha x-\Psi_{E,0,\alpha}x.}
\tag{N2}
\]

对于满足第 44 章边定义的 W，N1 就是该章的固定参考边流，N2 精确回到原 H_can。对于另外的原生分配，它是**原生参考运输扩展的定义**，不是已证明 H_native=H_can。参考文件必须记录这个差别，不能让普通 NEP 文件和 qNEP 原生文件互换。

本扩展只改变参考输入规范，外层仍为 J_c+xΔHv；没有把完整当前位置流的二次 Taylor 系数当作 H，也没有默认加入 J_H/J_G 极化补全。还需单独检验实际 qNEP 的 Γ、周期像和稳定参考条件，才能声称该原生扩展符合材料验收要求。

## 5. 为什么不能直接校准完整当前位置流的二次项

当前位置流的二次 Taylor 系数还包括势能携带项。若其矩阵记 T_α，则

\[
(T_\alpha^Tx)_{j\mu}=(C_\alpha x)_{j\mu}
+\delta_{\alpha\mu}(Vx)_j.
\]

它与 N2 的差为

\[
v^T(T_\alpha^T-H_\alpha^T)x
=\sum_jv_{j\alpha}(Vx)_j+\sum_i x_{i\alpha}(Vv)_i
+\sum_jx_{j\alpha}F_{E,j}\cdot v_j.
\]

稳定平衡参考 F_E(R0)=0 时，前两项是 d/dt[Σ_i x_iα(V_0x)_i]。将 T 放入完整核会把这部分也纳入二次校准，改变原站点参考目标；不能悄悄这样做。非平衡参考还含最后一项，不能由代数恒等式自动获得平衡 Gaussian 校准。

## 6. 平移、质量与固定线性的核对

对严格平移 t=1⊗e_β，若 U_i 和 W_native 整体平移不变，则 C_α t=0；又有

\[
V^TL_\alpha t=-\delta_{\alpha\beta}F_E,
\quad\Psi_{E,\alpha}t=\delta_{\alpha\beta}F_E.
\]

所以 H_native^T t=0，包括非平衡的代数检查。能量 Hessian K_E 同样湮灭平移。令

\[
D=M^{-1/2}K_EM^{-1/2},\qquad
B_\alpha^T=M^{-1/2}H_{{\rm native},\alpha}^TM^{-1/2},
\]

则质量加权平移为 M^(1/2)t；不能使用平均质量，不能对 B 对称化。

有限网格可能具有位置混叠误差；实际 PPPM 的平移残差必须测量。投影只能按已指定三维平移空间处理，并将投影改变量记入输入误差，不能宣称投影证明原系数正确。

生产的参考作用必须是 R0 处的固定线性算子。按当前方向 x 对 R0±δx 做有限差分有依赖 x³ 的余项，不能直接冒充固定矩阵乘法。小体系可先按坐标列缓存中央差分作独立基线；大体系需要参考解析/AD 的 JVP/VJP。

## 7. 从实际 PPPM 场插值推导长程作用

以下均冻结 R0、晶胞、网格、Green 核、温度及参数。用 a_i(R_i) 表示原子 i 的实际五点/方向插值权重向量，A=[a_1,…,a_N]；用 q 表示包括实际介电缩放、中和规则的电荷。定义

\[
\rho=Aq,\quad\mathcal G=\mathcal F^{-1}\operatorname{diag}(G_k)\mathcal F,
\quad g=\mathcal G\rho.
\]

FFT 归一化在 G 中按生产代码处理，不额外重复除网格数。设 c=K_C 为实际内部库仑常数，则场插值倒空间输出形式为

\[
U_i^{\rm rec}=c q_i a_i^Tg,
\quad W_{i,\alpha\mu}^{\rm rec}=c q_i a_i^T\mathcal G_{\alpha\mu}\rho.
\tag{N3}
\]

这里 G_αμ 与 `find_mesh_virial` 的核相同：对 k≠0，乘子为 G_k[δ_αμ−(2α_factor+2/k²)k_αk_μ]；k=0 按生产代码为零。不要把仅倒空间本身的对称六分量限制推广到完整 qNEP virial，局域链式项仍须保存九分量。

令 J_q=∂q/∂R，δq=J_qx，δa_i=(∂a_i)x_i。电荷中和 q=Pq_raw 时，J_q=PJ_raw，且反向作用 J_q^Tb=J_raw^TP^Tb。中和项不能省略。

沉积的完整方向作用为

\[
\boxed{\delta\rho=\mathcal T x=A J_qx+\sum_iq_i\delta a_i.}
\tag{N4}
\]

第一项是电荷变化，第二项是位置移动；它们是参考机械势的链式导数，不是新的逐帧电荷热流项。

由 N3 逐因子求导，得到倒空间 virial 的参考 JVP：

\[
\boxed{(C_\alpha^{\rm rec}x)_{i\mu}
=c[(\delta q_i)a_i^T\mathcal G_{\alpha\mu}\rho
+q_i(\delta a_i)^T\mathcal G_{\alpha\mu}\rho
+q_i a_i^T\mathcal G_{\alpha\mu}\mathcal T x].}
\tag{N5}
\]

计算 V^Tb 不需要生成 N×3N 的 V。令 ρ_b=A(b⊙q)，φ_i=a_i^Tg，先将加权站点能量写成

\[
\sum_i b_iU_i^{\rm rec}=c\rho_b^T\mathcal G\rho.
\]

对 q 和位置分别求导；利用 G 的自伴性，得

\[
\boxed{(V_{\rm rec}^Tb)
=cJ_q^T[b\odot\phi+A^T\mathcal G\rho_b]
+c\,\{q_i(\partial_{R_i}a_i)^T[b_i g+\mathcal G\rho_b]\}_{i=1}^N.}
\tag{N6}
\]

将 b=L_αx 代入，和 N5、Ψ 组合，即为 N2 的倒空间部分。三个方向共享 δq、δρ 等中间量；一次批量作用可以输出全部方向。实际实空间、self/ZBL、局域 NEP 能量和普通电荷链式 virial 必须分别按实际表达求 JVP/VJP 后相加，不能只实现 N5 而漏掉其余部分。

### 7.1 完整站点能量的反向链式微分

把实际站点能量写成 U_i(R)=U_i^loc(R)+E_i(R,q(R))，E_i 收集静电实空间、倒空间、自能及相应实际修正。固定 b，定义 E_b=Σ_i b_i E_i，则

\[
\boxed{V^Tb=(V_{\rm loc})^Tb
+\left.\partial_RE_b\right|_q
+J_q^T\partial_qE_b.}
\tag{N6a}
\]

这里 b=L_αx 在参考作用中是输入权重，不随求导变量再变化。N6 是 N6a 的倒空间展开。不同站点 q_i 的依赖通过最后一项完整保留，不可只微分中心电荷。

若某条实际实空间能量边为 E_ij=q_iq_j f(d_ij)，两端各分一半，则 E_b 的该边为 (b_i+b_j)q_iq_j f/2。因此固定 q 的位置导数具有两端相反的边梯度，q_i 的偏导为 (b_i+b_j)q_j f/2。遍历实际周期边就能求这两个作用，无须形成 V；自能、屏蔽及其他参数化形式必须按生产表达另外加入，不能假定所有 charge 模式都具有同一个 f。

### 7.2 普通电荷链式 virial 的三项导数

设有效未中和电荷为 q_raw，q=Pq_raw。用 Z_ae,μ=∂q_raw,a/∂d_ae,μ 表示参考局域电荷边导数（包括实际介电缩放）；D=∂E_el/∂q，Dbar=P^TD。原生局域电荷链式 virial 的结构为

\[
W^{\rm chain}_{j,\alpha\mu}
=-\sum_{a,e:\,j_e=j}d_{ae,\alpha}\,\bar D_a Z_{ae,\mu}.
\]

参考方向 x 下逐因子求导，而不是只对 D 求导：

\[
\boxed{(C_\alpha^{\rm chain}x)_{j\mu}
=-\sum_{a,e:\,j_e=j}
[(x_{j\alpha}-x_{a\alpha})\bar D_a^0Z_{ae,\mu}^0
+d_{ae,\alpha}^0(\delta\bar D_a)Z_{ae,\mu}^0
+d_{ae,\alpha}^0\bar D_a^0\delta Z_{ae,\mu}].}
\tag{N6b}
\]

其中

\[
\delta\bar D=P^T(E_{qR}x+E_{qq}\delta q),\qquad
\delta Z_a=\mathcal K_{q_{{\rm raw},a}}S_ax.
\]

第一项是边几何项，第二项含长程场及电荷耦合，第三项是电荷网络局域二阶微分；三者都必须保留。S_a 对每个周期像单独取相对位移，即使多个像对应同一个邻居速度也不合并边 Hessian 记录。局域短程能量 virial 同样求几何项和边梯度方向导数。

### 7.3 固定电荷实空间 virial 的导数

对实际对势边，若两端原生 virial 为 W_i,αμ=W_j,αμ=−q_iq_j d_α(∂_μf)/2，则

\[
\delta W_{i,\alpha\mu}
=-\tfrac12[(\delta q_iq_j+q_i\delta q_j)d_\alpha\partial_\mu f
+q_iq_j\delta d_\alpha\partial_\mu f
+q_iq_j d_\alpha(\nabla_d^2 f\,\delta d)_\mu],
\quad\delta d=x_j-x_i.
\tag{N6c}
\]

必须包括截断/屏蔽函数导数。该式说明实现的产品求导形式；最终必须对照 `find_force_charge_real_space` 的实际分配、常数和 charge mode，不能仅据通用对势式宣布整个实空间已经核实。

完整 C_native 是短程局域项、实际固定电荷实空间项、倒空间 N5、普通链式 N6b 及其他获准实际修正的和；完整 V^T 是同一能量分配的 N6a。然后统一组成 N2，而不是每个通道分别猜一个总 Hessian 流分配。

## 8. 总参考 Hessian 的经济作用

由 U_rec=cρ^TGρ、G 自伴，得

\[
F_E^{\rm rec}=-2c\mathcal T^Tg.
\]

再求方向导数：

\[
\boxed{K_E^{\rm rec}x
=2c[\mathcal T^T\mathcal G\mathcal T x
+(\partial_x\mathcal T)^Tg].}
\tag{N7}
\]

后一项不能省略；它含电荷网络的二阶导数、插值权重二阶导数和位置/电荷交叉项。若只保留前一项，会得到另一近似 Hessian，并可能漏掉负曲率。局域网络的所有多体交叉块也须保留。

完整复合势 U=U_loc+E_el(R,q(R)) 的总公式为

\[
\boxed{K_Ex=K_{\rm loc}x+E_{RR}x
+E_{Rq}J_qx+J_q^TE_{qR}x+J_q^TE_{qq}J_qx
+\sum_a D_a\nabla_R^2q_a\,x.}
\tag{N7a}
\]

E 的各偏导在固定另一变量时计算，全部取 R0。N7 是倒空间的等价网格分解，不应再与同一倒空间 N7a 重复相加。q=Pq_raw 时，最后一项等于 Σ_a(P^TD)_a ∇²q_raw,a x，可用加权局域电荷 Hessian 作用实现。完整 Hamiltonian 的实空间配对 Hessian、局域能量 Hessian、自能等加在其余对应通道中。

`pppm.cu` 的原生固定电荷力通过 ik 频域场计算；离散网格能量的精确梯度则包含插值权重导数。二者的一致性不可未经验证假定。因此需单独测量 F_native−F_E 和 −∂F_native−K_E 的残差、网格/步长收敛。参考 K 必须明确来自选定的同一能量 Hamiltonian；不能对不一致的力 Jacobian简单对称化后就宣布通过。

## 9. 与完整量子核连接的最终形式

正常模只用于小体系独立验收。生产不形成 ΔH，不逐帧对角化；继续使用第 43 章的完整稳定核和第 45 章已受控的 P/Q 系数递推。

令 y=M^(1/2)x_c、u=M^(1/2)v_c、d_y=Dy、d_u=Du，Z=2D/Λ−I。第 45 章分解给出

\[
\Delta J_\alpha\simeq
\tau^4\sum_r\lambda_r^P
[B_\alpha^Tp_r(Z)d_y]^T[p_r(Z)d_u]
+\tau^2\sum_r\lambda_r^Q
[q_r(Z)d_y]^T[B_\alpha^Tq_r(Z)u],
\quad\tau=\beta\hbar.
\tag{N8}
\]

这里低秩是标量完整核系数的压缩，不是默许截断材料长程 H。D 和 B^T 的新参考作用由 N2、N4–N7 提供。实际系数须沿用完整核表的符号、τ 幂和误差预算。τ=0 时 ΔJ=0；软模不得按任意阈值删除，真实负曲率不得取绝对值。

## 10. 数学独立验算与现有源码检查

主代理使用三原子、两组正弦/余弦倒空间基的独立双精度模型：q=P tanh(LR)，L 为相对坐标矩阵；G、G_vir 在每个正弦/余弦对上使用相同特征值。该模型有电荷链式依赖和严格整体平移不变性，未使用生产 PPPM 或子代理测试公式作真值。

为使验算可复现，取 N=3、c=0.7，R0=(0.1,0.9,2.2)、x=(0.3,−0.4,0.2)、b=(0.2,0.7,−0.3)，P=I−11^T/3；L 的三行为 (−1,1,0)、(0,−1,1)、(1,0,−1)。a_i 的五个分量为 (1,cos R_i,sin R_i,cos 2R_i,sin 2R_i)，G=diag(0,0.8,0.8,0.3,0.3)，G_vir=diag(0,0.5,0.5,−0.2,−0.2)。两者是实空间模基矩阵，本检查没有调用 FFT。

计算顺序为：t=tanh(LR)、s=1−t²、q=Pt、J_q=P diag(s)L；由 N4 求 T，再用 U_i=cq_i a_i^TGρ 和 W_i=cq_i a_i^TG_virρ 作独立被差分函数。对加权能量 b^TU 按三个坐标分别差分得到 V^Tb；对 W(R0±δx) 差分得到 Cx；对解析总能量梯度 2cT^TGρ 差分得到 Kx。N1 的独立被差分函数使用坐标差分得到的 V，而不直接调用 N2 作为真值。所有中央差分 δ=10^−5；刚性平移方向为 (1,1,1)。该 R0 并未被宣布为实际平衡材料参考，保留 Ψ 项检验非平衡代数。

以实际能量/virial函数的中央有限差分分别对照 N6、N5、N7，以及 N1 的导数与 N2；最大绝对残差依次为 5.05e−11、3.61e−12、8.55e−11、8.89e−11，H^T 刚性平移残差为 2.22e−16。它们验证这里的乘积求导和转置代数，不验证真实 PPPM 插值/网格核、charge1/charge2 参数、实际稳定谱或 CUDA。

当前普通 NEP 的九项 sparse CPU 检查通过；测量源码合同检查通过；四项站点分配 toy 检查通过；`git diff --check` 通过（只有 LF→CRLF 提示）。合成 N=5896 CSR 的运行结果不是 qNEP 长程算子的性能证据。原生质心 baseline/candidate CLI 对照脚本已具备，但未在本机运行生产 CUDA。

## 11. 具体修改契约与验收门槛

参考端代码子代理下一轮的范围应为私有参考 evaluator、实际局域能量/charge 微分和 PPPM 参考作用；先完成小体系列差分对照，再落实 N2、N6、N7 的固定解析作用。输入记录需新增 `native_reference_transport` 规范标识、charge mode、kspace method、mesh尺寸/间距、alpha、介电缩放、中和规则、势指纹和数值误差/谱检查。普通 NEP 的已有文件和路径保持。

测量端代码子代理负责把既有 D/三方向 B^T 作用调用接到 qNEP 参考后端；仍保存三方向 J_c、ΔJ、J_A，关联完整 J_A 并保持全部交叉项；默认关闭，禁止用普通 NEP 参考文件绕过 qNEP校验。主代理审核 N2 的转置、九分量布局、所有链式项、冻结参考状态和下列数值门槛后，才能放开 qNEP J_A 开关。

必要门槛：小体系能量/力一致性；参考谱、严格零模和非平衡参考拒绝；按列差分步长收敛；N2 与明确定义的 N1 导数一致；实际采样 PPPM 分配/周期像一致；非对称三方向/异质量对照；完整核密集模空间与经济作用一致；经典极限、零修正、HAC交叉项；启停不改变物理 bead 状态及 BEC调度。

PPPM 是本节完成具体网格推导的路径。Ewald、不同修正势及其他 charge 模式只有逐项核实后才能支持；不能把 PPPM 的证明自动扩展为所有路径已经验收。

## 12. 5896 原子的成本与尚未关闭的事项

若 A/J_q/局域二阶作用具有有界工作量，网格数记 M_g，则单次固定参考作用可按 O(局域导数工作量+M_g log M_g) 计算，工作内存为 O(局域参考数据+M_g+秩向量×3N)，不要求三张 (3N)² double ΔH。完整核仍需约 (3L+2) 个向量 D 作用和 3(R_P+R_Q) 个 B^T 作用；不能将其成本说成每帧仅一次 FFT。

N=5896 时三张全局 double ΔH 约 6.99 GiB，尚未计 K 和缓存。省掉全局矩阵不意味着局域 Hessian 存储便宜：存完整站点块仍需 72Σ_i z_i² 字节。经济实现必须测量该常数及 FFT批处理峰值；理论复杂度不是毫秒耗时证明。

以上解析算子作用是理论经济路线，其复杂度不代表当前源码已经实现。2026-10-04 实施采用固定坐标列有限差分作为正确性基线：私有原生 qNEP 求值生成全局 V、C 和力 Jacobian，同时保存 h 与 h/2 数据；离线准备按 N2 组合参考 H，再作受控分块表示。生产采样只做固定 GPU 算子作用，复用原有完整 P/Q 核；不按样点重新有限差分，也不静默截断长程响应。

该路线新增 qNEP v3 参考及测量接入，限 charge1/charge2+PPPM，并验证机械配置指纹；普通 NEP v1/v2 路径保留。实际准备代码的小体系 CPU 检查已通过，包含运输转置/质量、跨128块密集镜像、内部单位、已知平移正交补及正/负软模。C++参考生成、文件生产读回、CUDA块作用与RPMD启停状态隔离仍待本机以外工具链运行；不能用合成raw准备成功替代这些验收。

N=5896、D=17688 时，该数值基线路线约需4D+13=70765次单构型求值，原始 fine/coarse V/C 与 K 共(23/3)D² doubles，约17.9 GiB磁盘；准备仍需要多张磁盘矩阵和O(D²)内存/O(D³)物理谱审核。最终块压缩的大小和速度依赖实际参考，尚无该材料实测。因此当前不能声明5896原子的经济路线或成本已验收；本机无CUDA编译工具链，真实qNEP的能量/力一致性和长程规范适用边界也保留为待验证项。

原 J_A 仍是有参考站点二次校准的实验候选；即使 N1–N8全部实现和检查通过，也不因此得到任意非谐量子热流、完整当前位置谐波流或量子 DC热导率精确性保证。

## 13. 用户要求进程内准备后的稳定性与软模证书

2026-10-04 用户要求把固定参考准备移入GPUMD，并注意效率。公共入口保持 generate_sparse，qNEP直接生成运行用v3及同名sidecar；Python准备只作为独立参照。以下是主代理本轮设计，实际实现和CUDA验收另行记录。

用已知质量加权平移向量t_i=sqrt(m_i/sum m)，每轴Householder Q_alpha=I−2ww^T，其中w=(t−e0)/||t−e0||。三轴变换后删去0,N,2N的解析平移坐标，得到严格相对坐标子空间A0=Qperp^T D0 Qperp。这里删除的是已知平移子空间，不是按小频率阈值删三个本征值。利用DPOTRF检验A0=L L^T的正定性；失败就拒绝真实负曲率或额外零模。

对重构的压缩矩阵Dc，用相同Qperp得到Ac。令E=Ac−A0，F=L^-1 E L^-T；通过两次三角求解和Frobenius范数求epsilon=||F||F。因||F||2<=||F||F，epsilon<1蕴含−epsilon I<=F<=epsilon I，故

(1−epsilon)A0 <= Ac <= (1+epsilon)A0。

由Courant-Fischer最小最大原理，各按序物理正特征值满足(1−epsilon)lambda_k(A0)<=lambda_k(Ac)<=(1+epsilon)lambda_k(A0)。角频率的相对改变因此被max(1−sqrt(1−epsilon),sqrt(1+epsilon)−1)控制。epsilon<=0.01比1%的相对频率门槛更保守；不依赖软模的绝对大小，也不需要全局特征向量。数值计算必须验证Cholesky重构和三角求解残差；这仍是双精度数值证书，不是区间证明。

D的相对证书不过时可仅D退回lossless，保留已经过误差检查的三张非对称Bt压缩。若显存不足以执行证书，应明确报资源限制，不能改用频率绝对阈值或无检查压缩。kernel使用已有完整P/Q系数与安全factor-row谱上界，不改高温Taylor形式。

GPU审核一次需要A0/L和E/F两张约3(N−1)方阵，N=5896合计约4.66GiB，另加库查询的DPOTRF workspace和小块缓冲；其运算仍为O(D³)的一次准备成本。生产采样的固定块作用和HAC保持不变，不包含Cholesky或逐帧参考更新。句柄和小块workspace复用，原始导数及最终tiles流式处理。最低耗时不能由复杂度推定，仍须实际CUDA基准。
