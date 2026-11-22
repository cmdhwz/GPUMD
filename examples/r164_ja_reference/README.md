# 在 GPUMD 中使用 JA 参考文件

## 用已有 R27 参数试跑

将 [ja_reference_r27.txt](ja_reference_r27.txt) 放到 `run.in` 同级目录。它对应原盐冰体系的 R27 固定双模，5896 个原子、29.3104 Å 立方胞、300 K，零基原子编号为 O715、H3382、H5674。模板使用原 mode2 qNEP 势文件的 SHA-256。

完整运行卡示例。沿用已平衡的全珠 restart，用下列配置跑 24 ps，并将 JA、Hardy 和 Centroid 的关联计算到约 12 ps。

```text
potential /root/data/PIMD/qnep_iter08_32w_new.txt
pimd_propagator exact
kspace pppm 0.5

time_step 0.2
ensemble rpmd 32 300
pimd_fix_com off
read_pimd_restart restart_beads_pimd.xyz

r164_rpmd r164_24ps 0 observer buffered centroid ja_reference ja_reference_r27.txt
compute_hac_ja 5 12000 5
run 120000
```

`5 12000 5` 每 5 步采一个热流点，计算 12000 个滞后点，再每 5 个滞后点平均为一行。0.2 fs 的时间步对应 1 fs 的热流采样间隔，最后一组覆盖约 12 ps。输出目录应是新的目录。省略 `ja_reference 文件名` 时使用原内置 R27。省略 `centroid` 时输出 JA 和 Hardy。只需要 HAC 文件时，可在 `compute_hac_ja` 行末加 `corr`。

参考文件在计算开始时加载。检查原子数、温度、晶胞、所选 O/H/H 的元素和质量，以及势文件的 SHA-256。输出目录的 `metadata.txt` 记录参考文件标识和 SHA-256。参考与新结构的局部物理环境是否相符，仍需用标定结果和短测确认。

## 更换盐冰构型

| 阶段 | 需要准备的输入 | 产物 |
| --- | --- | --- |
| 确定体系 | 新的 `model.xyz`（保留原子顺序、元素、质量和晶胞）、生产用的 qNEP 势文件和 `run.in`（含温度、PPPM 设置）。若已选好目标水分子，提供 O/H/H 的零基原子编号 | 新结构与目标双键的定义 |
| R26 静态参考 | 上一步的构型和势。固定晶胞弛豫，在两条 O–H 伸缩方向做两档力差分，检查投影曲率和模外耦合 | `relaxed.npz`、`stretch.npz` |
| R27 静态标定 | R26 输出和同一 qNEP/PPPM 设置下的完整普通 Hardy 流后端。用两档步长检查运输系数 | `calibration.npz`、`calibration.json` 和新 JA 参考文件 |
| RPMD 短测 | 新 JA 文件、使用同一原子顺序的 `model.xyz` 和已平衡的全珠 `restart_beads_pimd.xyz` | 用实际载入的同一帧坐标和动量离线复算 `d2`，与 `series.txt` 对照 |
| 正式配对计算 | 通过短测的 JA 文件及各条轨迹的平衡 restart | `hac_ja.out`、`hac_hardy.out`，并可选 `hac_centroid.out` |

已有的 `relax.py`、`stretch.py`、`calibrate.py` 是标定起点。原脚本部分路径、行序和 64³ 网格绑定原 R27 体系，移植时要按新结构与生产后端调整；标定阶段使用静态构型。

短测时保留上面运行卡的物理设置，将最后三行替换为以下两行。这样每一步保存时序和全珠快照。

```text
r164_rpmd ja_ref_check 1 observer buffered 1 ja_reference ja_reference_new.txt
run 5
```

核对 `ja_ref_check/metadata.txt` 的参考标识和原子编号，检查 `series.txt` 的 `d2_valid`、`J2_valid` 和逐分量的 `J2 = b + d2`。离线复算使用同一周期提升坐标下的参考位置和实际全珠动量。开始长轨迹前完成这一步。

## 参考文件字段

按模板顺序保留关键字。数值用空白分隔，数组可以换行，保留标定的完整精度。

| 字段 | 来源与约定 |
| --- | --- |
| `r164_ja_reference 1` | 格式版本 |
| `id` | 一个不含空白的参考标识，每套标定使用独立标识 |
| `potential_sha256` | 生产 `potential` 文件的 SHA-256，64 位小写十六进制。Linux 可用 `sha256sum 势文件` |
| `atoms`、`temperature_K`、`cell_A` | 总原子数、标定温度（K）和正交周期胞的 x/y/z 边长（Å） |
| `rows` | O/H/H 的三个零基原子编号，按全珠 restart 的每珠顺序 |
| `masses_eV_fs2_A2` | 从 `calibration.npz` 的 `mass[rows]` 取质量，单位 eV·fs²/Å² |
| `reference_A` | 三个原子的校准参考坐标，按 O/H/H、x/y/z，共 9 个值。参考坐标与轨迹使用同一周期提升约定 |
| `basis` | `B[rows, :, :]`，按 O/H/H、x/y/z、模态 0/1，共 18 个值。两列以质量度量正交归一 |
| `D` | `delta_coeff`，按 x/y/z、Π 模态、Q 模态，共 12 个值 |

同一标定环境下的不同平衡初态可以共用参考。更换 NaCl 位点或质子排布时，重新选定双键并核对标定。同一次跨轨迹统计使用同一套参考，或在分析中按参考分组。
