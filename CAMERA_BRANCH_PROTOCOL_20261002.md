# DynaMesh4D Camera Branch 正式消融协议（2026-10-02）

## 0. 审计范围与结论等级

本协议来自对当前服务器源码、checkpoint、Phase-2 现有 614/614 validation 输出，以及 `/root/data1/sam-body4d-master/dataset/3DPW/` 元数据的只读审计。本轮没有训练、模型推理或大型 benchmark，也没有访问 EMDB。

结论分为三类：

- **VERIFIED**：由当前源码、文件哈希或现有审计产物直接确认。
- **DERIVED**：由已确认的字段和变换关系推导，执行前仍需数值等价性 gate。
- **BLOCKED**：当前实现或证据不足，禁止作为正式论文结果运行或汇报。

## 1. Camera Branch 真实结构

### 1.1 Foundation SAM camera path

`scripts/train_dynamesh_v2_camera_phase1.py::decode_condition` 的真实路径是：

1. 输入缓存的 SAM pose token `raw_pose_token`，shape `(T, 1024)`；另有 `mhr_init_pose`。
2. 冻结的 `model.head_pose(raw_pose_token, mhr_init_pose)` 产生 `pred_keypoints_3d`。
3. 取固定 MHR/OpenPose 对应点作为 `points3d`，并 `detach`；这些点是局部人体几何，单位米，已按 SAM 缓存约定翻转 y、z 到 camera axes。
4. 冻结的 `model.head_camera(camera_token, init_camera)` 产生 `pred_cam`。
5. `perspective_projection(points3d, pred_cam, bbox center/size, image size, K)` 产生 `pred_cam_t` 与全图 2D 投影。

因此 Raw camera 输出是：

- `pred_cam`：camera head 的内部三参数表示；不是 3×3 rotation，也不是 4×4 extrinsic。
- `pred_cam_t`：把局部 MHR points 放入 camera coordinates 的 3D translation，shape `(T,3)`，单位米。
- `pred_xy`：投影后的 2D keypoints。

### 1.2 Frame-wise camera calibration（Spatial module）

源码：`scripts/ablate_dynamesh_v2_camera_phase111.py::FramewiseLatentAdapter`。

结构：

```text
token(1024)
  -> LayerNorm(1024)
  -> Linear(1024,128)
  -> GELU
  -> Linear(128,1024), zero-init
  -> residual add to original token
  -> frozen SAM head_camera
  -> pred_cam, pred_cam_t, pred_xy
```

输入与输出：

- 输入：每帧独立的 SAM pose token `(B,T,1024)`；模块本身没有时间混合。
- 输出：校准后的 token，再经冻结 `head_camera` 得到 `pred_cam`、`pred_cam_t` 和 `pred_xy`。
- 不重新运行或修改 `head_pose`；投影使用与 Raw 完全相同且已 detach 的 `points3d`。

作用范围：

- camera translation：**是**，通过重新预测 `pred_cam` 最终改变 `pred_cam_t`。
- camera rotation：**否**，没有 rotation 输出。
- root translation：**间接**；`pred_cam_t` 可用于放置局部 root，但模块不输出独立 root 参数。
- global orientation：**否**。
- world trajectory：**否**；没有 world-space head 或预测 extrinsic。

### 1.3 Temporal camera / trajectory refinement

核心源码：

- `models/sam_3d_body/sam_3d_body/models/modules/camera_temporal.py`
- `scripts/train_dynamesh_v2_camera_phase112.py::temporal_residual_forward`
- `scripts/train_dynamesh_v2_camera_phase112.py::temporal_forward`

真实 Phase-2 结构：

```text
raw SAM token
  -> frozen Phase-2 FramewiseLatentAdapter
  -> frame-wise calibrated token
  -> input LayerNorm + projection
  + reliability-condition embedding
  -> bidirectional Mamba stack
  -> stable output LayerNorm + output projection
  -> temporal residual add
  -> frozen SAM head_camera
  -> pred_cam, pred_cam_t, pred_xy
```

Reliability/inference cues：visibility、bbox center delta、bbox scale delta、temporal latent disagreement；缺失 cue 由 mask 显式表示。GT 2D、GT reprojection residual、GT camera error 不允许进入 inference features。

Phase-2 的 temporal residual 输入是 **frame-wise calibrated token**，不是 Raw token。其损失直接监督 `pred_cam_t` 的 absolute、velocity、acceleration，并带 reprojection auxiliary loss。

作用范围：

- camera translation：**是**。
- camera rotation：**否**。
- root translation：**间接**。
- global orientation：**否**。
- world trajectory：**否**；只能在 evaluator 中借助数据集 GT extrinsic 从 camera-space 人体轨迹派生。

### 1.4 Body Branch 参与关系

Camera Phase-2 训练/forward 中：

- `load_frozen_model()` 加载的是冻结 foundation SAM-Body4D；Camera optimizer 只拥有 Camera adapter 参数。
- 正式 Body temporal adapter checkpoint 的 SHA 被协议锁定，但 Camera forward 不调用 Body Branch temporal adapter。
- `head_pose` 产生的 raw `points3d` 参与 camera projection，因此 Camera Branch **依赖 foundation body prediction 的局部人体几何**。
- Body Branch refined output 不参与 Camera adapter 的 forward；`points3d` 被 detach。
- `dynamesh_v2_integration.py::compose_parameters` 只在结果组合阶段分别替换 body keys 与 camera keys，并禁止跨分支泄漏。
- 现有 full-integration validation 报告：body-only 与 full 的 body 最大差为 0；camera-only 与 full 的 camera 最大差为 0。

### 1.5 真实调用链

```text
cached SAM token + mhr_init_pose
    |-- frozen head_pose ----------------------> raw local points3d (fixed for all Camera ablations)
    |
    `-- Raw path: frozen head_camera ----------> raw pred_cam -> raw pred_cam_t
            |
            `-- Spatial: FramewiseLatentAdapter
                    -> frozen head_camera ------> spatial pred_cam / pred_cam_t
                            |
                            `-- Temporal: CameraFeatureTemporalAdapter
                                    -> frozen head_camera
                                    -> full pred_cam / pred_cam_t

Body Branch refined pose is composed separately and is never changed by Camera ablation.
World output is evaluator-derived only:
camera-space body/root + inverse 3DPW GT cam_pose -> world-space body/root.
```

## 2. 正式 checkpoint 冻结

| 角色 | 路径 | SHA256 | 阶段 | 代码入口 |
|---|---|---|---|---|
| Body（固定） | `outputs/dynamesh_v2_phase2_training/best.pt` | `f1509cc1a8bfae84501aeac2004529b029fb8a4b60afa8a4349faf31efe60329` | Body Phase-2 | Body inference/evaluator |
| Frame-wise Camera（正式） | `outputs/dynamesh_v2_camera_phase2/framewise/best.pt` | `fd318ddbe3e3a1e9a13134e649ce92aa60c9b28385bf809ffb3ffc44ef516cb2` | Camera Phase-2 Stage A | `train_dynamesh_v2_camera_phase2.py::train_framewise`; `evaluate...::output('framewise')` |
| Temporal Camera（正式） | `outputs/dynamesh_v2_camera_phase2/temporal_true/best.pt` | `e7562abc45825de76ddb7a24b60bdf3675061c9ba63ac150071558b98f6ac865` | Camera Phase-2 Stage B | `train_dynamesh_v2_camera_phase2.py::train_temporal`; `train_dynamesh_v2_camera_phase112.py::temporal_forward` |
| Shuffled temporal control | `outputs/dynamesh_v2_camera_phase2/temporal_shuffled/best.pt` | `e428f1d25e6e058a8c73e631e3f8cbdef57921fa0365e3d46396d967b8d50193` | Phase-2 negative/control | Phase-2 evaluator only |
| 历史 direct-temporal reference | `outputs/dynamesh_v2_camera_phase110_multiwindow_stable_norm/best.pt` | `1df36e60c503c402f1aa507a31f0a9950d9533d68dd1ec76a4ff01dfbe939183` | Phase-1.10 | `train_dynamesh_v2_camera_phase110_stable_norm.py` |

重要约束：

- Camera Phase-2 **没有一个合并 checkpoint**；正式 Full Camera 是 frame-wise 与 temporal 两个 checkpoint 串联加载。
- 未找到 Body+Camera 联合训练或联合保存的 checkpoint。集成由 `models/.../dynamesh_v2_integration.py` 在 inference/evaluation 时组合，三个冻结权重文件独立存在。
- 历史 Phase-1.10 checkpoint 不能冒充 Phase-2 的“Temporal only”，因为训练输入、目标与阶段均不同。

## 3. 四组严格 Camera ablation 可实现性

所有组必须固定：

- 相同 Body checkpoint、Body restoration strength、Body window/stride；
- 相同 raw SAM token、`mhr_init_pose`、raw decoded/detached `points3d`；
- 相同 bbox、K、image size、frame/person keys、条件与窗口拼接；
- 相同 local body output；Camera ablation 不允许改变 body pose/global orientation/shape。

### A. Raw — 可实现

- 输入：raw SAM token + frozen `head_camera`；projection 使用固定 raw `points3d`。
- checkpoint：无 Camera checkpoint。
- disable：FramewiseLatentAdapter、CameraFeatureTemporalAdapter 均 OFF。
- 输出：raw `pred_cam`、`pred_cam_t`、`pred_xy`。
- Body output：不变。

### B. Spatial only — 可实现

- 输入：raw SAM token。
- checkpoint：Phase-2 `framewise/best.pt`。
- enable：FramewiseLatentAdapter ON。
- disable：temporal adapter OFF。
- 输出：frame-wise calibrated `pred_cam`、`pred_cam_t`、`pred_xy`。
- Body output：不变；projection points 与 Raw 相同。

### C. Temporal only — 当前不能严格实现（BLOCKED）

要求是 frame-wise OFF、temporal ON，但正式 Phase-2 temporal checkpoint 的训练和 forward contract 都是：

```text
raw token -> frozen Phase-2 framewise -> temporal residual
```

把 `temporal_true/best.pt` 直接作用于 raw token 会违反训练分布与现有 `temporal_forward` contract；这不是合法 disable。改用 Phase-1.10 direct-temporal checkpoint 又会同时改变训练阶段、loss、数据选择和 numerical stabilization，不能与 Phase-2 A/B/D 构成严格单因素消融。

因此本轮不得伪造 C。若论文必须保留 2×2 因子设计，只能未来预注册并训练一个“raw-token temporal-only”对照；当前禁止重新训练。

### D. Full Camera Branch — 可实现，但 safety gate 尚未通过

- 输入：raw SAM token。
- checkpoints：Phase-2 `framewise/best.pt` + `temporal_true/best.pt`。
- enable：frame-wise ON，true temporal ON。
- 输出：最终 `pred_cam`、`pred_cam_t`、`pred_xy`。
- Body output：不变。
- 当前 validation 定位：`CAMERA_FRAMEWISE_CONFIRMED_TEMPORAL_AUXILIARY`；存在 `over_smoothing_risk = true`，因此不能在 world-trajectory 表中无条件称为最终 Full Camera。

### 推荐论文表结构

当前可合法报告三组主消融：Raw、Spatial only、Full；另加 Shuffled temporal 作为时间顺序控制。不要报告不存在的 Temporal only。若审稿格式强制四项，第四项应写作 `Temporal only — unavailable under frozen Phase-2 architecture`，而不是填入 Phase-1.10 数值。

## 4. 3DPW GT 字段审计

数据根：`/root/data1/sam-body4d-master/dataset/3DPW/`。

SequenceFiles 实测：train 24、validation 12、test 24 个 PKL。样本结构在三个 split 一致。

| 字段 | 存在 | shape | 坐标系/表示 | 单位 | 逐帧 | person-specific | evaluator 可用性 |
|---|---|---|---|---|---|---|---|
| `cam_poses` | 是 | `(T,4,4)` | 左乘齐次列向量的 world→camera extrinsic | translation 为米 | 是 | 否，sequence/frame 共享 | 可；先检查 rigid/finite/valid |
| camera rotation | 是，`cam_poses[:,:3,:3]` | `(T,3,3)` | `R_cw`，world→camera | 无量纲 | 是 | 否 | 可作为 GT；当前模型无对应预测 |
| extrinsic translation | 是，`cam_poses[:,:3,3]` | `(T,3)` | `t_cw`，不是 world camera center | 米 | 是 | 否 | 可；camera center 为 `-R_cw^T t_cw` |
| camera→world | 未单独存储，可逆得到 | `(T,4,4)` | `inv(cam_poses)` | 米 | 是 | 否 | 可，必须统一用矩阵逆 |
| `poses` | 是，list P | 每人 `(T,72)` | SMPL axis-angle；前 3 维为 global orient，余下 body pose | 弧度 | 是 | 是 | 可解码 world SMPL |
| `trans` | 是，list P | 每人 `(T,3)` | SMPL world translation | 米 | 是 | 是 | 可；不是直接等同于任意 pelvis 定义 |
| `jointPositions` | 是，list P | 每人 `(T,72)`，可 reshape `(T,24,3)` | 3DPW world joints | 米 | 是 | 是 | 可，但正式 joint mapping 必须锁定 |
| `poses2d` | 是，list P | 每人 `(T,3,18)` | OpenPose-18，全图 x/y/confidence | 像素 | 是 | 是 | 可作为 reprojection auxiliary |
| `cam_intrinsics` | 是 | `(3,3)` | pinhole K | 像素 | sequence 常量 | 否 | 可 |
| `campose_valid` | 是，list P | 每人 `(T,)` | validity mask | bool/uint8 | 是 | 是 | 必须用于过滤 |
| `img_frame_ids` | 是 | `(T,)` | 60 Hz mocap sync index；正常 RGB 相邻差为 2 | index | 是 | 否 | 可；必须与 RGB local index 区分 |
| explicit timestamp | 否 | — | — | — | — | — | 由 30 fps 与 `img_frame_ids` 差推导，非显式时间戳 |
| `poses_60Hz` / `trans_60Hz` | 是 | 每人 `(2T,72)` / `(2T,3)` | world SMPL at 60 Hz | rad / m | 是 | 是 | 可用于额外 timebase audit，不与 30 fps 输出直接混用 |

`3DPW_oc.json`：

- `images` 20,529 条，包含 sequence、RGB `frame_idx`、image size、focal/principal point。
- `annotations` 20,243 条，包含 person id、`joint_cam (24,3)`、`fitted_joint_cam (24,3)`、`joint_img (24,2)`、OpenPose `(18,3)`、bbox，以及 `smpl_param`（shape、pose72、trans3、gender）。
- OC JSON 的人体字段是 camera-space spatial annotation，适合 OC 实例空间指标；连续 world trajectory 应以 sequenceFiles 的完整时间轴为准。

## 5. 指标支持矩阵

状态针对“3DPW GT + 当前 Camera 输出”共同判断。

| 指标 | 状态 | 严格解释 |
|---|---|---|
| 1. Root Translation Error | **SUPPORTED_WITH_TRANSFORM** | Phase-2 的 `pred_cam_t` 可与拟合的 `t_all` 做内部 camera translation error；若称物理 root error，必须先把同一定义的 local root 加到 `pred_cam_t`，并与 GT 同根比较。当前 root mapping gate 未通过。 |
| 2. Global pelvis position error | **SUPPORTED_WITH_TRANSFORM** | 可从预测 camera-space pelvis 经 `inv(cam_poses)` 得到 world pelvis；必须锁定 SMPL/H36M pelvis 定义和 conversion。 |
| 3. W-MPJPE | **SUPPORTED_WITH_TRANSFORM** | 3DPW 提供 world SMPL 与 extrinsic；预测需先构造 camera joints，再严格变换到 world。必须在协议中明确定义是否绝对、是否首帧对齐，禁止借用同名不同协议。 |
| 4. WA-MPJPE | **SUPPORTED_WITH_TRANSFORM** | 可以实现，但 alignment 必须是一个预注册的 sequence-level SE(3)/yaw/first-frame 对齐；禁止 per-frame Procrustes 冒充 WA-MPJPE。 |
| 5. ATE | **SUPPORTED_WITH_TRANSFORM**（person root only） | 可对 world pelvis trajectory 做 sequence-level rigid-aligned或绝对 ATE；当前模型没有 predicted camera trajectory，因此 camera ATE **NOT_SUPPORTED**。 |
| 6. RTE | **SUPPORTED_WITH_TRANSFORM**（person root only） | 可对固定时间间隔的相对 world pelvis displacement 做 RTE；camera RTE **NOT_SUPPORTED**。 |
| 7. Camera rotation geodesic error | **NOT_SUPPORTED** | 3DPW 有 GT `R_cw`，但 Camera Branch 不输出 camera rotation/extrinsic rotation。 |
| 8. Root/world velocity error | **SUPPORTED_WITH_TRANSFORM** | world pelvis 一阶差分，按真实 30 fps/有效连续帧缩放为 m/s。 |
| 9. Root/world acceleration error | **SUPPORTED_WITH_TRANSFORM** | world pelvis 二阶差分，按 `30²` 缩放为 m/s²，仅使用连续有效 triplet。 |
| 10. Trajectory jitter | **SUPPORTED_WITH_TRANSFORM** | 必须同时报告 GT-relative acceleration error与 prediction-only jerk/高频能量；单独“越平滑越好”无效。 |

在当前 root mapping blocker 清除前，只有 Phase-2 已有的 camera-space `pred_cam_t` 拟合目标误差、reprojection、velocity/acceleration diagnostics 可正式复用；world 指标处于“数据支持、实现未获准”的状态。

## 6. 坐标系与唯一合法 transform chain

### 6.1 已确认约定

- SAM/MHR `points3d`：局部人体坐标，单位米；在缓存前已按当前 pipeline 翻转 y、z，使其处于 SAM camera-axis convention。
- Camera output `pred_cam_t`：同一 camera-axis convention 下的 translation，单位米。
- Camera Branch 不输出旋转；`pred_cam` 不是 rotation matrix。
- 3DPW sequenceFiles `poses/trans/jointPositions`：world coordinates。
- `cam_poses = T_cw`：world→camera，列向量左乘：`X_c = R_cw X_w + t_cw`。
- 逆变换：`X_w = R_cw^T (X_c - t_cw)`。
- RGB 推理帧索引是 local index `0..T-1`；`img_frame_ids` 是 60 Hz sync id，30 fps 相邻通常差 2。
- SMPL pose 是 axis-angle；不要把 `poses[:3]` 与 camera `R_cw` 相加或直接相乘而不先转 rotation matrix。

### 6.2 Prediction → GT 唯一合法链

对于 camera-space translation/reprojection：

```text
raw/fixed local MHR points X_local_m
  + chosen ablation pred_cam_t_m
  = X_pred_camera_m
  -> K projection for 2D auxiliary metrics
```

对于未来 world body/root metrics：

```text
frozen Body output
  -> frozen MHR-to-SMPL conversion / common joint regressor
  -> local predicted SMPL joints in SAM camera axes
  + chosen Camera ablation pred_cam_t
  = predicted camera-space joints X_c
  -> X_w = R_cw^T (X_c - t_cw), using same-frame 3DPW GT cam_pose
  -> compare with GT world SMPL joints decoded from poses/trans/betas
```

严禁：

- 把 `t_cw` 当作 camera center；
- 对行向量右乘而不转置；
- 同时应用 `cam_poses` 和其 inverse；
- 把 OC JSON 的 camera-space `smpl_param.trans` 当作 sequenceFiles world `trans`；
- 混用 MHR hip midpoint、SMPL native joint 0 与 H36M pelvis；
- 把 per-frame Procrustes 后结果称为 W/WA trajectory metric。

### 6.3 当前坐标/根定义 blocker

现有 `outputs/dynamesh_v2_camera_root_gate_50f/summary.json` 的 verdict 是 `ROOT_MAPPING_NOT_RELIABLE`：MHR 左右髋中点与 Meta-converted SMPL/H36M pelvis 的局部映射有约 88 mm 均值差，且随 pose 变化。该差异不是一个可安全扣除的固定 magic offset。

因此：

- 变换方向、单位、帧索引本身已明确；
- **root/pelvis representation mapping 仍有歧义**；
- 在建立同一最终表示上的 numerical equivalence gate 之前，不得运行或汇报正式 world trajectory metric。

## 7. 过平滑 safety gate

### 7.1 已存在且冻结的 Phase-2 gates

当前 validation evaluator 已定义：

- absolute translation：Temporal ≤ `1.02 × Framewise`；
- velocity error：Temporal < Framewise；
- acceleration error：Temporal < Framewise；
- temporal order：True temporal velocity error < Shuffled temporal；
- stable subset：Temporal translation ≤ `1.05 × Framewise`；
- high-motion subset：按 GT velocity magnitude 的 top 25%；
- over-smoothing flag：`temporal velocity-amplitude-ratio < 0.8 × framewise ratio`。

614/614 validation 结果：

- temporal 在 absolute/velocity/acceleration 多数 matched windows 胜过 framewise；
- high-motion velocity amplitude ratio：Framewise `6.736`，True temporal `3.882`；
- 因 `3.882 < 0.8 × 6.736`，`over_smoothing_risk = true`；
- 正式 verdict：`CAMERA_FRAMEWISE_CONFIRMED_TEMPORAL_AUXILIARY`，不是 full temporal generalization confirmed。

### 7.2 新增、现在预注册且不得按 test 调整的 gates

正式 smoke/full 之前新增以下 evaluator-only checks：

1. **Velocity magnitude preservation**：分别报告 `||v_pred||/||v_gt||` 及 Temporal/Framewise 比值；high-motion 中 Temporal/Framewise 不得低于 `0.80`。
2. **Acceleration magnitude preservation**：同上；high-motion 中 Temporal/Framewise 不得低于 `0.80`。
3. **High-motion subset**：只按 GT camera-space root speed 的 top quartile 定义；不得看模型性能后改样本。
4. **Static/low-motion subset**：只按 GT speed bottom quartile 定义；Temporal 的 prediction-only acceleration/jerk 不得比 Framewise 高 `>5%`。
5. **GT-relative gate**：velocity error 与 acceleration error必须同时不劣于 Framewise；仅降低 prediction amplitude 不算通过。
6. **Spatial safety**：absolute root/translation error不得比 Framewise高 `>2%`；stable/low-motion subset 不得高 `>5%`。

这些阈值在任何新 3DPW smoke/test 结果产生前写入本协议。当前已有结果明确未通过 velocity non-collapse gate，因此 Full Camera 暂只作为 auxiliary variant。

## 8. 3DPW 开发 smoke 序列选择

### 8.1 选择规则

只使用 train + validation（共 36 条），不使用 test。对每个 person track：

1. 读取 world SMPL translation `p_w(t)=trans[t]`；
2. 用 `cam_poses` 转为 camera-space trajectory：`p_c(t)=R_cw(t)p_w(t)+t_cw(t)`；
3. 仅保留 `campose_valid` 且相邻 `img_frame_ids` 差为 2 的边；
4. 计算 track mean speed：`mean(||p_c(t+1)-p_c(t)|| × 30)` m/s；
5. sequence score = person track mean speed 的中位数；
6. 在 36 个 sequence scores 中选择最接近 q10、q50、q90 的序列作为 low/medium/high。

该规则只依赖 GT motion magnitude，不读取 Raw、Dyna 或任何 Camera checkpoint 性能。

### 8.2 本轮只读排名得到的候选

| 档位 | quantile target | sequence | split | score m/s |
|---|---:|---|---|---:|
| Low | q10 = 0.189253 | `courtyard_arguing_00` | train | 0.190335 |
| Medium | q50 = 0.360579 | `courtyard_relaxOnBench_00` | train | 0.362279 |
| High | q90 = 0.919164 | `courtyard_jumpBench_01` | validation | 0.906687 |

`outdoors_crosscountry_00` 的该定义分数为 0，因此未选作 low；q10 能避免退化的零运动样本。

本轮不运行这三条序列。

## 9. EMDB 迁移接口

未访问任何 EMDB 数据。未来 evaluator 应分成数据适配器与共享 metric core：

```text
DatasetAdapter
  -> frame_id / timestamp / validity
  -> GT world joints + root
  -> optional T_cw / T_wc
  -> predicted camera-space joints/root
  -> canonical world trajectory record
  -> shared W-MPJPE / WA-MPJPE / ATE / RTE / velocity / acceleration / jitter core
```

EMDB-1：常规 body/generalization，复用 body-centric MPJPE/PA/MPVPE，Camera 不作为主要结论。

EMDB-2：world trajectory / Camera Branch。只有当数据提供可验证的 world body/root、timebase 及所需 camera transform 时，才启用 W-MPJPE、WA-MPJPE、ATE、RTE 和 trajectory metrics。

当前 evaluator 的差分和聚合逻辑可复用，但不能直接复用 3DPW 的字段名、30 fps 假设、pelvis mapping 或 transform convention。EMDB adapter 必须单独通过：单位、坐标方向、rotation convention、root joint、frame/timestamp continuity 五个 gates。

## 10. 当前 blockers 与执行许可

1. **Temporal-only 不可实现**：正式 Phase-2 temporal checkpoint 依赖 frame-wise base；严格 2×2 四组消融缺一组。
2. **Root mapping 未通过**：现有 verdict `ROOT_MAPPING_NOT_RELIABLE`，阻止正式 world root/W-MPJPE/WA-MPJPE/ATE/RTE。
3. **过平滑 gate 未通过**：Full temporal 当前只能定位为 auxiliary。
4. **Camera rotation 不可评**：模型没有 rotation prediction。
5. **无联合 checkpoint**：Body、frame-wise Camera、temporal Camera 必须作为三个冻结 checkpoint 独立记录。
6. **指标命名待最终冻结**：特别是 W-MPJPE、WA-MPJPE、ATE、RTE 的 alignment scope 必须在 evaluator 实现前逐项写死。

执行结论：

- 可以进入三条非 test 序列的 **camera-space translation/reprojection diagnostic smoke**，前提是只使用 Raw、Spatial、Full、Shuffled temporal，且不宣称 world metric。
- 不能进入正式 3DPW world-trajectory smoke；必须先解决 common-root numerical equivalence gate。
- 不得自动启动任何实验、训练或大型 benchmark。
