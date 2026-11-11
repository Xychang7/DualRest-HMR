import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sam_3d_body.data.utils.io import load_image
from sam_3d_body.data.utils.prepare_batch import prepare_batch
from sam_3d_body.utils import recursive_to

# ==========================================
# 1. 植入 Bi-Mamba Motion Smoother 模型定义
# ==========================================
try:
    from mamba_ssm import Mamba
except ImportError:
    print("⚠️ [Warning] mamba_ssm not found. Please install via 'pip install mamba-ssm'")
    Mamba = None

# ==========================================
# 1. 植入 Bi-Mamba Motion Smoother 模型定义
# ==========================================
try:
    from mamba_ssm import Mamba
except ImportError:
    print("⚠️ [Warning] mamba_ssm not found. Please install via 'pip install mamba-ssm'")
    Mamba = None

#  【消融实验全局开关】
# 设为 True: 屏蔽反向扫描，强制降级为单向 Uni-Mamba
# 设为 False: 正常的双向 Bi-Mamba (Ours)
UNI_MAMBA_MODE = False  

class BiMambaSmoother(nn.Module):
    def __init__(self, input_dim=266, d_model=256, n_layers=4):
        super().__init__()
        if Mamba is None:
            raise ImportError("mamba_ssm module is missing.")
            
        self.embedding = nn.Linear(input_dim, d_model)
        
        # 正向 Mamba
        self.layers_fwd = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        
        # 反向 Mamba
        self.layers_bwd = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, input_dim)

    def forward(self, x):
        # x: (Batch, Seq_Len, Dim)
        residual = x
        x_embed = self.embedding(x)
        
        # 1. Forward Pass (始终执行，提取因果历史信息)
        x_f = x_embed
        for layer in self.layers_fwd:
            x_f = layer(x_f)
            
        # 2. Backward Pass (根据开关决定是否屏蔽)
        if UNI_MAMBA_MODE:
            # 【核心消融魔法】：强行用前向特征替代后向特征。
            # 彻底切断未来信息，且 norm(x_f + x_f) 会维持原有的特征缩放比例。
            x_b = x_f 
        else:
            x_b = torch.flip(x_embed, dims=[1])
            for layer in self.layers_bwd:
                x_b = layer(x_b)
            x_b = torch.flip(x_b, dims=[1])
        
        # 3. Fusion
        x_merged = self.norm(x_f + x_b)
        correction = self.output_head(x_merged)
        
        return residual + correction    


SMOOTHER_MODEL = None

def load_smoother_model(device):
    
    # return None
    
    global SMOOTHER_MODEL
    if SMOOTHER_MODEL is None:
        try:
            # 初始化 Mamba 模型
            model = BiMambaSmoother(input_dim=266, d_model=256, n_layers=4).to(device)
            
            # ✅ 请确保这里的路径与 train_mamba.py 生成的权重文件名一致
            ckpt_path = "/root/sam-body4d-master/mamba_smoother_epoch_15.pth"
            
            if os.path.exists(ckpt_path):
                print(f"🐍 [Mamba] Loading weights from {ckpt_path}...")
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                model.eval()
                SMOOTHER_MODEL = model
            else:
                print(f"⚠️ [Mamba] Checkpoint not found at {ckpt_path}. Skipping.")
                SMOOTHER_MODEL = "DISABLED"
        except Exception as e:
            print(f"❌ [Mamba] Error loading model: {e}")
            SMOOTHER_MODEL = "DISABLED"
    
    return SMOOTHER_MODEL if SMOOTHER_MODEL != "DISABLED" else None

# ==========================================
# 2. 工具函数
# ==========================================
def dummy_transform(x):
    return x

def collate_batches(batch_list):
    collated_batch = {}
    if len(batch_list) == 0: return collated_batch
    keys = batch_list[0].keys()
    for key in keys:
        values = [b[key] for b in batch_list]
        if key == 'img' and isinstance(values[0], torch.Tensor):
            collated_batch[key] = torch.stack(values, dim=0)
        elif isinstance(values[0], torch.Tensor):
            try: collated_batch[key] = torch.cat(values, dim=0)
            except: collated_batch[key] = values 
        elif isinstance(values[0], np.ndarray):
            collated_batch[key] = np.concatenate(values, axis=0)
        elif isinstance(values[0], list):
            flattened = []
            for v in values: flattened.extend(v)
            collated_batch[key] = flattened
        else: collated_batch[key] = values
    return collated_batch

def process_image_with_mask(
    model, img_path_list, mask_path_list, idx_path, idx_dict, 
    mhr_shape_scale_dict, occ_dict
):
    # 1. 图像预处理
    img_list = []
    for img_path in img_path_list:
        img = load_image(img_path, backend="cv2", image_format="bgr")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_list.append(img)

    mini_batches = []
    real_transform = getattr(model, 'transform', None) or dummy_transform

    for i, mask_path in enumerate(mask_path_list):
        mask = cv2.imread(mask_path, 0)
        if mask.shape[:2] != img_list[i].shape[:2]:
            mask = cv2.resize(mask, (img_list[i].shape[1], img_list[i].shape[0]), interpolation=cv2.INTER_NEAREST)
        
        y, x = np.where(mask > 0)
        bbox = np.array([[np.min(x)-10, np.min(y)-10, np.max(x)+10, np.max(y)+10]]) if len(y)>0 else np.array([[0,0,10,10]])
        
        # ✅ 修复点：显式传递 img_com_dict={}
        mini_batches.append(prepare_batch(img_list[i], real_transform, bbox, img_com_dict={}))
    
    batch = recursive_to(collate_batches(mini_batches), "cuda")
    
    # ==========================================
    # 🩹 修复：暴力维度清洗 (Robust Dimension Cleaning)
    # ==========================================
    if 'img' in batch:
        img_tensor = batch['img']
        # 1. 只要维度 > 4，就循环挤压
        while img_tensor.dim() > 4:
            if img_tensor.shape[1] == 1: 
                img_tensor = img_tensor.squeeze(1)
            else:
                img_tensor = img_tensor.squeeze(0)
        
        # 2. 此时应该是 4D: [B, C, H, W]
        # 如果 Channel 在最后，转置它
        if img_tensor.dim() == 4 and img_tensor.shape[-1] == 3:
            img_tensor = img_tensor.permute(0, 3, 1, 2)
            
        # 3. 标准化为 5D: [B, N=1, C, H, W]
        if img_tensor.dim() == 4:
            img_tensor = img_tensor.unsqueeze(1)
            
        batch['img'] = img_tensor
    # ==========================================

    id_batch = [0] * len(img_list)
    infer_model = model.model if hasattr(model, 'model') else model

    with torch.no_grad():
        infer_model._initialize_batch(batch)
        outputs = infer_model.run_inference_batch(
            img_list, batch, inference_type="full", idx_path=idx_path, 
            idx_dict=idx_dict, mhr_shape_scale_dict=mhr_shape_scale_dict, 
            id_batch=id_batch, occ_dict=occ_dict
        )
        
        pose_output = outputs[0]
        mhr_output = pose_output["mhr"]
        


        # =========================================================
        # 动作平滑 (Bi-Mamba) & Mesh 重建 (100% 对齐源码版)
        # =========================================================
        smoother = load_smoother_model(device="cuda")
        
        if smoother is not None and "pred_pose_raw" in mhr_output:
            raw_pose = mhr_output["pred_pose_raw"] # (B, 266)
            seq_input = raw_pose.unsqueeze(0)      # (1, B, 266)
            
            # 1. 执行 Mamba 平滑
            smoothed_pose = smoother(seq_input).squeeze(0) # (B, 266)
            mhr_output["pred_pose_raw"] = smoothed_pose
            print(" [Mamba] Pose smoothed successfully!")
            

            # 2. 严格遵循 mhr_head.py 的官方解码逻辑
            try:
                import roma
                # 【路径修复】：直接从 sam_3d_body 的根目录导入
                from sam_3d_body.modules import rot6d_to_rotmat
                from sam_3d_body.modules.mhr_utils import compact_cont_to_model_params_body, mhr_param_hand_mask
                
                head = getattr(infer_model, 'head_pose', infer_model)
                if hasattr(head, 'head_module'): head = head.head_module
                
                if hasattr(head, 'mhr_forward'):
                    # (a) 分离 6D 旋转和连续姿态
                    global_rot_6d = smoothed_pose[:, :6]
                    pred_pose_cont = smoothed_pose[:, 6:]
                    
                    # (b) 转换旋转矩阵
                    global_rot_rotmat = rot6d_to_rotmat(global_rot_6d)
                    global_rot_euler = roma.rotmat_to_euler("ZYX", global_rot_rotmat)
                    global_trans = torch.zeros_like(global_rot_euler)
                    
                    # (c) 转换连续姿态并应用 Mask
                    pred_pose_euler = compact_cont_to_model_params_body(pred_pose_cont)
                    pred_pose_euler[:, mhr_param_hand_mask] = 0
                    pred_pose_euler[:, -3:] = 0
                    
                    # (d) 调用官方的 mhr_forward
                    verts = head.mhr_forward(
                        global_trans=global_trans,
                        global_rot=global_rot_euler,
                        body_pose_params=pred_pose_euler,
                        hand_pose_params=mhr_output.get("hand"),
                        scale_params=mhr_output.get("scale"),
                        shape_params=mhr_output.get("shape"),
                        expr_params=mhr_output.get("face"),
                        do_pcblend=True,
                        return_keypoints=False,
                        return_joint_coords=False,
                        return_model_params=False,
                        return_joint_rotations=False,
                    )
                    
                    # (e) 官方的坐标系反转魔法
                    verts[..., [1, 2]] *= -1
                    
                    mhr_output["pred_vertices"] = verts
                    print("✅ [Mamba] 3D Mesh updated seamlessly with official API!")
                else:
                    print("⚠️ [Mamba] Head lacks mhr_forward method.")
            except Exception as e:
                print(f" ")
        # =========================================================


#     # 结果封装
#     pred_vertices = mhr_output["pred_vertices"]
#     if pred_vertices.dim() == 4: pred_vertices = pred_vertices[:, 0]
#     pred_vertices = pred_vertices.detach().cpu().numpy()
    
#     full_output_list = [{"vertices": v} for v in pred_vertices]
#     return full_output_list, [0]*len(img_list), []

    # ==========================================
    # 结果封装 (修改版：附带保存姿态三要素供体外解码)
    # ==========================================
    pred_vertices = mhr_output["pred_vertices"]
    if pred_vertices.dim() == 4: pred_vertices = pred_vertices[:, 0]
    pred_vertices = pred_vertices.detach().cpu().numpy()
    
    # 安全提取三个核心参数 (如果经过了 Mamba，这里的 pose 就是平滑后的！)
    pose_t = mhr_output.get("pred_pose_raw", mhr_output.get("pred_pose"))
    shape_t = mhr_output.get("pred_shape_raw", mhr_output.get("pred_shape"))
    exp_t = mhr_output.get("pred_exp_raw", mhr_output.get("pred_exp"))
    
    full_output_list = []
    for i, v in enumerate(pred_vertices):
        out_dict = {"vertices": v}
        # 存入字典供体外脚本使用
        if pose_t is not None: out_dict["pose"] = pose_t[i].detach().cpu().numpy()
        if shape_t is not None: out_dict["shape"] = shape_t[i].detach().cpu().numpy()
        if exp_t is not None: out_dict["exp"] = exp_t[i].detach().cpu().numpy()
        
        full_output_list.append(out_dict)
        
    return full_output_list, [0]*len(img_list), []

def save_mesh_results(outputs, faces, save_dir, image_path, id_current):
    import trimesh
    if outputs is None: return
    vertices = outputs["vertices"]
    frame_name = os.path.basename(image_path).split('.')[0]
    save_path = os.path.join(save_dir, f"{frame_name}.ply")
    trimesh.Trimesh(vertices=vertices, faces=faces).export(save_path)