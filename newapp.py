import os
import sys
import uuid
import time
import cv2
import glob
import random
import numpy as np
import torch
import torch.nn.functional as F
import gradio as gr

from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from omegaconf import OmegaConf

# ===============================
# 1. 环境与路径设置
# ===============================
ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["GRADIO_TEMP_DIR"] = os.path.join(ROOT, "gradio_tmp")
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'models', 'sam_3d_body'))
sys.path.append(os.path.join(current_dir, 'models', 'diffusion_vas'))

from utils import (
    draw_point_marker, mask_painter, images_to_mp4, DAVIS_PALETTE, 
    jpg_folder_to_mp4, is_super_long_or_wide, keep_largest_component, 
    is_skinny_mask, bbox_from_mask, gpu_profile, resize_mask_with_unique_label
)

from models.sam_3d_body.sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from models.sam_3d_body.notebook.utils import process_image_with_mask, save_mesh_results
from models.diffusion_vas.demo import (
    init_amodal_segmentation_model, init_rgb_model, init_depth_model, 
    load_and_transform_masks, load_and_transform_rgbs, rgb_to_depth
)

# ===============================
# 2. 显卡设置
# ===============================
torch.cuda.empty_cache()
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# ===============================
# 3. 全局变量定义
# ===============================
CONFIG = None
sam3_model = None
predictor = None
sam3_3d_body_model = None

# RUNTIME: 唯一的真理来源
RUNTIME = {
    'video_path': None,          # 当前用户界面上显示的视频路径
    'current_loaded_path': None, # 模型内部实际加载的视频路径
    'clicks': {},
    'masks': {},
    'batch_size': 1,
    'inference_state': None,     # SAM3 状态对象
    'id': 1,
    'objects': {},
    'out_obj_ids': []
} 
OUTPUT_DIR = ""

pipeline_mask = None
pipeline_rgb = None
depth_model = None
max_occ_len = None
generator = None

# ===============================
# 4. 辅助函数
# ===============================
def gen_id():
    t = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    u = uuid.uuid4().hex[:8]
    return f"{t}_{u}"

def build_sam3_from_config(cfg):
    from models.sam3.sam3.model_builder import build_sam3_video_model
    sam3_model = build_sam3_video_model(checkpoint_path=cfg.sam3['ckpt_path'])
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone
    return sam3_model, predictor

def build_sam3_3d_body_config(cfg):
    mhr_path = cfg.sam_3d_body['mhr_path']
    fov_path = cfg.sam_3d_body['fov_path']
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, model_cfg = load_sam_3d_body(
        cfg.sam_3d_body['ckpt_path'], device=device, mhr_path=mhr_path
    )
    
    human_detector, human_segmentor = None, None
    from models.sam_3d_body.tools.build_fov_estimator import FOVEstimator
    fov_estimator = FOVEstimator(name='moge2', device=device, path=fov_path)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )
    return estimator

def build_diffusion_vas_config(cfg):
    model_path_mask = cfg.completion['model_path_mask']
    model_path_rgb = cfg.completion['model_path_rgb']
    depth_encoder = cfg.completion['depth_encoder']
    model_path_depth = cfg.completion['model_path_depth']
    max_occ_len = min(cfg.completion['max_occ_len'], cfg.sam_3d_body['batch_size'])

    generator = torch.manual_seed(23)

    pipeline_mask = init_amodal_segmentation_model(model_path_mask)
    pipeline_rgb = init_rgb_model(model_path_rgb)
    depth_model = init_depth_model(model_path_depth, depth_encoder)

    return pipeline_mask, pipeline_rgb, depth_model, max_occ_len, generator

def init_runtime(config_path: str = os.path.join(ROOT, "configs", "body4d.yaml")):
    global CONFIG, sam3_model, predictor, sam3_3d_body_model, RUNTIME, OUTPUT_DIR
    global pipeline_mask, pipeline_rgb, depth_model, max_occ_len, generator
    
    print("[DEBUG] Initializing Runtime...")
    CONFIG = OmegaConf.load(config_path)
    sam3_model, predictor = build_sam3_from_config(CONFIG)
    sam3_3d_body_model = build_sam3_3d_body_config(CONFIG)

    if CONFIG.completion.get('enable', False):
        pipeline_mask, pipeline_rgb, depth_model, max_occ_len, generator = build_diffusion_vas_config(CONFIG)
    else:
        pipeline_mask, pipeline_rgb, depth_model, max_occ_len, generator = None, None, None, None, None
    
    OUTPUT_DIR = os.path.join(CONFIG.runtime['output_dir'], gen_id())
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    RUNTIME['batch_size'] = CONFIG.sam_3d_body.get('batch_size', 1)
    RUNTIME['detection_resolution'] = CONFIG.completion.get('detection_resolution', [256, 512])
    RUNTIME['completion_resolution'] = CONFIG.completion.get('completion_resolution', [512, 1024])
    print(f"[DEBUG] Runtime initialized.")

# ===============================
# 5. 视频处理逻辑
# ===============================
EXAMPLE_1 = os.path.join(ROOT, "assets", "examples", "example1.mp4")
EXAMPLE_2 = os.path.join(ROOT, "assets", "examples", "example2.mp4")
EXAMPLE_3 = os.path.join(ROOT, "assets", "examples", "example3.mp4")
SUPPORTED_EXTS = {".mp4",}

def read_video_metadata(path: str):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    RUNTIME['video_fps'] = fps
    return fps, total

def read_frame_at(path: str, idx: int):
    if not os.path.exists(path): return None
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok: return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)

def get_thumb(path: str):
    if not os.path.exists(path): return None
    return read_frame_at(path, 0)

EX1_THUMB = get_thumb(EXAMPLE_1)
EX2_THUMB = get_thumb(EXAMPLE_2)
EX3_THUMB = get_thumb(EXAMPLE_3)

def prepare_video(path: str):
    """
    核心：这里更新了全局 RUNTIME['video_path']。
    因为 on_upload 是同步执行的，这里更新完，之后的 on_click 读到的绝对是最新的。
    """
    global RUNTIME
    torch.cuda.empty_cache()
    print(f"\n>>> [DEBUG] prepare_video CALLED. Path received: {path}")
    
    if path is None or not os.path.exists(path):
        print(f">>> [ERROR] Path is invalid or not found: {path}")
        return (None, 1.0, None, gr.update(minimum=0, maximum=0, value=0), "00:00 / 00:00", "No video loaded")

    # 【更新全局状态】
    RUNTIME['video_path'] = path
    # 重置点击/Mask数据，防止旧视频数据残留
    RUNTIME['clicks'] = {}
    RUNTIME['masks'] = {}
    RUNTIME['id'] = 1
    
    print(f">>> [DEBUG] GLOBAL PATH UPDATED to: {RUNTIME['video_path']}")

    fps, total = read_video_metadata(path)
    first_frame = read_frame_at(path, 0)
    
    slider_cfg = gr.update(minimum=0, maximum=total - 1, value=0)
    dur = total / fps
    total_text = f"{int(dur // 60):02d}:{int(dur % 60):02d}"
    time_text = f"00:00 / {total_text}"
    status_text = f"Loaded: {os.path.basename(path)}"

    # 这里我们只更新路径，真正的模型初始化放到 on_click（Lazy Load）
    
    return path, fps, first_frame, slider_cfg, time_text, status_text

# ===============================
# 6. 事件处理函数
# ===============================

def on_upload(file_path):
    print(f"[DEBUG] on_upload triggered with: {file_path}")
    return prepare_video(file_path)

def on_example_select(evt: gr.SelectData):
    idx = evt.index
    if isinstance(idx, (list, tuple)): idx = idx[0]

    if idx == 0: path = EXAMPLE_1
    elif idx == 1: path = EXAMPLE_2
    elif idx == 2: path = EXAMPLE_3
    else: return gr.skip()

    if not os.path.exists(path):
        raise gr.Error(f"Video not found: {path}")
        
    return prepare_video(path)

def update_frame(idx, fps):
    # 【改动】不再接收 video_state，直接查全局
    path = RUNTIME.get('video_path')
    if path is None or not os.path.exists(path): 
        return None, "00:00 / 00:00"

    idx = int(idx)
    frame = read_frame_at(path, idx)
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    dur = total / fps if fps > 0 else 0.0
    cur_sec = idx / fps if fps > 0 else 0.0
    cur_text = f"{int(cur_sec // 60):02d}:{int(cur_sec % 60):02d}"
    end_text = f"{int(dur // 60):02d}:{int(dur % 60):02d}"
    return frame, f"{cur_text} / {end_text}"

def on_click(evt: gr.SelectData, point_type: str, frame_idx: int):

    global RUNTIME
    
    # 1. 强制使用全局路径
    target_path = RUNTIME.get('video_path')
    print(f"\n>>> [DEBUG] on_click executing for GLOBAL path: {target_path}")
    
    if not target_path or not os.path.exists(target_path):
        print(">>> [FATAL] No valid video loaded in backend. Please upload/select a video.")
        return gr.skip()

    # 2. 状态切换检查（Lazy Loading）
    # 检查模型当前加载的，是不是我们想要的那个
    if RUNTIME['current_loaded_path'] != target_path:
        print(f">>> [SWITCH] Detecting video change.")
        print(f"    Current Model holds: {RUNTIME['current_loaded_path']}")
        print(f"    User wants to edit: {target_path}")
        print(">>> [INIT] Re-initializing SAM3 state...")
        
        try:
            RUNTIME['inference_state'] = predictor.init_state(video_path=target_path)
            predictor.clear_all_points_in_video(RUNTIME['inference_state'])
            
            # 只有初始化成功了，才更新 current_loaded_path
            RUNTIME['current_loaded_path'] = target_path
            RUNTIME['clicks'] = {}
            RUNTIME['masks'] = {}
            RUNTIME['id'] = 1
            RUNTIME['out_obj_ids'] = []
            print(">>> [INIT] Done.")
        except Exception as e:
            print(f">>> [ERROR] Init state failed: {e}")
            return gr.skip()

    # 3. 读取帧 (修复了 NameError)
    try:
        frame = read_frame_at(target_path, int(frame_idx))
    except Exception as e:
        print(f"[ERROR] Read frame failed: {e}")
        return gr.skip()

    if frame is None:
        print("[ERROR] Frame is None")
        return gr.skip()

    # 显式获取尺寸，防止 NameError
    width, height = frame.size
    x, y = evt.index
    print(f"[DEBUG] Click ({x}, {y}) on frame {frame_idx} of {os.path.basename(target_path)}")

    point_type_norm = point_type.lower()
    if point_type_norm not in ("positive", "negative"): point_type_norm = "positive"

    # 4. 记录点击
    if frame_idx not in RUNTIME['clicks']: RUNTIME['clicks'][frame_idx] = []
    RUNTIME['clicks'][frame_idx].append((x, y, point_type))
    
    pts = []
    lbs = []
    for (cx, cy, ct) in RUNTIME['clicks'][frame_idx]:
        pts.append([int(cx), int(cy)])
        lbs.append(1 if ct.lower() == "positive" else 0)
        
    input_point = np.array(pts, dtype=np.int32)
    input_label = np.array(lbs, dtype=np.int32)
    
    rel_points = [[cx / width, cy / height] for cx, cy in input_point]
    points_tensor = torch.tensor(rel_points, dtype=torch.float32)
    points_labels_tensor = torch.tensor(input_label, dtype=torch.int32)

    # 5. 调用模型
    # 此时 inference_state 已经保证是 target_path 的了
    try:
        _, RUNTIME['out_obj_ids'], low_res_masks, video_res_masks = predictor.add_new_points_or_box(
            inference_state=RUNTIME['inference_state'],
            frame_idx=frame_idx,
            obj_id=RUNTIME['id'],
            points=points_tensor,
            labels=points_labels_tensor,
        )
    except Exception as e:
        print(f"[ERROR] Predictor failed: {e}")
        return gr.skip()
    
    # 6. 绘图
    mask_np = (video_res_masks[-1, 0].detach().cpu().numpy() > 0)
    mask = (mask_np > 0).astype(np.uint8) * 255
    painted_image = mask_painter(np.array(frame, dtype=np.uint8), mask, mask_color=4+RUNTIME['id'])
    
    if RUNTIME['id'] not in RUNTIME['masks']: RUNTIME['masks'][RUNTIME['id']] = {}
    RUNTIME['masks'][RUNTIME['id']][frame_idx] = mask

    for k, v in RUNTIME['masks'].items():
        if k == RUNTIME['id']: continue
        if frame_idx in v:
            painted_image = mask_painter(painted_image, v[frame_idx], mask_color=4+k)

    frame = Image.fromarray(painted_image)
    updated = draw_point_marker(frame, x, y, point_type_norm)
    return updated

def add_target(targets, selected):
    name = f"Target {len(targets) + 1}"
    if RUNTIME['clicks'] == {}:
        return targets, selected, gr.update(choices=targets, value=selected)
    targets = targets + [name]
    selected = selected + [name]
    RUNTIME['objects'][RUNTIME['id']] = RUNTIME['clicks']
    RUNTIME['id'] += 1
    RUNTIME['clicks'] = {}
    return targets, selected, gr.update(choices=targets, value=selected)

def toggle_upload(open_state: bool):
    new_state = not open_state
    label = "Upload Video (click to close)" if new_state else "Upload Video (click to open)"
    return new_state, gr.update(visible=new_state), gr.update(value=label)

def on_mask_generation():
    # 同样不接受参数，只信全局变量
    video_path = RUNTIME.get('video_path')
    if not video_path: raise gr.Error("No video loaded.")
    
    print(f"[DEBUG] Mask Generation Started for: {video_path}")
    
    # 再次确认
    if RUNTIME['current_loaded_path'] != video_path:
        print("[WARN] State mismatch at Generation. Forcing reload...")
        RUNTIME['inference_state'] = predictor.init_state(video_path=video_path)
        RUNTIME['current_loaded_path'] = video_path

    video_segments = {}
    for step_result in predictor.propagate_in_video(
        RUNTIME['inference_state'],
        start_frame_idx=0,
        max_frame_num_to_track=1800,
        reverse=False,
        propagate_preflight=True,
    ):
        if len(step_result) == 6:
            frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores, iou_scores = step_result
        elif len(step_result) == 5:
            frame_idx, obj_ids, video_res_masks, obj_scores, iou_scores = step_result
        elif len(step_result) == 3:
             frame_idx, obj_ids, video_res_masks = step_result
        else:
             frame_idx, obj_ids = step_result[0], step_result[1]
             video_res_masks = step_result[2]

        video_segments[frame_idx] = {
            out_obj_id: (video_res_masks[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(RUNTIME['out_obj_ids'])
        } 

    out_h = RUNTIME['inference_state']['video_height']
    out_w = RUNTIME['inference_state']['video_width']
    img_to_video = []
    
    IMAGE_PATH = os.path.join(OUTPUT_DIR, 'images')
    MASKS_PATH = os.path.join(OUTPUT_DIR, 'masks')
    os.makedirs(IMAGE_PATH, exist_ok=True)
    os.makedirs(MASKS_PATH, exist_ok=True)
    
    sorted_frames = sorted(video_segments.keys())
    for out_frame_idx in sorted_frames:
        img = RUNTIME['inference_state']['images'][out_frame_idx].detach().float().cpu()
        img = (img + 1) / 2
        img = img.clamp(0, 1)
        img = F.interpolate(img.unsqueeze(0), size=(out_h, out_w), mode="bilinear", align_corners=False).squeeze(0)
        img = img.permute(1, 2, 0).numpy() * 255
        img = img.astype("uint8")
        
        img_pil = Image.fromarray(img).convert('RGB')
        msk = np.zeros_like(img[:, :, 0])
        
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            mask = (out_mask[0] > 0).astype(np.uint8) * 255
            
            if mask.shape[:2] != img.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            img = mask_painter(img, mask, mask_color=4 + out_obj_id)
            msk[mask == 255] = out_obj_id
            
        img_to_video.append(img)
        msk_pil = Image.fromarray(msk).convert('P')
        msk_pil.putpalette(DAVIS_PALETTE)
        
        img_pil.save(os.path.join(IMAGE_PATH, f"{out_frame_idx:08d}.jpg"))
        msk_pil.save(os.path.join(MASKS_PATH, f"{out_frame_idx:08d}.png"))
        
    out_video_path = os.path.join(OUTPUT_DIR, f"video_mask_{time.time():.0f}.mp4")
    images_to_mp4(img_to_video, out_video_path, fps=RUNTIME['video_fps'])
    return out_video_path

def on_4d_generation():
    # 同样只信全局
    torch.cuda.empty_cache()
    video_path = RUNTIME.get('video_path')
    if not video_path: raise gr.Error("No video loaded.")

    print(f"[DEBUG] 4D Generation Started for: {video_path}")

    IMAGE_PATH = os.path.join(OUTPUT_DIR, 'images')
    MASKS_PATH = os.path.join(OUTPUT_DIR, 'masks')
    
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.tiff", "*.webp"]
    
    images_list = sorted([
        image
        for ext in image_extensions
        for image in glob.glob(os.path.join(IMAGE_PATH, ext))
    ])
    masks_list = sorted([
        image
        for ext in image_extensions
        for image in glob.glob(os.path.join(MASKS_PATH, ext))
    ])
    
    if len(images_list) == 0:
        raise gr.Error("No Mask/Images found. Please run Mask Generation first.")

    os.makedirs(f"{OUTPUT_DIR}/rendered_frames", exist_ok=True)
    for obj_id in RUNTIME['out_obj_ids']:
        os.makedirs(f"{OUTPUT_DIR}/mesh_4d_individual/{obj_id}", exist_ok=True)

    batch_size = RUNTIME['batch_size']
    n = len(images_list)
    
    pred_res = RUNTIME['detection_resolution']
    pred_res_hi = RUNTIME['completion_resolution']
    modal_pixels_list = []
    
    if pipeline_mask is not None:
        print("[DEBUG] Running Diffusion VAS Pipeline...")
        for obj_id in RUNTIME['out_obj_ids']:
            modal_pixels, ori_shape = load_and_transform_masks(OUTPUT_DIR + "/masks", resolution=pred_res, obj_id=obj_id)
            modal_pixels_list.append(modal_pixels)
        rgb_pixels, _, raw_rgb_pixels = load_and_transform_rgbs(OUTPUT_DIR + "/images", resolution=pred_res)
        depth_pixels = rgb_to_depth(rgb_pixels, depth_model)

    mhr_shape_scale_dict = {}
    
    print(f"[DEBUG] Processing {n} frames in batches of {batch_size}...")
    
    for i in tqdm(range(0, n, batch_size)):
        batch_images = images_list[i:i + batch_size]
        batch_masks  = masks_list[i:i + batch_size]

        idx_dict = {}
        idx_path = {}
        occ_dict = {}
        
        for obj_id in RUNTIME['out_obj_ids']:
            occ_dict[obj_id] = [1] * len(batch_masks)

        mask_outputs, id_batch, empty_frame_list = process_image_with_mask(
            sam3_3d_body_model, 
            batch_images, 
            batch_masks, 
            idx_path, 
            idx_dict, 
            mhr_shape_scale_dict, 
            occ_dict
        )
        
        for frame_id in range(len(batch_images)):
            image_path = batch_images[frame_id]
            if frame_id in empty_frame_list:
                mask_output = None
                id_current = None
            else:
                mask_output = mask_outputs[frame_id]
                id_current = id_batch[frame_id]
            
            save_mesh_results(
                outputs=mask_output, 
                faces=sam3_3d_body_model.faces, 
                save_dir=f"{OUTPUT_DIR}/mesh_4d_individual",
                image_path=image_path,
                id_current=id_current,
            )

    print("[DEBUG] Generating placeholder video...")
    out_4d_path = os.path.join(OUTPUT_DIR, f"4d_{time.time():.0f}.mp4")
    
    dummy_frames = []
    W, H = 512, 512
    img = Image.new('RGB', (W, H), color = (0, 0, 0))
    d = ImageDraw.Draw(img)
    text = "Success!\n\nVisualization Skipped.\nMesh files saved in:\noutput/mesh_4d_individual"
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
        
    d.text((50,200), text, fill=(255,255,255), font=font)
    img_np = np.array(img)
    
    for _ in range(int(RUNTIME['video_fps'] * 3)):
        dummy_frames.append(img_np)
        
    images_to_mp4(dummy_frames, out_4d_path, fps=RUNTIME['video_fps'])
    
    return out_4d_path

on_mask_generation = gpu_profile(on_mask_generation)
on_4d_generation   = gpu_profile(on_4d_generation)

# ===============================
# 7. UI 构建
# ===============================
with gr.Blocks(title="SAM-Body4D") as demo:
    video_state = gr.State(value=None)
    fps_state = gr.State(1.0)
    point_type_state = gr.State("positive")
    targets_state = gr.State([])
    selected_targets_state = gr.State([])
    upload_open_state = gr.State(False)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Example Videos")
            examples_gallery = gr.Gallery(
                value=[(EX1_THUMB, "Example 1"), (EX2_THUMB, "Example 2"), (EX3_THUMB, "Example 3")],
                show_label=False, columns=3, height=160,
            )

            current_frame = gr.Image(
                label="Current Frame", interactive=True, type="pil"
            )
            
            # 添加一个状态栏，让你知道后台到底加载了啥
            status_info = gr.Markdown("No video loaded.")

            toggle_upload_btn = gr.Button("Upload Video (click to open)", size="sm", variant="secondary")
            upload_panel = gr.Row(visible=False)
            with upload_panel:
                upload = gr.File(label="Video File", file_count="single", type="filepath")

            frame_slider = gr.Slider(minimum=0, maximum=0, value=0, step=1, label="Frame Index")
            time_text = gr.Text("00:00 / 00:00", label="Time")
            point_radio = gr.Radio(choices=["Positive", "Negative"], value="Positive", label="Point Type")
            targets_box = gr.CheckboxGroup(label="Targets", choices=[], value=[])
            add_target_btn = gr.Button("Add Target")

        with gr.Column(scale=1):
            result_display = gr.Video(label="Segmentation Result")
            with gr.Row():
                mask_gen_btn = gr.Button("Mask Generation")
                gen4d_btn = gr.Button("4D Generation (Save Mesh)")
            fourd_display = gr.Video(label="4D Result (Mesh Saved)")

    toggle_upload_btn.click(fn=toggle_upload, inputs=[upload_open_state], outputs=[upload_open_state, upload_panel, toggle_upload_btn])
    
    # Upload 更新 status_info
    upload.change(
        fn=on_upload, 
        inputs=[upload], 
        outputs=[video_state, fps_state, current_frame, frame_slider, time_text, status_info]
    )
    
    # Gallery 更新 status_info
    examples_gallery.select(
        fn=on_example_select, 
        inputs=None, 
        outputs=[video_state, fps_state, current_frame, frame_slider, time_text, status_info]
    )
    
    frame_slider.change(fn=update_frame, inputs=[frame_slider, video_state, fps_state], outputs=[current_frame, time_text])
    
    point_radio.change(fn=lambda v: v.lower(), inputs=[point_radio], outputs=[point_type_state])
    
    # 【核心改动】移除 video_state 输入
    current_frame.select(
        fn=on_click, 
        inputs=[point_type_state, frame_slider], # 移除了 video_state
        outputs=[current_frame]
    )
    
    add_target_btn.click(fn=add_target, inputs=[targets_state, selected_targets_state], outputs=[targets_state, selected_targets_state, targets_box])
    
    # 移除 video_state 输入
    mask_gen_btn.click(fn=on_mask_generation, inputs=None, outputs=[result_display])
    
    # 移除 video_state 输入
    gen4d_btn.click(fn=on_4d_generation, inputs=None, outputs=[fourd_display])

if __name__ == "__main__":
    init_runtime()
    print("\n[AUTO-LOAD] Pre-loading Example 1...")
    if os.path.exists(EXAMPLE_1):
        prepare_video(EXAMPLE_1)
    else:
        print("[WARN] Example 1 not found. Please upload a video manually.")

    demo.launch(share=True, server_name="0.0.0.0")