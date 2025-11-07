import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
import json
import copy
import glob
import argparse
from ultralytics import YOLO
import pycocotools.mask as mask_util
import shutil
import multiprocessing
import sys
import gc

# Set Grounded-SAM-2 base path (relative to current script location)
# Current script is in STVG-R1/Time-R1/preprocess/
# Grounded-SAM-2-main is in STVG-R1/Grounded-SAM-2-main/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GROUNDED_SAM2_BASE = os.path.join(SCRIPT_DIR, "..", "..", "Grounded-SAM-2-main")
GROUNDED_SAM2_BASE = os.path.abspath(GROUNDED_SAM2_BASE)

# Add Grounded-SAM-2 to Python path for imports
if GROUNDED_SAM2_BASE not in sys.path:
    sys.path.insert(0, GROUNDED_SAM2_BASE)

from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo

def mask_to_rle(mask):
    """Convert binary mask to RLE format"""
    mask = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_util.encode(mask)
    if isinstance(rle['counts'], bytes):
        rle['counts'] = rle['counts'].decode('utf-8')
    return rle

def save_mask_data(mask_array, frame_masks, mask_data_dir, json_data_dir, image_base_name):
    """Save mask array, RLE data and JSON data"""
    mask_path = os.path.join(mask_data_dir, f"mask_{image_base_name}.npy")
    json_path = os.path.join(json_data_dir, f"mask_{image_base_name}.json")
    rle_path = os.path.join(mask_data_dir, f"mask_{image_base_name}.json")
    
    # Save mask array
    np.save(mask_path, mask_array)
    
    # Build RLE data
    rle_data = {
        "height": mask_array.shape[0],
        "width": mask_array.shape[1],
        "objects": {}
    }
    
    for obj_id in frame_masks.labels:
        obj_mask = (mask_array == obj_id).astype(np.uint8)
        if np.any(obj_mask):
            rle = mask_to_rle(obj_mask)
            obj_info = frame_masks.labels[obj_id]
            box_data = None
            if hasattr(obj_info, 'box') and obj_info.box is not None:
                box_data = obj_info.box.tolist() if isinstance(obj_info.box, torch.Tensor) else obj_info.box
            rle_data["objects"][str(obj_id)] = {
                "rle": rle,
                "class_name": obj_info.class_name,
                "box": box_data
            }
    
    # Save RLE and JSON data
    with open(rle_path, 'w') as f:
        json.dump(rle_data, f)
    frame_masks.to_json(json_path)

def draw_masks_from_rle(video_dir, mask_data_dir, json_data_dir, output_dir, overlay_on_original=True):
    """
    Visualize masks directly from RLE format JSON files
    
    Args:
    video_dir: Original video frames directory
    mask_data_dir: Directory containing RLE format mask data
    json_data_dir: Directory containing object info JSON files
    output_dir: Output directory for visualization results
    overlay_on_original: True to overlay on original image, False to overlay on white background
    """
    
    os.makedirs(output_dir, exist_ok=True)
    mask_files = [f for f in os.listdir(mask_data_dir) if f.startswith("mask_") and f.endswith(".json")]
    mask_files.sort(key=lambda f: int(f.split("_")[1].split(".")[0]))
    color_map = {}
    for mask_file in mask_files:
        frame_index = int(mask_file.split("_")[1].split(".")[0])
        frame_path = os.path.join(video_dir, f"{frame_index}.jpg")
        if not os.path.exists(frame_path):
            frame_path = os.path.join(video_dir, f"{frame_index}.png")
        
        if not os.path.exists(frame_path):
            continue
        
        image = cv2.imread(frame_path)
        if image is None:
            continue
            
        mask_path = os.path.join(mask_data_dir, mask_file)
        with open(mask_path, 'r') as f:
            mask_data = json.load(f)
        json_path = os.path.join(json_data_dir, mask_file)
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                json_data = json.load(f)
        else:
            json_data = {"labels": {}}
        
        # Choose background based on parameter
        overlay = image.copy() if overlay_on_original else np.ones_like(image) * 255
        
        for obj_id, obj_info in mask_data["objects"].items():
            if obj_id not in color_map:
                color_map[obj_id] = (
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255)
                )
            
            rle = obj_info["rle"]
            mask = mask_util.decode(rle)
            color = color_map[obj_id]
            colored_mask = np.zeros_like(image)
            colored_mask[mask > 0] = color
            mask_area = mask > 0
            
            if overlay_on_original:
                overlay[mask_area] = (overlay[mask_area] * 0.3 + colored_mask[mask_area] * 0.7).astype(np.uint8)
            else:
                overlay[mask_area] = colored_mask[mask_area]
        
        output_path = os.path.join(output_dir, f"{frame_index}.jpg")
        cv2.imwrite(output_path, overlay)

def find_video_path(video_name, video_base_dir):
    """Find video path in hcstvg dataset structure (searches in subfolders 0-9)"""
    for folder in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        potential_path = os.path.join(video_base_dir, folder, video_name)
        if os.path.exists(potential_path):
            return potential_path
    return None

def extract_frames(video_path, output_dir):
    """Extract frames from video"""
    CommonUtils.creat_dirs(output_dir)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    success = True
    
    while success:
        success, frame = cap.read()
        if success:
            output_path = os.path.join(output_dir, f"{frame_count}.jpg")
            cv2.imwrite(output_path, frame)
            frame_count += 1
    
    cap.release()
    return frame_count

def get_video_fps(video_path):
    """Get video FPS"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def should_process_class(class_name, target_mode):
    """
    Determine if a class should be processed based on target mode
    
    Args:
    class_name: Name of the detected class
    target_mode: "person", "non-person", or "all"
    
    Returns:
    True if the class should be processed, False otherwise
    """
    if target_mode == "all":
        return True
    elif target_mode == "person":
        return class_name == "person"
    elif target_mode == "non-person":
        return class_name != "person"
    else:
        # Default to non-person if invalid mode
        return class_name != "person"

def process_video(video_name, video_info, config):
    """Process single video - single GPU version"""
    
    device = "cuda:0"
    torch.cuda.set_device(0)
    
    video_predictor = None
    sam2_image_model = None
    image_predictor = None
    yolo_model = None
    inference_state = None
    
    # Get target mode from config
    target_mode = config.get("target_mode", "non-person")
    
    def force_cleanup():
        """Clean up all models and variables"""
        nonlocal video_predictor, sam2_image_model, image_predictor, yolo_model, inference_state
        
        def safe_delete(obj, cleanup_func=None):
            if obj is not None:
                try:
                    if cleanup_func:
                        cleanup_func(obj)
                except:
                    pass
                try:
                    del obj
                except:
                    pass
            return None
        
        if video_predictor is not None:
            safe_delete(video_predictor, lambda x: x.reset_state(None))
            video_predictor = None
        if sam2_image_model is not None:
            safe_delete(sam2_image_model, lambda x: x.clear_cache() if hasattr(x, 'clear_cache') else None)
            sam2_image_model = None
        if image_predictor is not None:
            safe_delete(image_predictor, lambda x: x.clear_cache() if hasattr(x, 'clear_cache') else None)
            image_predictor = None
        if yolo_model is not None:
            try:
                if hasattr(yolo_model, 'clear_cache'):
                    yolo_model.clear_cache()
                if hasattr(yolo_model, 'model'):
                    del yolo_model.model
            except:
                pass
            yolo_model = None
        if inference_state is not None:
            safe_delete(inference_state, lambda x: x.clear() if hasattr(x, 'clear') else None)
            inference_state = None
        
        # Force GPU memory cleanup
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
        except:
            pass
    
    try:
        print(f"Processing video: {video_name}")
        
        sam2_checkpoint = config["sam2_checkpoint"]
        model_cfg = config["model_cfg"]
        
        # Save current working directory
        original_cwd = os.getcwd()
        try:
            # Change to Grounded-SAM-2-main directory for config loading
            os.chdir(GROUNDED_SAM2_BASE)
            # Use relative path for model_cfg (relative to GROUNDED_SAM2_BASE)
            model_cfg_rel = os.path.relpath(model_cfg, GROUNDED_SAM2_BASE)
            video_predictor = build_sam2_video_predictor(model_cfg_rel, sam2_checkpoint)
            sam2_image_model = build_sam2(model_cfg_rel, sam2_checkpoint, device=device)
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
        
        image_predictor = SAM2ImagePredictor(sam2_image_model)
        yolo_model = YOLO("/data/yolov12/yolov12x.pt")
        
        video_path = find_video_path(video_name, config["video_base_dir"])
        if not video_path:
            print(f"Could not find video: {video_name}, skipping...")
            return False
        
        original_fps = get_video_fps(video_path)
        video_name_without_ext = os.path.splitext(video_name)[0]
        video_output_dir = os.path.join(config["base_output_dir"], video_name_without_ext)
        CommonUtils.creat_dirs(video_output_dir)
        
        frames_dir = os.path.join(video_output_dir, "frames")
        if not os.path.exists(frames_dir) or len(os.listdir(frames_dir)) == 0:
            extract_frames(video_path, frames_dir)
        
        mask_data_dir = os.path.join(video_output_dir, "mask_data")
        json_data_dir = os.path.join(video_output_dir, "json_data")
        for d in [mask_data_dir, json_data_dir]:
            CommonUtils.creat_dirs(d)
        
        frame_names = [p for p in os.listdir(frames_dir) if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        
        inference_state = video_predictor.init_state(video_path=frames_dir, offload_video_to_cpu=True, async_loading_frames=True)
        step = 15
        PROMPT_TYPE_FOR_VIDEO = "mask"
        objects_count = 0
        frame_object_count = {}
        global_mask_dict = MaskDictionaryModel()
        
        # Step 1: Process first frame, detect targets and track entire video
        first_frame_idx = 0
        img_path = os.path.join(frames_dir, frame_names[first_frame_idx])
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)
        image_base_name = frame_names[first_frame_idx].split(".")[0]
        
        first_mask_dict = MaskDictionaryModel(promote_type=PROMPT_TYPE_FOR_VIDEO, mask_name=f"mask_{image_base_name}.npy")
        results = yolo_model.predict(source=img_path, conf=0.25, verbose=False)
        result = results[0]
        
        boxes = []
        labels = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = result.names[class_id]
            if not should_process_class(class_name, target_mode):
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append([x1, y1, x2, y2])
            labels.append(class_name)
        
        input_boxes = torch.tensor(boxes).to(device) if boxes else torch.zeros((0, 4)).to(device)
        
        if len(boxes) > 0:
            image_predictor.set_image(image_np)
            masks, scores, logits = image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
            
            if masks.ndim == 2:
                masks = masks[None]
                scores = scores[None]
                logits = logits[None]
            elif masks.ndim == 4:
                masks = masks.squeeze(1)
            
            first_mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(device), box_list=input_boxes, label_list=labels)
            
            for i, mask in enumerate(masks):
                objects_count += 1
                object_info = ObjectInfo(instance_id=objects_count)
                object_info.mask = torch.tensor(mask).to(device)
                object_info.class_name = labels[i] if i < len(labels) else None
                if i < len(input_boxes):
                    object_info.box = input_boxes[i]
                if hasattr(object_info, 'update_box'):
                    object_info.update_box()
                global_mask_dict.labels[objects_count] = object_info
            
            frame_object_count[first_frame_idx] = objects_count
            video_predictor.reset_state(inference_state)
            for object_id, object_info in global_mask_dict.labels.items():
                video_predictor.add_new_mask(inference_state, first_frame_idx, object_id, object_info.mask)
            
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
                inference_state, max_frame_num_to_track=len(frame_names) - first_frame_idx
            ):
                image_base_name = frame_names[out_frame_idx].split(".")[0]
                frame_masks = MaskDictionaryModel()
                frame_masks.mask_name = f"mask_{image_base_name}.npy"
                
                if out_mask_logits.shape[-2:]:
                    mask_array = np.zeros(out_mask_logits.shape[-2:], dtype=np.uint16)
                else:
                    img = cv2.imread(os.path.join(frames_dir, frame_names[out_frame_idx]))
                    mask_array = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint16)
                
                for i, out_obj_id in enumerate(out_obj_ids):
                    out_mask = (out_mask_logits[i] > 0.0)
                    if out_mask.sum() == 0:
                        continue
                    
                    object_info = ObjectInfo(instance_id=out_obj_id)
                    object_info.mask = out_mask[0]
                    object_info.class_name = global_mask_dict.labels[out_obj_id].class_name if out_obj_id in global_mask_dict.labels else None
                    if hasattr(object_info, 'update_box'):
                        object_info.update_box()
                    frame_masks.labels[out_obj_id] = object_info
                    mask_array[out_mask[0].cpu().numpy().astype(bool)] = out_obj_id
                
                save_mask_data(mask_array, frame_masks, mask_data_dir, json_data_dir, image_base_name)
            
            del video_predictor
            video_predictor = None
            # Change to Grounded-SAM-2-main directory for config loading
            original_cwd = os.getcwd()
            try:
                os.chdir(GROUNDED_SAM2_BASE)
                model_cfg_rel = os.path.relpath(model_cfg, GROUNDED_SAM2_BASE)
                video_predictor = build_sam2_video_predictor(model_cfg_rel, sam2_checkpoint)
            finally:
                os.chdir(original_cwd)
        else:
            # Create empty mask files if no targets detected in first frame
            for frame_idx in range(len(frame_names)):
                image_base_name = frame_names[frame_idx].split(".")[0]
                img = cv2.imread(os.path.join(frames_dir, frame_names[frame_idx]))
                mask_array = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint16) if img is not None else np.zeros((480, 640), dtype=np.uint16)
                frame_masks = MaskDictionaryModel()
                frame_masks.mask_name = f"mask_{image_base_name}.npy"
                save_mask_data(mask_array, frame_masks, mask_data_dir, json_data_dir, image_base_name)
                
        # Step 2: Process subsequent keyframes, detect new targets and continue tracking
        for start_frame_idx in range(step, len(frame_names), step):
            img_path = os.path.join(frames_dir, frame_names[start_frame_idx])
            image = Image.open(img_path).convert("RGB")
            image_np = np.array(image)
            image_base_name = frame_names[start_frame_idx].split(".")[0]
            
            json_data_path = os.path.join(json_data_dir, f"mask_{image_base_name}.json")
            mask_data_path = os.path.join(mask_data_dir, f"mask_{image_base_name}.npy")
            
            if not os.path.exists(json_data_path) or not os.path.exists(mask_data_path):
                continue
            
            existing_json_data = MaskDictionaryModel().from_json(json_data_path)
            existing_mask_array = np.load(mask_data_path)
            
            results = yolo_model.predict(source=img_path, conf=0.25, verbose=False)
            result = results[0]
            
            boxes = []
            labels = []
            for box in result.boxes:
                class_id = int(box.cls.item())
                class_name = result.names[class_id]
                if not should_process_class(class_name, target_mode):
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append([x1, y1, x2, y2])
                labels.append(class_name)
            
            input_boxes = torch.tensor(boxes).to(device) if boxes else torch.zeros((0, 4)).to(device)
            
            if len(boxes) > 0:
                image_predictor.set_image(image_np)
                masks, scores, logits = image_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )
                
                if masks.ndim == 2:
                    masks = masks[None]
                    scores = scores[None]
                    logits = logits[None]
                elif masks.ndim == 4:
                    masks = masks.squeeze(1)
                
                mask_tensors = [torch.tensor(mask).to(device) for mask in masks]
                filtered_new_masks, filtered_new_boxes, filtered_new_labels, updated_existing_mask_array = filter_masks_by_containment(
                    mask_tensors, input_boxes.tolist(), labels, existing_mask_array, existing_json_data, global_mask_dict
                )
                existing_mask_array = updated_existing_mask_array
                new_masks, new_boxes, new_labels = filtered_new_masks, filtered_new_boxes, filtered_new_labels
                
                if new_masks:
                    video_predictor.reset_state(inference_state)
                    for obj_id in existing_json_data.labels:
                        obj_mask = existing_mask_array == obj_id
                        if np.any(obj_mask):
                            video_predictor.add_new_mask(inference_state, start_frame_idx, obj_id, torch.tensor(obj_mask).to(device))
                    
                    new_object_ids = []
                    for i, new_mask in enumerate(new_masks):
                        objects_count += 1
                        object_info = ObjectInfo(instance_id=objects_count)
                        object_info.mask = new_mask
                        object_info.class_name = new_labels[i] if i < len(new_labels) else None
                        if i < len(new_boxes):
                            object_info.box = new_boxes[i]
                        if hasattr(object_info, 'update_box'):
                            object_info.update_box()
                        
                        global_mask_dict.labels[objects_count] = object_info
                        existing_json_data.labels[objects_count] = object_info
                        existing_mask_array[new_mask.cpu().numpy().astype(bool)] = objects_count
                        video_predictor.add_new_mask(inference_state, start_frame_idx, objects_count, new_mask)
                        new_object_ids.append(objects_count)
                    
                    frame_object_count[start_frame_idx] = objects_count
                    save_mask_data(existing_mask_array, existing_json_data, mask_data_dir, json_data_dir, image_base_name)
                    
                    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
                        inference_state, max_frame_num_to_track=len(frame_names) - start_frame_idx, start_frame_idx=start_frame_idx
                    ):
                        filtered_obj_ids = [obj_id for i, obj_id in enumerate(out_obj_ids) if obj_id in new_object_ids]
                        if not filtered_obj_ids:
                            continue
                        
                        filtered_mask_logits = torch.cat([out_mask_logits[i:i+1] for i, obj_id in enumerate(out_obj_ids) if obj_id in new_object_ids], dim=0)
                        
                        image_base_name = frame_names[out_frame_idx].split(".")[0]
                        existing_json_path = os.path.join(json_data_dir, f"mask_{image_base_name}.json")
                        existing_mask_path = os.path.join(mask_data_dir, f"mask_{image_base_name}.npy")
                        
                        if os.path.exists(existing_json_path) and os.path.exists(existing_mask_path):
                            frame_masks = MaskDictionaryModel().from_json(existing_json_path)
                            mask_array = np.load(existing_mask_path)
                        else:
                            frame_masks = MaskDictionaryModel()
                            frame_masks.mask_name = f"mask_{image_base_name}.npy"
                            if filtered_mask_logits.shape[-2:]:
                                mask_array = np.zeros(filtered_mask_logits.shape[-2:], dtype=np.uint16)
                            else:
                                img = cv2.imread(os.path.join(frames_dir, frame_names[out_frame_idx]))
                                mask_array = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint16)
                        
                        for i, out_obj_id in enumerate(filtered_obj_ids):
                            out_mask = (filtered_mask_logits[i] > 0.0)
                            if out_mask.sum() == 0:
                                continue
                            
                            object_info = ObjectInfo(instance_id=out_obj_id)
                            object_info.mask = out_mask[0]
                            object_info.class_name = global_mask_dict.labels[out_obj_id].class_name if out_obj_id in global_mask_dict.labels else None
                            if hasattr(object_info, 'update_box'):
                                object_info.update_box()
                            frame_masks.labels[out_obj_id] = object_info
                            mask_array[out_mask[0].cpu().numpy().astype(bool)] = out_obj_id
                        
                        save_mask_data(mask_array, frame_masks, mask_data_dir, json_data_dir, image_base_name)
                    
                    del video_predictor
                    video_predictor = None
                    # Change to Grounded-SAM-2-main directory for config loading
                    original_cwd = os.getcwd()
                    try:
                        os.chdir(GROUNDED_SAM2_BASE)
                        model_cfg_rel = os.path.relpath(model_cfg, GROUNDED_SAM2_BASE)
                        video_predictor = build_sam2_video_predictor(model_cfg_rel, sam2_checkpoint)
                    finally:
                        os.chdir(original_cwd)

        # Generate final visualization with all targets and create video
        result_dir_final = os.path.join(video_output_dir, "result_final")
        CommonUtils.creat_dirs(result_dir_final)
        draw_masks_from_rle(frames_dir, mask_data_dir, json_data_dir, result_dir_final, overlay_on_original=True)
        
        # Create final video (all_targets.mp4)
        output_video_path_final = os.path.join(video_output_dir, f"{video_name_without_ext}_all_targets_masks.mp4")
        if os.path.exists(result_dir_final) and len(os.listdir(result_dir_final)) > 0:
            if create_video_from_masks(result_dir_final, output_video_path_final, original_fps):
                print(f"All targets video created: {output_video_path_final}")
        
        # Clean up temporary visualization directory
        if os.path.exists(result_dir_final):
            shutil.rmtree(result_dir_final)

        # Clean up temporary files (keep mask_data_dir, only remove npy files and temporary directories)
        for file in os.listdir(mask_data_dir):
            if file.endswith(".npy"):
                os.remove(os.path.join(mask_data_dir, file))
        for dir_path in [frames_dir, json_data_dir]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)

        return True
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"GPU out of memory error: {video_name} - {str(e)}")
        force_cleanup()
        # Force additional cleanup before exit
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
        except:
            pass
        # Exit program to let bash script restart it
        # Note: failed video will be recorded by bash script detecting the error message
        print(f"Exiting due to GPU OOM to allow memory cleanup...")
        sys.exit(1)
        
    except Exception as e:
        print(f"Error processing video: {video_name} - {str(e)}")
        force_cleanup()
        return False
        
    finally:
        force_cleanup()

def filter_masks_by_containment(new_masks, new_boxes, new_labels, existing_mask_array, existing_json_data, global_mask_dict, containment_threshold=0.6, iou_threshold=0.4):
    """
    Filter masks by IoU or containment relationship, remove smaller targets that are contained or contain other masks
    
    Args:
    new_masks: List of newly detected masks
    new_boxes: Corresponding bounding boxes
    new_labels: Corresponding labels
    existing_mask_array: Existing mask array
    existing_json_data: Existing target data
    global_mask_dict: Global mask dictionary
    containment_threshold: Containment threshold (default 0.6)
    iou_threshold: IoU threshold (default 0.4)
    
    Returns:
    filtered_new_masks: Filtered new masks
    filtered_new_boxes: Filtered new boxes
    filtered_new_labels: Filtered new labels
    updated_existing_mask_array: Updated existing mask array
    """
    
    filtered_new_masks = []
    filtered_new_boxes = []
    filtered_new_labels = []
    updated_existing_mask_array = existing_mask_array.copy()
    
    for i, mask in enumerate(new_masks):
        mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else mask
        should_add_new = True
        existing_to_remove = []
        
        for obj_id in list(existing_json_data.labels.keys()):
            existing_mask = (updated_existing_mask_array == obj_id)
            if existing_mask.sum() == 0:
                continue
            
            intersection = np.logical_and(mask_np, existing_mask)
            intersection_count = intersection.sum()
            
            if intersection_count == 0:
                continue
            
            union = np.logical_or(mask_np, existing_mask).sum()
            iou = intersection_count / union if union > 0 else 0
            
            new_mask_count = mask_np.sum()
            existing_mask_count = existing_mask.sum()
            new_in_existing_ratio = intersection_count / new_mask_count if new_mask_count > 0 else 0
            existing_in_new_ratio = intersection_count / existing_mask_count if existing_mask_count > 0 else 0
            
            has_high_iou = iou >= iou_threshold
            has_containment = (new_in_existing_ratio >= containment_threshold or existing_in_new_ratio >= containment_threshold)
            
            if has_high_iou:
                should_add_new = False
                break
            elif has_containment:
                if new_mask_count > existing_mask_count:
                    existing_to_remove.append(obj_id)
                else:
                    should_add_new = False
                    break
        
        for obj_id in existing_to_remove:
            updated_existing_mask_array[updated_existing_mask_array == obj_id] = 0
            if obj_id in existing_json_data.labels:
                del existing_json_data.labels[obj_id]
            if obj_id in global_mask_dict.labels:
                del global_mask_dict.labels[obj_id]
        
        if should_add_new:
            filtered_new_masks.append(mask)
            if i < len(new_boxes):
                filtered_new_boxes.append(new_boxes[i])
            if i < len(new_labels):
                filtered_new_labels.append(new_labels[i])
    
    return filtered_new_masks, filtered_new_boxes, filtered_new_labels, updated_existing_mask_array

def create_video_from_masks(image_dir, output_video_path, fps=30):
    """Create video from mask image directory"""
    try:
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
        
        if not image_files:
            return False
        
        first_frame = cv2.imread(os.path.join(image_dir, image_files[0]))
        if first_frame is None:
            return False
        
        height, width, _ = first_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            return False
        
        for image_file in image_files:
            frame = cv2.imread(os.path.join(image_dir, image_file))
            if frame is not None:
                out.write(frame)
        
        out.release()
        return True
        
    except Exception as e:
        print(f"Error creating video: {str(e)}")
        return False

def process_video_wrapper(video_name, video_info, config):
    """Wrapper for process_video function"""
    try:
        return process_video(video_name, video_info, config)
    except Exception as e:
        print(f"Error: {e}")
        return False

def cleanup_failed_video_folders(base_output_dir, video_names):
    """
    Clean up video folders that don't have mp4 output files (indicating failed processing)
    
    Args:
    base_output_dir: Base output directory
    video_names: List of all video names to check
    
    Returns:
    Number of cleaned up folders
    """
    cleaned_count = 0
    
    if not os.path.exists(base_output_dir):
        return 0
    
    # Check all existing folders in output directory
    for video_name in video_names:
        video_name_without_ext = os.path.splitext(video_name)[0]
        video_output_dir = os.path.join(base_output_dir, video_name_without_ext)
        
        # Check if folder exists
        if not os.path.exists(video_output_dir) or not os.path.isdir(video_output_dir):
            continue
        
        # Check if mp4 file exists
        expected_mp4 = os.path.join(video_output_dir, f"{video_name_without_ext}_all_targets_masks.mp4")
        
        if not os.path.exists(expected_mp4):
            # No mp4 file found, this is a failed processing, remove the folder
            print(f"Found failed video folder (no mp4): {video_name_without_ext}, removing...")
            try:
                shutil.rmtree(video_output_dir)
                cleaned_count += 1
                print(f"Removed failed folder: {video_output_dir}")
            except Exception as e:
                print(f"Error removing folder {video_output_dir}: {str(e)}")
    
    return cleaned_count

def main():
    parser = argparse.ArgumentParser(description="Single GPU video processing with Grounded-SAM-2 for hcstvg dataset")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start index of videos to process")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="End index of videos to process (exclusive)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of videos to process in one batch")
    parser.add_argument("--target_mode", type=str, 
                        choices=["person", "non-person", "all"],
                        default="non-person",
                        help="Target mode: 'person' (only person), 'non-person' (exclude person), 'all' (all classes)")
    parser.add_argument("--video_base_dir", type=str,
                        default="/data/hcstvg-v2/video/mnt/data1/tzh/HCVG/video_parts",
                        help="Base directory containing input videos")
    parser.add_argument("--base_output_dir", type=str,
                        default="../vipdata/hcstvgv2",
                        help="Base directory for output results")
    parser.add_argument("--hcvg_json_path", type=str,
                        default="../hcstvgv2/anno_v2/train_v2.json",
                        help="Path to HCVG JSON file")
    args = parser.parse_args()
    
    # Convert relative paths to absolute paths based on script location
    if not os.path.isabs(args.base_output_dir):
        args.base_output_dir = os.path.abspath(os.path.join(SCRIPT_DIR, args.base_output_dir))
    if not os.path.isabs(args.hcvg_json_path):
        args.hcvg_json_path = os.path.abspath(os.path.join(SCRIPT_DIR, args.hcvg_json_path))
    
    # Use the GROUNDED_SAM2_BASE defined at module level
    config = {
        "sam2_checkpoint": os.path.join(GROUNDED_SAM2_BASE, "checkpoints", "sam2.1_hiera_large.pt"),
        "model_cfg": os.path.join(GROUNDED_SAM2_BASE, "configs", "sam2.1", "sam2.1_hiera_l.yaml"),
        "grounding_model_id": "IDEA-Research/grounding-dino-tiny",
        "text_prompt": None,
        "target_mode": args.target_mode,
        "video_base_dir": args.video_base_dir,
        "base_output_dir": args.base_output_dir
    }
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        return
    
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Target mode: {config['target_mode']}")
    # Create output directory (will create parent directories if needed)
    CommonUtils.creat_dirs(config["base_output_dir"])
    
    # Load HCVG JSON file
    if not os.path.exists(args.hcvg_json_path):
        print(f"Error: HCVG JSON file not found: {args.hcvg_json_path}")
        return
    
    with open(args.hcvg_json_path, 'r') as f:
        hcvg_data = json.load(f)
    video_names = list(hcvg_data.keys())
    
    # Validate video files exist
    video_names = [v for v in video_names if find_video_path(v, config["video_base_dir"])]
    print(f"Found {len(video_names)} valid video files")
    
    # Clean up failed video folders (folders without mp4 files, indicating OOM or other failures)
    print("Checking for failed video folders (no mp4 files)...")
    cleaned_count = cleanup_failed_video_folders(config["base_output_dir"], video_names)
    if cleaned_count > 0:
        print(f"Cleaned up {cleaned_count} failed video folders")
    
    # Check completed videos (after cleanup, folders without mp4 are removed)
    # A video is considered completed if it has the mp4 output file
    completed_videos = []
    for v in video_names:
        video_name_without_ext = os.path.splitext(v)[0]
        expected_mp4 = os.path.join(config["base_output_dir"], video_name_without_ext, f"{video_name_without_ext}_all_targets_masks.mp4")
        if os.path.exists(expected_mp4):
            completed_videos.append(v)
    video_names = [v for v in video_names if v not in completed_videos]
    print(f"Found {len(completed_videos)} already completed videos, {len(video_names)} videos remaining")
    
    if not video_names:
        print("All videos have been processed, exiting")
        return
    
    if args.end_idx is not None:
        video_names = video_names[args.start_idx:args.end_idx]
    else:
        video_names = video_names[args.start_idx:]
    
    failed_videos_file = os.path.join(config["base_output_dir"], "failed_videos.txt")
    if os.path.exists(failed_videos_file):
        with open(failed_videos_file, 'r') as f:
            failed_videos = [line.strip() for line in f]
        video_names = [v for v in video_names if v not in failed_videos]
    else:
        failed_videos = []

    for video_name in video_names:
        print(f"Processing {video_name} ...")
        video_info = hcvg_data.get(video_name, {})
        try:
            success = process_video(video_name, video_info, config)
            if not success:
                failed_videos.append(video_name)
                with open(failed_videos_file, 'a') as f:
                    f.write(f"{video_name}\n")
        except Exception as e:
            print(f"ERROR: Failed to process video {video_name}: {str(e)}")
            failed_videos.append(video_name)
            with open(failed_videos_file, 'a') as f:
                f.write(f"{video_name}\n")

if __name__ == "__main__":
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    main()
