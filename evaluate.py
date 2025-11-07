from data_configs import DATASETS
import argparse
import numpy as np
import json
from tqdm import tqdm
import os
import re
import pickle
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import random
import pycocotools.mask as mask_util

VIDEO_INFO_CACHE = {}

def get_args():
    parser = argparse.ArgumentParser(description='Evaluation for training-free video temporal grounding (Single GPU Version)')
    parser.add_argument('--dataset', default='charades', type=str, help='Specify the dataset.')
    parser.add_argument('--split', default='default', type=str, help='Specify the split.')
    parser.add_argument("--model_base", type=str, default="/path/to/qwen-model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU device to use")
    parser.add_argument("--bbox_data_file", type=str, default=None, help="Path to JSON file containing GT bbox data")
    parser.add_argument("--mask_data_dir", type=str, default=None, help="Base directory containing mask data")
    return parser.parse_args()

def calc_iou(candidates, gt):
    start, end = candidates[:,0], candidates[:,1]
    s, e = gt[0], gt[1]
    inter = np.minimum(end, e) - np.maximum(start, s)
    union = np.maximum(end, e) - np.minimum(start, s)
    return inter.clip(min=0) / union

def load_mask_data(mask_dir, frame_idx):
    """Load mask data for a specific frame"""
    mask_file = os.path.join(mask_dir, f"mask_{frame_idx}.json")
    if not os.path.exists(mask_file):
        return None
    with open(mask_file, 'r') as f:
        return json.load(f)

def rle_to_bbox(rle_data):
    """Convert RLE format mask to bbox"""
    try:
        mask = mask_util.decode(rle_data)
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return None
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        return [x1, y1, x2, y2]
    except:
        return None

def calculate_bbox_iou(bbox1, bbox2):
    """Calculate IoU between two bboxes"""
    if bbox1 is None or bbox2 is None:
        return 0.0
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    if union == 0:
        return 0.0
    return intersection / union

def find_closest_bbox(mask_dir, target_frame, pred_id, frames_with_bbox):
    """Find the closest frame with the corresponding number bbox and select the bbox with the highest IoU"""
    if not frames_with_bbox:
        return None
    
    # Find the closest frame
    closest_frame = min(frames_with_bbox, key=lambda x: abs(x - target_frame))
    
    # Load mask data for the closest frame
    mask_data = load_mask_data(mask_dir, closest_frame)
    if mask_data is None or str(pred_id) not in mask_data.get('objects', {}):
        return None
    
    # Get the bbox of the target number in the closest frame
    rle_data = mask_data['objects'][str(pred_id)]['rle']
    closest_bbox = rle_to_bbox(rle_data)
    
    # If the target frame also has mask data, compare all bboxes with the closest frame bbox
    target_mask_data = load_mask_data(mask_dir, target_frame)
    if target_mask_data is None:
        return closest_bbox
    
    # Calculate IoU between all bboxes in the target frame and the closest frame bbox, select the maximum
    max_iou = 0.0
    best_bbox = closest_bbox
    
    for obj_id, obj_data in target_mask_data.get('objects', {}).items():
        try:
            current_bbox = rle_to_bbox(obj_data['rle'])
            if current_bbox is not None:
                iou = calculate_bbox_iou(current_bbox, closest_bbox)
                if iou > max_iou:
                    max_iou = iou
                    best_bbox = current_bbox
        except:
            continue
    
    return best_bbox

def find_closest_bbox_from_frame(mask_dir, target_frame, pred_id, last_valid_frame):
    """Find the bbox of the corresponding number in the closest valid frame and select the bbox with the highest IoU"""
    if last_valid_frame is None:
        return None
    
    # Load mask data for the closest frame
    mask_data = load_mask_data(mask_dir, last_valid_frame)
    if mask_data is None or str(pred_id) not in mask_data.get('objects', {}):
        return None
    
    # Get the bbox of the target number in the closest frame
    rle_data = mask_data['objects'][str(pred_id)]['rle']
    closest_bbox = rle_to_bbox(rle_data)
    
    # If the target frame also has mask data, compare all bboxes with the closest frame bbox
    target_mask_data = load_mask_data(mask_dir, target_frame)
    if target_mask_data is None:
        return closest_bbox
    
    # Calculate IoU between all bboxes in the target frame and the closest frame bbox, select the maximum
    max_iou = 0.0
    best_bbox = closest_bbox
    
    for obj_id, obj_data in target_mask_data.get('objects', {}).items():
        try:
            current_bbox = rle_to_bbox(obj_data['rle'])
            if current_bbox is not None:
                iou = calculate_bbox_iou(current_bbox, closest_bbox)
                if iou > max_iou:
                    max_iou = iou
                    best_bbox = current_bbox
        except:
            continue
    
    return best_bbox

def find_largest_bbox(mask_data):
    """Find the largest bbox from mask_data"""
    if mask_data is None or 'objects' not in mask_data:
        return None
    
    max_area = 0
    largest_bbox = None
    
    for obj_id, obj_data in mask_data['objects'].items():
        try:
            rle_data = obj_data['rle']
            current_bbox = rle_to_bbox(rle_data)
            if current_bbox is not None:
                area = (current_bbox[2] - current_bbox[0]) * (current_bbox[3] - current_bbox[1])
                if area > max_area:
                    max_area = area
                    largest_bbox = current_bbox
        except:
            continue
    
    return largest_bbox

def find_bbox_continuation(mask_data, current_bbox):
    """Try to continue the current bbox in the current frame"""
    if mask_data is None or 'objects' not in mask_data:
        return None
    
    # Find the bbox most similar to the current bbox in the current frame
    max_iou = 0.0
    best_bbox = None
    
    for obj_id, obj_data in mask_data['objects'].items():
        try:
            rle_data = obj_data['rle']
            candidate_bbox = rle_to_bbox(rle_data)
            if candidate_bbox is not None:
                iou = calculate_bbox_iou(candidate_bbox, current_bbox)
                if iou > max_iou:
                    max_iou = iou
                    best_bbox = candidate_bbox
        except:
            continue
    
    # If IoU is greater than threshold, consider it can be continued
    if max_iou > 0:  # This threshold can be adjusted
        return best_bbox
    else:
        # If no overlapping bbox is found, select the largest bbox in the current frame as the bbox for this frame
        return find_largest_bbox(mask_data)

def calculate_spatial_iou_for_item(item, pred_id, pred_start, pred_end, gt_id, gt_start, gt_end, 
                                    bbox_data, mask_data_dir, fps):
    """Calculate spatial IoU for a single item"""
    if bbox_data is None or mask_data_dir is None:
        return None
    
    vid = item['vid']
    # Try different video extensions to find corresponding GT data
    possible_extensions = ['.mkv', '.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm']
    sample_data = None
    matched_video_name = None
    
    for ext in possible_extensions:
        full_video_name = vid + ext
        if full_video_name in bbox_data:
            sample_data = bbox_data[full_video_name]
            if 'bbox' in sample_data and sample_data['bbox']:
                matched_video_name = full_video_name
                break
    
    if not matched_video_name and vid in bbox_data:
        sample_data = bbox_data[vid]
        if 'bbox' in sample_data and sample_data['bbox']:
            matched_video_name = vid
    
    if sample_data is None or 'bbox' not in sample_data or not sample_data['bbox']:
        return None
    
    # Convert timestamps to frame indices
    def time_to_frames(time_start, time_end, fps):
        start_frame = int(time_start * fps)
        end_frame = int(time_end * fps)
        return start_frame, end_frame
    
    gt_start_frame, gt_end_frame = time_to_frames(gt_start, gt_end, fps)
    pred_start_frame, pred_end_frame = time_to_frames(pred_start, pred_end, fps)
    
    # Calculate intersection frame range
    overlap_start = max(gt_start_frame, pred_start_frame)
    overlap_end = min(gt_end_frame, pred_end_frame)
    
    # Get mask data directory
    mask_dir = os.path.join(mask_data_dir, vid, 'mask_data')
    if not os.path.exists(mask_dir):
        return None
    
    # Calculate union frame range (for denominator)
    union_start = min(gt_start_frame, pred_start_frame)
    union_end = max(gt_end_frame, pred_end_frame)
    union_frames = union_end - union_start + 1
    
    # If no intersection, return 0.0
    if overlap_start >= overlap_end:
        return 0.0
    
    # First, find all frames with the target number bbox
    frames_with_bbox = []
    for frame_idx in range(overlap_start, overlap_end + 1):
        mask_data = load_mask_data(mask_dir, frame_idx)
        if mask_data is not None and pred_id is not None and str(pred_id) in mask_data.get('objects', {}):
            frames_with_bbox.append(frame_idx)
    
    # Calculate IoU for each frame in intersection range and sum
    # Then divide by union frame count (consistent with hcstvgv1_9_3.py)
    total_iou = 0.0
    
    # Record current continuing bbox information and the most recent valid frame
    current_bbox = None
    last_valid_frame = None
    
    for frame_idx in range(overlap_start, overlap_end + 1):
        mask_data = load_mask_data(mask_dir, frame_idx)
        if mask_data is None:
            continue  # Skip frames without mask data (consistent with hcstvgv1_9_3.py)
        
        # Get predicted bbox
        pred_bbox = None
        if pred_id is not None and str(pred_id) in mask_data.get('objects', {}):
            # Current frame has bbox for this number
            rle_data = mask_data['objects'][str(pred_id)]['rle']
            pred_bbox = rle_to_bbox(rle_data)
            current_bbox = pred_bbox  # Update current continuing bbox
            last_valid_frame = frame_idx  # Update most recent valid frame
        else:
            # Current frame does not have bbox for this number
            if current_bbox is not None:
                # Try to continue the current bbox
                pred_bbox = find_bbox_continuation(mask_data, current_bbox)
                if pred_bbox is not None:
                    # Continuation successful
                    pass
                else:
                    # Continuation failed, compare with the most recent valid frame
                    pred_bbox = find_closest_bbox_from_frame(mask_dir, frame_idx, pred_id, last_valid_frame)
                    if pred_bbox is not None:
                        current_bbox = pred_bbox  # Update current continuing bbox
            else:
                # No bbox to continue, find the closest bbox
                pred_bbox = find_closest_bbox(mask_dir, frame_idx, pred_id, frames_with_bbox)
                if pred_bbox is not None:
                    current_bbox = pred_bbox  # Update current continuing bbox
                    # Find the corresponding frame as the most recent valid frame
                    for valid_frame in frames_with_bbox:
                        if valid_frame <= frame_idx:
                            last_valid_frame = valid_frame
                        else:
                            break
            
            # If still no bbox found, select the largest bbox in the current frame
            if pred_bbox is None:
                pred_bbox = find_largest_bbox(mask_data)
        
        # Get GT bbox
        gt_bbox = None
        bbox_list = sample_data.get('bbox', [])
        st_frame = sample_data.get('st_frame', 0)
        frame_in_sample = frame_idx - st_frame
        if 0 <= frame_in_sample < len(bbox_list):
            bbox = bbox_list[frame_in_sample]
            if bbox and len(bbox) >= 4:
                x1, y1, w, h = bbox[:4]
                gt_bbox = [x1, y1, x1 + w, y1 + h]
        
        # Calculate IoU for current frame
        frame_iou = calculate_bbox_iou(pred_bbox, gt_bbox)
        
        total_iou += frame_iou
    
    if union_frames == 0:
        return 0.0
    
    # Divide by union frame count (consistent with hcstvgv1_9_3.py)
    return total_iou / union_frames

def save_prediction_results(results_path, predictions):
    with open(results_path, 'w', encoding='utf-8') as f:
        for vid_sent_id, pred_info in sorted(predictions.items()):
            f.write(f"ID: {vid_sent_id}\n")
            f.write(f"Query: {pred_info['query']}\n")
            f.write(f"Ground Truth: {pred_info['gt_id']}, {pred_info['gt_timestamp']}\n")
            f.write(f"Prediction:  {pred_info['pred_id']}, {pred_info['pred_timestamp']}\n")
            f.write(f"IoU: {pred_info['iou']:.4f}\n")
            f.write(f"Marker Value: {pred_info['markervalue']}\n")
            f.write(f"Model Output: {pred_info['model_output']}\n")
            f.write("-" * 80 + "\n")
            
def cached_process_vision_info(messages, return_video_kwargs=False):
    global VIDEO_INFO_CACHE
    
    video_path = None
    for msg in messages:
        for content in msg.get('content', []):
            if isinstance(content, dict) and 'video' in content:
                video_path = content['video']
                break
    
    cache_key = f"{video_path}_{return_video_kwargs}"
    if cache_key in VIDEO_INFO_CACHE:
        return VIDEO_INFO_CACHE[cache_key]
    
    result = process_vision_info(messages, return_video_kwargs=return_video_kwargs)
    VIDEO_INFO_CACHE[cache_key] = result
    
    return result

def inference(video_path, prompt, model, processor, max_new_tokens=2048, device="cuda:0"):
    messages = [
        {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"video": video_path, 
                "total_pixels": 2048 * 28 * 28, 
                "min_pixels": 16 * 28 * 28,
                },
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    image_inputs, video_inputs, video_kwargs = cached_process_vision_info(messages, return_video_kwargs=True)
    fps_inputs = video_kwargs['fps']
    
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, fps=fps_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    generated_ids = [output_ids[i][len(inputs.input_ids[i]):] for i in range(len(output_ids))]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]

def parse_timestamp_output(output_string):
    """
    Parses Target ID, start time, and end time from model output.
    Returns: (target_id, start_time, end_time) or (None, None, None) if parsing fails.
    """
    # 1. Extract the last <answer> block
    answer_matches = re.findall(r"<answer>(.*?)</answer>", output_string, re.DOTALL)
    if not answer_matches:
        print("Error: No <answer> tag found.")
        return None, None, None
    last_answer = answer_matches[-1].strip()
    print(f"Raw answer content: {last_answer}")  # Debug log
    # 2. Parse Target ID (single ID only)
    id_match = re.search(
        r"Target ID:\s*(\d+)", 
        last_answer, 
        re.IGNORECASE
    )
    target_id = int(id_match.group(1)) if id_match else None
    # 3. Parse Time range (supports "to" or "and" as separators)
    time_match = re.search(
        r"Time range:\s*(\d+\.?\d*)\s*(?:to|and)\s*(\d+\.?\d*)", 
        last_answer, 
        re.IGNORECASE
    )
    if time_match:
        start_time = float(time_match.group(1))
        end_time = float(time_match.group(2))
    else:
        start_time = end_time = None
    # 4. Validation
    if None in (target_id, start_time, end_time):
        print(f"Failed to parse: ID={target_id}, Time={start_time}-{end_time}")
        return None, None, None
    return target_id, start_time, end_time

GROUND_TEMPLATE = """Each object in the video is marked with a red number at its center, representing its object ID. To accurately pinpoint the event "[EVENT]" in the video:

1.Determine the precise time period of the event occurs.

2.Identify which object ID is corresponding to the described event.

Output your thought process within the <think> </think> tags, including both a temporal analysis (e.g., specific timestamps "xx.xx" or time ranges "xx.xx to xx.xx") and a spatial analysis (e.g., object ID "xx").

Then, provide the start and end times (in seconds, precise to two decimal places), and object ID in the format "Target ID: [ID], Time range: [start time to end time]" within the <answer> </answer> tags. For example: "Target ID: 1, Time range: 12.54 to 17.83"."""

def create_work_items(data):
    work_items = []
    for vid, ann in data.items():
        for i in range(len(ann['sentences'])):
            work_items.append({
                'vid': vid,
                'ann': ann,
                'sentence_idx': i
            })
    random.shuffle(work_items)
    return work_items

def setup_model(model_base, device):
    print(f"Setting up model on device {device}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_base,
        torch_dtype=torch.bfloat16,
        use_sliding_window=True,
        attn_implementation="flash_attention_2",
        device_map=device
    )
    processor = AutoProcessor.from_pretrained(model_base)
    return model, processor

def get_checkpoint_path(checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(checkpoint_dir, "checkpoint.pkl")

def get_results_path(checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(checkpoint_dir, "prediction.txt")

def load_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    return {'processed_items': set(), 'ious': [], 'temporal_recall': np.array([0, 0, 0]), 'spatial_ious': [], 'visual_recall': np.array([0, 0, 0])}

def save_checkpoint(checkpoint_path, state):
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(state, f)

def process_work_items(work_items, video_dir_path, model_base, device, checkpoint_dir, resume=False, 
                       bbox_data=None, mask_data_dir=None):
    ious = []
    spatial_ious = []
    predictions = {}
    thresh = np.array([0.3, 0.5, 0.7])
    temporal_recall = np.array([0, 0, 0])
    visual_recall = np.array([0, 0, 0])
    
    checkpoint_path = get_checkpoint_path(checkpoint_dir)
    results_path = get_results_path(checkpoint_dir)
    processed_items = set()
    
    if resume and os.path.exists(checkpoint_path):
        checkpoint = load_checkpoint(checkpoint_path)
        processed_items = checkpoint['processed_items']
        ious = checkpoint['ious']
        temporal_recall = checkpoint.get('temporal_recall', checkpoint.get('recall', np.array([0, 0, 0])))
        predictions = checkpoint.get('predictions', {})
        spatial_ious = checkpoint.get('spatial_ious', [])
        visual_recall = checkpoint.get('visual_recall', np.array([0, 0, 0]))
        # Ensure spatial_ious has the same length as ious
        if len(spatial_ious) < len(ious):
            spatial_ious.extend([0.0] * (len(ious) - len(spatial_ious)))
        print(f"Resuming from checkpoint with {len(processed_items)} processed items")

    model, processor = setup_model(model_base, device)
    
    item_ids = [f"{item['vid']}_{item['sentence_idx']}" for item in work_items]
    remaining_items = [(i, item) for i, (item, item_id) in enumerate(zip(work_items, item_ids)) 
                      if not resume or item_id not in processed_items]
    
    if not remaining_items:
        print("All items already processed")
        return ious, temporal_recall, spatial_ious, visual_recall
    
    print(f"Processing {len(remaining_items)} out of {len(work_items)} items")
    
    pbar = tqdm(remaining_items)
    for idx, (_, item) in enumerate(pbar):
        vid = item['vid']
        ann = item['ann']
        sentence_idx = item['sentence_idx']
        item_id = f"{vid}_{sentence_idx}"
        
        prompt = GROUND_TEMPLATE.replace('[EVENT]', ann['sentences'][sentence_idx])
        
        duration = ann['duration'] if 'duration' in ann else ann['video_duration']
        video_path = None
        for ext in ['mp4', 'mkv', 'webm']:
            path = os.path.join(video_dir_path, f"{vid}.{ext}")
            if os.path.isfile(path):
                video_path = path
                break
          
        if video_path:
            try:
                ans = inference(video_path, prompt, model, processor, device=device)
                # print('prompt', prompt)
                # print('ans', ans)
                pred_id, sp, ep = parse_timestamp_output(ans)
                print(f"Predicted: Target ID: {pred_id}, Time range: {sp} to {ep}")
                
                print(f"Ground truth: {ann['timestamps'][sentence_idx]}")
                print('-' * 50)
                
                query = ann['sentences'][sentence_idx]
                gt_timestamp = ann['timestamps'][sentence_idx]

                
                if (sp is not None) and (ep is not None):
                    s, e = ann['timestamps'][sentence_idx]
                    gt_id = ann['objectid'][sentence_idx]
                    iou_ = (min(e, ep) - max(s, sp)) / (max(e, ep) - min(s, sp))
                    ious.append(max(iou_, 0))
                    temporal_recall += (thresh <= iou_)
                    
                    if pred_id == gt_id:
                        markervalue = 1
                    else:
                        markervalue = 0
                    
                    # Calculate spatial IoU
                    spatial_iou = 0.0
                    if bbox_data is not None and mask_data_dir is not None:
                        fps = ann.get('fps', 25.0)
                        spatial_iou = calculate_spatial_iou_for_item(
                            item, pred_id, sp, ep, gt_id, s, e, 
                            bbox_data, mask_data_dir, fps
                        )
                        if spatial_iou is None:
                            spatial_iou = 0.0
                    
                    spatial_ious.append(spatial_iou)
                    visual_recall += (thresh <= spatial_iou)
                else:
                    markervalue = 0
                    ious.append(0)
                    spatial_ious.append(0.0)
                    visual_recall += (thresh <= 0.0)

                predictions[item_id] = {
                    'query': query,
                    'gt_timestamp': gt_timestamp,
                    'gt_id': gt_id,
                    'pred_timestamp': [sp, ep] if sp is not None and ep is not None else [0, 0],
                    'pred_id': pred_id,
                    'iou': iou_,
                    'model_output': ans,
                    'markervalue': markervalue
                }

                processed_items.add(item_id)
                
                if (idx + 1) % 5 == 0 or idx == len(remaining_items) - 1:
                    state = {
                        'processed_items': processed_items,
                        'ious': ious,
                        'temporal_recall': temporal_recall,
                        'predictions': predictions,
                        'spatial_ious': spatial_ious,
                        'visual_recall': visual_recall
                    }
                    save_checkpoint(checkpoint_path, state)
                    save_prediction_results(results_path, predictions)
                    
                miou = sum(ious) / len(ious) if ious else 0
                mean_spatial_iou = sum(spatial_ious) / len(spatial_ious) if spatial_ious else 0.0
                temporal_recall_str = str(temporal_recall / len(ious) if ious else [0, 0, 0])
                visual_recall_str = str(visual_recall / len(spatial_ious) if spatial_ious else [0, 0, 0])
                pbar.set_postfix({"m_tIoU": f"{miou:.4f}", "m_vIoU": f"{mean_spatial_iou:.4f}", 
                                 "tIoU@0.5": f"{temporal_recall[1]/len(ious) if ious else 0:.4f}",
                                 "vIoU@0.5": f"{visual_recall[1]/len(spatial_ious) if spatial_ious else 0:.4f}"})
                
            except Exception as e:
                print(f"Error processing {vid}_{sentence_idx}: {e}")
    
    print('=== final result ===')
    if ious:
        m_tiou = sum(ious) / len(ious)
        print(f'm_tIoU: {m_tiou:.4f}')
        for th, r in zip(thresh, temporal_recall):
            print(f'tIoU@{th}: {r / len(ious):.4f}')
    
    if spatial_ious:
        m_viou = sum(spatial_ious) / len(spatial_ious)
        print(f'm_vIoU: {m_viou:.4f}')
        for th, r in zip(thresh, visual_recall):
            print(f'vIoU@{th}: {r / len(spatial_ious):.4f}')
    
    save_prediction_results(results_path, predictions)
                
    return ious, temporal_recall, spatial_ious, visual_recall

def evaluate(data, args):
    dataset = DATASETS[args.dataset]
    video_dir_path = dataset['video_path']
    
    # Load bbox data (if provided)
    bbox_data = None
    if args.bbox_data_file and os.path.exists(args.bbox_data_file):
        print(f"Loading bbox data from {args.bbox_data_file}...")
        with open(args.bbox_data_file, 'r', encoding='utf-8') as f:
            bbox_data = json.load(f)
        print(f"Loaded bbox data for {len(bbox_data)} videos")
    else:
        print("No bbox data file provided, spatial IoU will not be calculated")
    
    work_items = create_work_items(data)
    
    ious, temporal_recall, spatial_ious, visual_recall = process_work_items(
        work_items, 
        video_dir_path, 
        args.model_base, 
        args.device, 
        args.checkpoint_dir,
        args.resume,
        bbox_data=bbox_data,
        mask_data_dir=args.mask_data_dir
    )
    
    return ious, temporal_recall, spatial_ious, visual_recall

if __name__=='__main__':
    args = get_args()
    assert args.dataset in DATASETS
    dataset = DATASETS[args.dataset]
    assert args.split in dataset['splits']
    
    print('evaluate', args.dataset, args.split)
    
    # load data
    with open(dataset['splits'][args.split]['annotation_file']) as f:
        data = json.load(f)

    evaluate(data, args)