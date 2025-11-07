import json
import os
import argparse
import cv2
import numpy as np
from pycocotools import mask as maskUtils
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from collections import Counter


def normalize_video_id(video_id):
    """Normalize video ID by removing file extension and any whitespace"""
    base_name = video_id.strip()
    if '.' in base_name:
        parts = base_name.rsplit('.', 1)
        if len(parts) == 2:
            base_name = parts[0]
    return base_name


def mask_to_bbox(mask):
    """Convert mask to bounding box"""
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    return [int(x1), int(y1), int(x2), int(y2)]


def compute_iou(boxA, boxB):
    """Compute IoU between two bounding boxes"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA + 1)
    interH = max(0, yB - yA + 1)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0.0
    return iou


def xywh_to_xyxy(bbox):
    """Convert bbox from xywh to xyxy format"""
    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h
    return [x1, y1, x2, y2]


def get_video_info(video_path):
    """Get video duration and fps"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else None
    
    cap.release()
    return duration, fps


def find_video_file(video_id, video_dir):
    """Find video file in directory (search recursively in subfolders)"""
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.webm', '.m4v']
    
    # First try direct match in video_dir
    for ext in video_extensions:
        video_path = os.path.join(video_dir, f"{video_id}{ext}")
        if os.path.exists(video_path):
            return video_path
    
    # Try in subfolders 0-9
    for folder_num in range(10):
        folder_path = os.path.join(video_dir, str(folder_num))
        if not os.path.exists(folder_path):
            continue
        
        for ext in video_extensions:
            video_path = os.path.join(folder_path, f"{video_id}{ext}")
            if os.path.exists(video_path):
                return video_path
        
        try:
            for f in os.listdir(folder_path):
                if f.startswith(video_id):
                    file_ext = os.path.splitext(f)[1].lower()
                    if file_ext in video_extensions:
                        return os.path.join(folder_path, f)
        except PermissionError:
            continue
    
    # Recursive search (fallback)
    try:
        for root, dirs, files in os.walk(video_dir):
            for f in files:
                if f.startswith(video_id):
                    file_ext = os.path.splitext(f)[1].lower()
                    if file_ext in video_extensions:
                        return os.path.join(root, f)
    except Exception as e:
        print(f"Warning: Error during recursive search: {e}")
    
    return None


def compute_best_marker_id(video_name, info, pred_root):
    """Compute best marker ID for a video by matching GT bbox with predicted masks"""
    pred_video_dir = os.path.join(pred_root, video_name)
    mask_data_dir = os.path.join(pred_video_dir, "mask_data_filtered_1_3")
    
    if not os.path.exists(mask_data_dir):
        return None
    
    st_frame = int(info.get("st_frame", 0))
    gt_bboxes = info.get("bbox", [])
    
    if not gt_bboxes:
        return None
    
    # Collect all best marker IDs from all frames
    marker_ids = []
    
    # Iterate through frames based on GT bbox information
    for gt_idx, gt_bbox_raw in enumerate(gt_bboxes):
        frame_idx = st_frame + gt_idx
        
        # Get GT bbox
        gt_bbox = xywh_to_xyxy(gt_bbox_raw)
        
        # Find pred best bbox
        mask_json_path = os.path.join(mask_data_dir, f"mask_{frame_idx}.json")
        best_iou = 0.0
        best_marker = None
        
        if os.path.exists(mask_json_path):
            with open(mask_json_path, "r") as f:
                mask_data = json.load(f)
            for marker_id, obj in mask_data.get("objects", {}).items():
                rle = obj["rle"]
                mask = maskUtils.decode(rle)
                pred_bbox = mask_to_bbox(mask)
                if pred_bbox is None:
                    continue
                iou = compute_iou(gt_bbox, pred_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_marker = marker_id
        
        # Collect marker ID if found
        if best_marker is not None:
            marker_ids.append(best_marker)
    
    # Find most common marker ID
    if not marker_ids:
        return None
    else:
        return Counter(marker_ids).most_common(1)[0][0]


def process_single_video(args):
    """Process single video: compute best marker ID and prepare output data"""
    video_file, info, pred_root, video_dir = args
    
    video_name = normalize_video_id(video_file)
    
    # Compute best marker ID
    best_marker_id = compute_best_marker_id(video_name, info, pred_root)
    
    if best_marker_id is None:
        return (video_name, "Skip: no marker found", None)
    
    # Get original data
    sample_data = info
    
    # Get duration and fps from video file
    duration = None
    fps = None
    if video_dir:
        video_path = find_video_file(video_name, video_dir)
        if video_path:
            duration, fps = get_video_info(video_path)
    
    # Get required fields from original data
    img_num = sample_data.get('img_num', 0)
    st_frame = sample_data.get('st_frame', 0)
    st_time = sample_data.get('st_time', 0)
    ed_time = sample_data.get('ed_time', 0)
    
    # If duration not found from video, calculate from img_num and fps
    if duration is None or duration <= 0:
        # Try to get fps first
        if fps is None or fps <= 0:
            # Try to estimate fps from existing data
            gt_duration = ed_time - st_time if ed_time > st_time else 0
            bbox_count = len(sample_data.get('bbox', []))
            if gt_duration > 0 and bbox_count > 0:
                fps = bbox_count / gt_duration
        
        # Calculate duration from img_num and fps
        if fps and fps > 0:
            duration = img_num / fps
        else:
            # Last resort: use a default fps to estimate
            fps = 25.0
            duration = img_num / fps if img_num > 0 else 0
    
    # If fps not found from video, calculate from img_num and duration
    if fps is None or fps <= 0:
        fps = img_num / duration if duration > 0 else 25.0
    
    # Add timestamps
    if st_time is not None and ed_time is not None:
        timestamps = [[st_time, ed_time]]
    else:
        timestamps = []
    
    # Add sentences
    sentences = []
    if 'English' in sample_data:
        sentences.append(sample_data['English'])
    
    # Create new data with only required fields
    new_sample_data = {
        'duration': duration,
        'timestamps': timestamps,
        'sentences': sentences,
        'objectid': [int(best_marker_id)],
        'fps': fps,
        'img_num': img_num,
        'st_frame': st_frame,
        'st_time': st_time
    }
    
    return (video_name, "OK", new_sample_data)


def main():
    parser = argparse.ArgumentParser(description="Compute best marker ID and generate JSON file")
    parser.add_argument("--gt_json", type=str,
                        default="../hcstvgv2/anno_v2/train_v2.json",
                        help="Path to ground truth JSON file")
    parser.add_argument("--pred_root", type=str,
                        default="../vipdata/hcstvgv2",
                        help="Root directory containing prediction results")
    parser.add_argument("--video_dir", type=str,
                        default="/data/hcstvg-v2/video/mnt/data1/tzh/HCVG/video_parts",
                        help="Directory containing video files (for getting duration and fps)")
    parser.add_argument("--output_json", type=str,
                        default="../hcstvgv2/anno_v2/train_marker_v2.json",
                        help="Path to output JSON file")
    parser.add_argument("--num_workers", type=int,
                        default=None,
                        help="Number of parallel workers (default: min(cpu_count(), 8))")
    
    args = parser.parse_args()
    
    # Load GT data
    print("Loading GT JSON file...")
    with open(args.gt_json, 'r') as f:
        gt_data = json.load(f)
    
    print(f"Loaded {len(gt_data)} videos from GT JSON")
    print(f"GT JSON: {args.gt_json}")
    print(f"Pred root: {args.pred_root}")
    print(f"Video dir: {args.video_dir}")
    print(f"Output JSON: {args.output_json}")
    
    # Prepare arguments list
    args_list = []
    for video_file, info in gt_data.items():
        args_list.append((video_file, info, args.pred_root, args.video_dir))
    
    num_workers = args.num_workers if args.num_workers is not None else min(cpu_count(), 8)
    print(f"Number of workers: {num_workers}")
    
    # Process videos with multiprocessing
    print("\nProcessing videos...")
    results = []
    with Pool(processes=num_workers) as pool:
        for result in tqdm(pool.imap(process_single_video, args_list), total=len(args_list), desc="Processing videos"):
            results.append(result)
    
    # Create output data
    new_data = {}
    processed_count = 0
    skipped_count = 0
    
    for video_name, status, sample_data in results:
        if status == "OK" and sample_data is not None:
            new_data[video_name] = sample_data
            processed_count += 1
        else:
            skipped_count += 1
            if status != "OK":
                print(f"Skipped {video_name}: {status}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_json)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save processed JSON file
    print(f"\nSaving processed JSON file to: {args.output_json}")
    with open(args.output_json, 'w') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessing complete!")
    print(f"- Successfully processed and kept: {processed_count} samples")
    print(f"- Skipped samples: {skipped_count} samples")
    print(f"- Final count: {len(new_data)} samples")
    print(f"- Output file: {args.output_json}")


if __name__ == "__main__":
    main()
