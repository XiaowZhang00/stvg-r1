import os
import cv2
import json
import numpy as np
import argparse
from pycocotools import mask as maskUtils
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def load_json(file_path):
    """Load JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data, file_path):
    """Save JSON file"""
    with open(file_path, 'w') as f:
        json.dump(data, f)


def count_mask_pixels(rle):
    """Count mask pixel count"""
    # Ensure RLE format is correct
    if isinstance(rle['counts'], str):
        rle = rle.copy()
        rle['counts'] = rle['counts'].encode('ascii')
    
    mask = maskUtils.decode(rle)
    return np.sum(mask)


def filter_masks_by_size(input_dir, output_dir, size_ratio_threshold=1/3):
    """
    Filter masks by size
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        size_ratio_threshold: Minimum size ratio to keep (relative to max mask)
    """
    # Get all JSON files
    json_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.json')])
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Statistics
    total_frames = len(json_files)
    total_masks_before = 0
    total_masks_after = 0
    filtered_frames_count = 0
    
    for json_file in tqdm(json_files, desc="Filtering frames", leave=False):
        data = load_json(os.path.join(input_dir, json_file))
        
        objects = data.get('objects', {})
        if not objects:
            # If no objects, save original file
            save_json(data, os.path.join(output_dir, json_file))
            continue
        
        # Calculate pixel count for each mask
        mask_sizes = {}
        for obj_id, obj_info in objects.items():
            if obj_info is None:
                mask_sizes[obj_id] = 0
            else:
                rle = obj_info['rle']
                pixel_count = count_mask_pixels(rle)
                mask_sizes[obj_id] = pixel_count
        
        total_masks_before += len(objects)
        
        # Find mask with maximum pixel count
        if not mask_sizes or all(size == 0 for size in mask_sizes.values()):
            # If all masks are empty, keep original data
            save_json(data, os.path.join(output_dir, json_file))
            total_masks_after += len(objects)
            continue
        
        max_size = max(mask_sizes.values())
        threshold_size = max_size * size_ratio_threshold
        
        # Filter masks
        filtered_objects = {}
        for obj_id, obj_info in objects.items():
            if mask_sizes[obj_id] >= threshold_size:
                filtered_objects[obj_id] = obj_info
        
        total_masks_after += len(filtered_objects)
        
        # If masks were filtered, record this frame
        if len(filtered_objects) < len(objects):
            filtered_frames_count += 1
        
        # Update data and save
        data['objects'] = filtered_objects
        save_json(data, os.path.join(output_dir, json_file))


def filter_single_video(args):
    """Filter masks for a single video folder"""
    video_name, input_root, size_ratio_threshold = args
    video_dir = os.path.join(input_root, video_name)
    input_dir = os.path.join(video_dir, "mask_data")
    if not os.path.exists(input_dir):
        return f"Skip {video_name}: mask_data folder not found"
    
    output_dir = os.path.join(video_dir, "mask_data_filtered_1_3")
    
    # Execute filtering
    try:
        filter_masks_by_size(input_dir, output_dir, size_ratio_threshold)
        return f"Filtered: {video_name}"
    except Exception as e:
        return f"Error filtering {video_name}: {str(e)}"


def get_video_path(video_name, video_root):
    """Find video file in the new folder structure"""
    # Iterate through folders 0-9
    for folder_num in range(10):
        folder_path = os.path.join(video_root, str(folder_num))
        if not os.path.exists(folder_path):
            continue
            
        # Find matching video file in current folder
        for f in os.listdir(folder_path):
            if f.startswith(video_name):
                return os.path.join(folder_path, f)
    return None

# Get mask center: middle y value and corresponding x middle value
def get_mask_center(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    # Get middle y value
    cy = int((ys.min() + ys.max()) / 2)
    # Find all x values corresponding to this y value
    x_at_midy = xs[ys == cy]
    if len(x_at_midy) == 0:  # If middle y value has no corresponding x
        cx = int((xs.min() + xs.max()) / 2)  # Fall back to x middle value
    else:
        cx = int(x_at_midy.mean())  # Get average x value for this row
    return (cx, cy)

def process_single_video(args):
    """Process single video"""
    video_name, input_root, video_root, output_root = args
    video_dir = os.path.join(input_root, video_name)
    mask_data_dir = os.path.join(video_dir, "mask_data_filtered_1_3")
    if not os.path.isdir(mask_data_dir):
        return f"Skip {video_name}: mask_data_filtered_1_3 folder not found"

    video_path = get_video_path(video_name, video_root)
    if video_path is None:
        return f"Skip {video_name}: video file not found"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return f"Skip {video_name}: cannot open video"

    # Get resolution and fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(output_root, f"{video_name}.mp4")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    maxlength = max(height, width)

    # Linearly scale font size
    target_fontsize = 20
    base_res = 336
    base_font_height = 30  # Empirical value, height when cv2.putText(fontScale=1)
    font_scale = (target_fontsize / base_font_height) * (maxlength / base_res)
    thickness = max(2, int(font_scale * 2))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        mask_json_path = os.path.join(mask_data_dir, f"mask_{frame_idx}.json")
        if os.path.exists(mask_json_path):
            with open(mask_json_path, "r") as f:
                mask_data = json.load(f)
            for marker_id, obj in mask_data.get("objects", {}).items():
                rle = obj["rle"]
                mask = maskUtils.decode(rle)
                center = get_mask_center(mask)
                if center is not None:
                    # Draw marker ID in red (BGR format: (0, 0, 255))
                    cv2.putText(
                        frame, str(marker_id), center,
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness, cv2.LINE_AA
                    )
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    return f"Saved: {out_path}"

def main():
    parser = argparse.ArgumentParser(description="Filter masks and generate marker videos")
    parser.add_argument("--input_root", type=str,
                        default="../vipdata/hcstvgv2",
                        help="Root directory containing video folders with mask_data")
    parser.add_argument("--video_root", type=str,
                        default="/data/hcstvg-v2/video/mnt/data1/tzh/HCVG/video_parts",
                        help="Root directory containing original video files")
    parser.add_argument("--output_root", type=str,
                        default="../vipdata/hcstvgv2video",
                        help="Output directory for marker videos")
    parser.add_argument("--size_ratio_threshold", type=float,
                        default=1/3,
                        help="Filter threshold: keep masks with pixel count >= threshold * max_mask_size (default: 1/3)")
    parser.add_argument("--num_workers", type=int,
                        default=None,
                        help="Number of parallel workers (default: min(8, cpu_count()))")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_root, exist_ok=True)
    
    # Get all videos to process
    video_names = []
    for video_name in os.listdir(args.input_root):
        video_dir = os.path.join(args.input_root, video_name)
        mask_data_dir = os.path.join(video_dir, "mask_data")
        if os.path.isdir(mask_data_dir):
            video_names.append(video_name)
    
    total_videos = len(video_names)
    print(f"Found {total_videos} videos to process")
    print(f"Input root: {args.input_root}")
    print(f"Video root: {args.video_root}")
    print(f"Output root: {args.output_root}")
    print(f"Size ratio threshold: {args.size_ratio_threshold}")
    
    num_workers = args.num_workers if args.num_workers is not None else min(8, cpu_count())
    print(f"Number of workers: {num_workers}")
    
    # Step 1: Filter masks by size
    print(f"\nStep 1: Filtering masks (size ratio threshold: {args.size_ratio_threshold})...")
    filter_args = [(video_name, args.input_root, args.size_ratio_threshold) for video_name in video_names]
    with Pool(num_workers) as pool:
        for i, result in enumerate(tqdm(pool.imap(filter_single_video, filter_args), total=total_videos, desc="Filtering masks")):
            if result:
                print(f"Filter progress: {i+1}/{total_videos} - {result}")
    
    # Step 2: Generate marker videos from filtered masks
    print(f"\nStep 2: Generating marker videos from filtered masks...")
    video_args = [(video_name, args.input_root, args.video_root, args.output_root) for video_name in video_names]
    with Pool(num_workers) as pool:
        for i, result in enumerate(tqdm(pool.imap(process_single_video, video_args), total=total_videos, desc="Processing videos")):
            if result:
                print(f"Video progress: {i+1}/{total_videos} - {result}")

if __name__ == "__main__":
    main()
