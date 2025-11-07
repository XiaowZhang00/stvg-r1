# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from src.open_r1.trainer import Qwen2VLGRPOTrainer_Video as Qwen2VLGRPOTrainer
from src.open_r1.trainer import Qwen2VLGRPOVLLMTrainer_Video as Qwen2VLGRPOVLLMTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from tqdm import tqdm
import torch
import json
import random
import torch.serialization
import numpy.core.multiarray
torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'iou', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["iou", "format"],
        metadata={"help": "List of reward functions. Possible values: 'iou', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )

    train_data_path: str = field(
        default="/share/wy/Video/Charades/charades_annotation/train.json",
        metadata={"help": "Path to the training data JSON file."},
    )
    eval_data_path: str = field(
        default="/share/wy/Video/Charades/charades_annotation/val.json",
        metadata={"help": "Path to the evaluation data JSON file."},
    )

    video_folder: str = field(
        default="/share/wy/Video/Charades/Charades_v1",  # Replace with your actual video folder path
        metadata={"help": "Path to the folder containing video files."},
    )
    preprocessed_data_path: Optional[str] = field( # Add preprocessed_data_path argument
        default="",
        metadata={"help": "Path to the preprocessed dataset directory. If provided, load preprocessed data instead of raw videos."},
    )


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

def iou_timestamp_reward(completions, solution, durations, **kwargs):
    """
    Combined reward function for spatial-temporal localization.
    Args:
        completions: List of model outputs (strings)
        solutions: List of tuples (gt_start, gt_end, gt_id)
        durations: List of video durations (unused in current implementation)
    Returns:
        List of combined rewards (time_iou + spatial_reward)
    """
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
    # 获取当前训练步数（如果可用）
    current_step = kwargs.get('current_step', 0)
    if isinstance(current_step, list):
        current_step = current_step[0] if current_step else 0
    
    for content, sol, duration in zip(completions, solution, durations):
        # 1. Parse model output
        pred_id, pred_start, pred_end = parse_timestamp_output(content)
        gt_id = sol[-3]
        gt_start = sol[-2]
        gt_end = sol[-1]
        n_markers = (len(sol) - 3) // 2

        # 2. Calculate Time IoU
        time_reward = 0.0
        if None not in (pred_start, pred_end):
            intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
            union = max(pred_end, gt_end) - min(pred_start, gt_start)
            time_reward = intersection / union if union > 0 else 0.0
            # # Use the timereward proposed by Time-R1
            # video_duration = duration
            # start_diff = abs(pred_start - gt_start) / video_duration
            # end_diff = abs(pred_end - gt_end) / video_duration
            # time_reward = time_reward * (1 - start_diff) * (1 - end_diff)
        
        # # 3. Calculate Spatial Reward (ID match)
        spatial_reward = 1.0 if (pred_id == gt_id) else 0.0

        combined_reward = time_reward + spatial_reward
        reward_type = "combined"
        
        # 5. Logging
        log_msg = (
            f"Content: {content}\n"
            f"GT: ID={gt_id}, Time={gt_start}-{gt_end}\n"
            f"Pred: ID={pred_id}, Time={pred_start}-{pred_end}\n"
            f"Time IoU: {time_reward:.2f}, Spatial Reward: {spatial_reward}\n"
            f"Step: {current_step}, Reward Type: {reward_type}\n"
            f"------------- {current_time} Combined Reward: {combined_reward:.2f} -------------\n"
        )
        print(log_msg)
        
        if os.getenv("DEBUG_MODE") == "true":
            with open(os.getenv("LOG_PATH"), "a") as f:
                f.write(log_msg)
        
        rewards.append(combined_reward)
    
    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = re.compile(r'<think>.*?</think>\s*<answer>.*?</answer>', re.DOTALL)
    matches = [re.fullmatch(pattern, content.strip()) for content in completions]
    print('matches:', matches)
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "iou": iou_timestamp_reward, # Modified registry to use iou_timestamp_reward
    "format": format_reward,
}

def load_json_dataset(train_data_path, eval_data_path, video_folder, preprocessed_data_path=None): # Modified to accept preprocessed_data_path
    def create_dataset_from_json(file_path, split_name):
        with open(file_path, 'r') as f:
            data = json.load(f)
        examples = []
        for video_id, video_data in tqdm(data.items()):
            # for sentence_id, (timestamps, objectid, marker_scores, marker_ids, sentence) in enumerate(zip(video_data['timestamps'], video_data['objectid'], video_data['marker_scores'], video_data['marker_ids'], video_data['sentences'])):
            for sentence_id, (timestamps, objectid, sentence) in enumerate(zip(video_data['timestamps'], video_data['objectid'], video_data['sentences'])):
                sentence = sentence.strip().lower()
                if sentence.endswith("."):
                    sentence = sentence[:-1]
                video_filename_base = video_id
                video_path = None
                for ext in ['mp4', 'mkv', 'webm']:
                    candidate_path = os.path.join(video_folder, f"{video_filename_base}.{ext}")
                    if os.path.isfile(candidate_path):
                        video_path = candidate_path
                        break
                if video_path is None:
                    print(f"Warning: Video file not found for ID: {video_id}")
                    continue
                print('video_path:', video_path)
                example = {
                    "problem": sentence,
                    "solution": [objectid, timestamps[0], timestamps[1]],
                    "video_path": video_path,
                    "durations": video_data['duration'],
                    "preprocessed_path": "" # Initialize preprocessed_path as None
                }
                if preprocessed_data_path != "": # If preprocessed data path is provided, construct the path
                    example["preprocessed_path"] = os.path.join(preprocessed_data_path, split_name, f"{video_id}_{sentence_id}")
                examples.append(example)

        random.shuffle(examples)
        print(len(examples))
        print(examples[:5])
        dataset = Dataset.from_list(examples)

        def __getitem__(self, idx): # Define getitem within the scope where dataset is available
            example = dataset[idx]

            # return example
            data_to_return = {k: v for k, v in example.items()} # Create a copy to avoid modifying original dataset

            # print(data_to_return)
            # print("preprocessed_path:", example["preprocessed_path"])
            if example["preprocessed_path"] != "": # Check if preprocessed path exists
                try:
                    # data_to_return["image_inputs"] = [torch.load(os.path.join(example["preprocessed_path"][0], "image_inputs.pt"))]
                    data_to_return["video_inputs"] = [torch.load(os.path.join(example["preprocessed_path"][0], "video_inputs.pt"))]
                    with open(os.path.join(example["preprocessed_path"][0], "video_kwargs.json"), 'r') as f:
                        data_to_return["video_kwargs"] = [json.load(f)]
                    data_to_return["use_preprocessed"] = [True] # Flag to indicate preprocessed data is used
                except Exception as e:
                    print(f"Warning: Error loading preprocessed data from {example['preprocessed_path'][0]}, falling back to video_path. Error: {e}")
                    data_to_return["use_preprocessed"] = [False] # Fallback to video_path if loading fails
            else:
                data_to_return["use_preprocessed"] = [False] #  No preprocessed data to use or path invalid

            return data_to_return

        dataset.__getitem__ = __getitem__.__get__(dataset, Dataset) # Bind getitem to the dataset

        return dataset

    train_dataset = create_dataset_from_json(train_data_path, "train")
    eval_dataset = create_dataset_from_json(eval_data_path, "eval")
    return DatasetDict({"train": train_dataset, "eval": eval_dataset})

def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # # Load the dataset
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    # Load the dataset, now handles both raw and preprocessed data
    dataset = load_json_dataset(
        script_args.train_data_path,
        script_args.eval_data_path,
        script_args.video_folder,
        script_args.preprocessed_data_path # Pass preprocessed_data_path
    )

    
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
    print("using: ", trainer_cls)

    # from peft import LoraConfig, get_peft_model
 
    # lora_config = LoraConfig(
    #     task_type="CAUSAL_LM",
    #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    #     inference_mode=False,
    #     r=64,
    #     lora_alpha=16,
    #     lora_dropout=0.05,
    #     bias="none",
    # )
    
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    # trainer.train(resume_from_checkpoint="/data/TimeZero-main/outputs_markervideo_vidstg1/checkpoint-50")
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
