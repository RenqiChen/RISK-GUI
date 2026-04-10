from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, Union
import torch
import re
import ast
import math

def format_reward_os(completions, reward_coeffs):
    """Check if the Qwen model output matches a specific format."""
    import re
    import os
    from datetime import datetime

    def gui_format_reward(predict_str: str) -> float:
        """
        检查 In summary, the next action I will perform is
        ``` ```
        """
        # 检查 <think> 和 <answer> 的外部结构
        outer_pattern = re.compile(r".*?In summary, the next action.*?```.*?```.*?", re.DOTALL)
        if not re.fullmatch(outer_pattern, predict_str):
            return 0.0
        else:
            return 1.0
    
    completions = [completions]

    completion_contents = [completion for completion in completions]
    matches = [gui_format_reward(content) for content in completion_contents]

    rewards = [1.0 if match>0.5 else 0.0 for match in matches]
    for reward_index in range(len(rewards)):
        rewards[reward_index] = reward_coeffs*rewards[reward_index] 
    # print(f"format rewards: {rewards}")
    return rewards[0]

def action_reward_os(completions, solution, reward_coeffs):
    """Calculate IoU reward between predicted bounding box from Qwen model and ground truth bounding box."""
    import re
    import os
    from datetime import datetime
    import json
    
    def calculate_f1_score(predicted_str, ground_truth_str):
        predicted_str=predicted_str.replace("[","").replace("]","")
        ground_truth_str=ground_truth_str.replace("[","").replace("]","")
        predicted_tokens = set(predicted_str.lower().split())
        ground_truth_tokens = set(ground_truth_str.lower().split())

        if len(predicted_tokens)==1 and len(ground_truth_tokens)==1:
            predicted_token=list(predicted_tokens)[0]
            ground_truth_token=list(ground_truth_tokens)[0]
            if predicted_token in ground_truth_token or ground_truth_token in predicted_token:
                return 1
        
        common_tokens = predicted_tokens.intersection(ground_truth_tokens)
        if len(predicted_tokens) == 0:
            precision = 0
        else:
            precision = len(common_tokens) / len(predicted_tokens)
        if len(ground_truth_tokens) == 0:
            recall = 0
        else:
            recall = len(common_tokens) / len(ground_truth_tokens)
        
        if precision + recall == 0:
            f1_score = 0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score
    
    def extract_numbers(text):
        # 使用正则表达式找出所有数字序列
        numbers = re.findall(r'\d+', text)
        # 将字符串形式的数字转换为整数
        return [int(num) for num in numbers]
    
    completions = [completions]
    solution = [solution]
    contents = [completion for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    answer_tag_pattern = r'```(.*?)```'
    write=0
    for content, sol in zip(contents, solution):
        reward = 0.0
        # print("content:")
        # print(content)
        try:
            sol = re.findall(answer_tag_pattern, sol, re.DOTALL)[-1]
            content = re.findall(answer_tag_pattern, content, re.DOTALL)[-1]
            sol_list = sol.strip().split('[')
            if len(sol_list)==1:
                reward_tmp = calculate_f1_score(content, sol)
                if reward_tmp >= 0.5:
                    reward = 1.0
                else:
                    reward = 0.0
            else:
                gt_action_value = sol_list[0].strip()
                if gt_action_value in ["click"]:
                    if extract_numbers(content)[0]!=extract_numbers(sol)[0]:
                        reward = 0.0
                    else:
                        reward = 1.0
                else:
                    reward_tmp = calculate_f1_score(content, sol)
                    if reward_tmp >= 0.5:
                        reward = 1.0
                    else:
                        reward = 0.0

        except Exception:
            pass  # Continue to next verification method if this fails
                
        rewards.append(reward)

    for reward_index in range(len(rewards)):
        rewards[reward_index] = reward_coeffs*rewards[reward_index]
    
    # print(f"action rewards: {rewards}")
    return rewards[0]

def compute_score(prediction, ground_truth):
    format_coeff = 0.1
    accuracy_coeff = 0.9
    score = format_reward_os(prediction, format_coeff) + action_reward_os(prediction, ground_truth, accuracy_coeff)
    return score