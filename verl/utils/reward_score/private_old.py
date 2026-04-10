from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, Union
import torch
import re
import ast
import math

def format_reward_private_old(completions, reward_coeffs, process_ids):
    """Check if the Qwen model output matches a specific format."""
    import re
    import os
    import json
    from datetime import datetime

    def gui_format_reward(predict_str: str) -> float:
        """
        验证能否读为json
        """
        try:
            # 使用非贪婪匹配找到最外层的大括号对
            bracket_pattern = re.compile(r".*?(\{.*\}).*?", re.DOTALL)
            match = bracket_pattern.search(predict_str)

            if not match:
                return 0.0
            else:
                # 获取大括号及其内容
                bracket_content_with_braces = match.group(1).strip()
                json_content = json.loads(bracket_content_with_braces)

            answer = json_content
            action = answer['action']
            evaluation_previous_goal = answer['evaluation_previous_goal']
            memory = answer['memory']
            next_goal = answer['next_goal']
            if not isinstance(answer['action'], list):
                return 0.0
            for action_index in answer['action']:
                if not isinstance(action_index, dict):
                    return 0.0
            return 1.0
        except:           
            return 0.0
    
    completions = [completions]
    completion_contents = [completion for completion in completions]
    matches = [gui_format_reward(content) for content in completion_contents]

    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
            f.write(f"------------- {current_time} Format reward -------------\n")
            for content, match in zip(completion_contents, matches):
                f.write(f"Content: {content}\n")
                f.write(f"Has format: {match>0.5}\n")
                break
    rewards = [1.0 if match>0.5 else 0.0 for match in matches]
    for reward_index in range(len(rewards)):
        rewards[reward_index] = reward_coeffs*rewards[reward_index]

    import math
    def reward_for_step(step, max_step=10):
        # 归一化步数到[-2, 2]区间
        x = (step - 1) / (max_step - 1) * 4 - 2
        base = 1 / (1 + math.exp(-x))  # sigmoid输出(0,1)
        # 缩放到[0.7, 1]
        reward = 0.7 + 0.3 * base
        return reward

    for reward_index in range(len(rewards)):
        rewards[reward_index] = rewards[reward_index]*reward_for_step(process_ids['step_id'],process_ids['total_steps']) 
    # print(f"format rewards: {rewards}")
    return rewards[0]
    
def action_reward_private(completions, solution, reward_coeffs, process_ids, step_id):
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
    
    def parse_action(s):
        try:
            return json.loads(s)
        except Exception:
            return ast.literal_eval(s)
    
    completions = [completions]
    solution = [solution]
    contents = [completion for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    answer_tag_pattern = r"['\"]action['\"]\s*:\s*(\[[\s\S]*?\])"
    write=0
    for content, sol in zip(contents, solution):
        reward = 0.0
        # print("content:")
        # print(content)
        try:
            sol = re.findall(answer_tag_pattern, sol, re.DOTALL)[-1].strip()
            sol = parse_action(sol)
            content = re.findall(answer_tag_pattern, content, re.DOTALL)[-1].strip()
            content = parse_action(content)
            if step_id < 3200:
                reward_index = 1.0 / len(sol)
                for index in range(len(sol)):
                    if index < len(content):
                        sol_index = json.dumps([sol[index]], ensure_ascii=False)
                        content_index = json.dumps([content[index]], ensure_ascii=False)
                        reward_tmp = calculate_f1_score(content_index, sol_index)
                        if reward_tmp >= 0.5:
                            reward = reward + reward_index
                        else:
                            reward = reward + 0.0
                if reward<0.9:
                    reward = 0.5*reward
            else:
                sol = json.dumps(sol, ensure_ascii=False)
                content = json.dumps(content, ensure_ascii=False)
                reward_tmp = calculate_f1_score(content, sol)
                if reward_tmp >= 0.5:
                    reward = 1.0
                else:
                    reward = 0.0
            # print(f"sol:{sol}, content:{content}, reward: {reward}")
        except Exception:
            pass  # Continue to next verification method if this fails
                
        rewards.append(reward)

    for reward_index in range(len(rewards)):
        rewards[reward_index] = reward_coeffs*rewards[reward_index]

    import math
    def reward_for_step(step, max_step=10):
        # 归一化步数到[-2, 2]区间
        x = (step - 1) / (max_step - 1) * 4 - 2
        base = 1 / (1 + math.exp(-x))  # sigmoid输出(0,1)
        # 缩放到[0.7, 1]
        reward = 0.7 + 0.3 * base
        return reward

    for reward_index in range(len(rewards)):
        rewards[reward_index] = rewards[reward_index]*reward_for_step(process_ids['step_id'],process_ids['total_steps'])
    # print(f"action rewards: {rewards}")
    return rewards[0]

def compute_score(prediction, ground_truth, extra_info, global_steps):
    process_id = extra_info.get("process_id")
    step_id = global_steps
    format_coeff = 0.1
    accuracy_coeff = 0.9
    score = format_reward_private_old(prediction, format_coeff, process_id) + action_reward_private(prediction, ground_truth, accuracy_coeff, process_id, step_id)
    return score

