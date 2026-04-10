from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, Union
import torch
import re
import ast
import math
import numpy as np

def format_reward_gui(completions, reward_coeffs):
    """Check if the Qwen model output matches a specific format."""
    import re
    import os
    from datetime import datetime

    def gui_format_reward(predict_str: str) -> float:
        """
        检查 predict_str 是否符合 <think></think><answer></answer> 的格式，
        并验证 <answer> 中的内容是否符合 [{'action': 'action', 'point': '[x,y]', 'input_text': 'no input text'}] 的格式要求。
        """
        # 检查 <think> 和 <answer> 的外部结构
        outer_pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
        if not re.fullmatch(outer_pattern, predict_str):
            return 0.0

        # 提取 <answer> 中的内容
        answer_match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
        if not answer_match:
            return 0.0

        # 提取 <answer> 内的内容并解析为 JSON 格式
        answer_content = answer_match.group(1).strip()
        try:
            actions = eval(answer_content)  # 尝试将 <answer> 的内容解析为 JSON

            # 验证 actions 是否为列表
            if not isinstance(actions, list):
                return 0.0

            # 验证每个 action 的格式
            for action in actions:
                if not isinstance(action, dict):
                    return 0.0
                # 检查 action 字典是否包含所需的键
                if "action" not in action or "point" not in action or "input_text" not in action:
                    return 0.0
                # 验证 action 的值是否符合要求
                if not isinstance(action["action"], str):
                    return 0.0
                if not (isinstance(action["point"][0],int) and isinstance(action["point"][1],int)):  # 匹配形如 [x,y] 的点
                    return 0.0
                if not isinstance(action["input_text"], str):
                    return 0.0
                if action["action"] in ['type', 'select','open_app'] and action["input_text"] in ['no input text']:
                    return 0.0
                if action["action"] in ['scroll'] and action["input_text"] not in ['left','right','up','down']:
                    return 0.0

            # 如果所有检查均通过，返回 1.0
            return 1.0
        except:
            return 0.0
    

    completions = [completions]
    completion_contents = [completion for completion in completions]
    matches = [gui_format_reward(content) for content in completion_contents]

    rewards = [1.0 if match>0.5 else 0.0 for match in matches]
    for reward_index in range(len(rewards)):
        rewards[reward_index] = reward_coeffs*rewards[reward_index]
    # print(f"format rewards: {rewards}")
    return rewards[0]

def gaussian_point_reward(completions, solution, reward_coeffs, image_size, grids):
    """Calculate IoU reward between predicted bounding box from Qwen model and ground truth bounding box."""
    import re
    import os
    from datetime import datetime
    import json

    def extract_action(content):
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        action_pattern = r'["\']action["\']:\s*["\'](\w+)["\']'
        content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
        if content_answer_match:
            content_answer = content_answer_match.group(1).strip()
            action_match = re.search(action_pattern, content_answer)
            if action_match:
                return action_match.group(1)
        return "no action"

    def extract_input_text(content):
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        action_pattern = r"'input_text':\s*'(.*?)'"
        content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
        if content_answer_match:
            content_answer = content_answer_match.group(1).strip()
            action_match = re.search(action_pattern, content_answer)
            if action_match:
                return action_match.group(1)
        return "no input text"

    def extract_coord(content):
        # Try to find the bbox within <answer> tags, if can not find, return [0, 0, 0, 0]
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern = r'\{.*\[(\d+),\s*(\d+)]\s*.*\}'
        content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
        try:
            if content_answer_match:
                content_answer = content_answer_match.group(1).strip()
                coord_match = re.search(bbox_pattern, content_answer)
                if coord_match:
                    coord = [int(coord_match.group(1)), int(coord_match.group(2))]
                    return coord, True
                else:
                    bbox_pattern = r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]"
                    match = re.search(bbox_pattern, content_answer)
                    if match:
                        coord = [float(match.group(i)) for i in range(1, 5)]
                        return coord, True
                    else:
                        return [0,0,0,0], False
            else:
                coord_pattern = r'\{.*\((\d+),\s*(\d+))\s*.*\}'
                coord_match = re.search(coord_pattern, content)
                if coord_match:
                    coord = [int(coord_match.group(1)), int(coord_match.group(2))]
                    return coord, True
                else:
                    bbox_pattern = r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]"
                    match = re.search(bbox_pattern, content_answer)
                    if match:
                        coord = [float(match.group(i)) for i in range(1, 3)]
                        return coord, True
                    else:
                        return [0,0,0,0], False
            return [0, 0, 0, 0], False
        except:
            return [0, 0, 0, 0], False
    
    def resize_bbox(bbox, input_height, input_width, image_height, image_width):

        if len(bbox)==4:
            bbox[0] = bbox[0] / input_width * image_width
            bbox[1] = bbox[1] / input_height * image_height
            bbox[2] = bbox[2] / input_width * image_width
            bbox[3] = bbox[3] / input_height * image_height
        else:
            bbox[0] = bbox[0] / input_width * image_width
            bbox[1] = bbox[1] / input_height * image_height
        
        # print(f"input width: {input_width}, input height: {input_height}, image width: {image_width}, image height: {image_height}")

        return bbox
    
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
    
    def g_point_reward(pred_point, gt_bbox):
        alpha = 0.5
        gt_x1, gt_y1, gt_x2, gt_y2 = gt_bbox
        
        # 计算中心点
        pred_center_x, pred_center_y = pred_point
        gt_center_x = (gt_x1 + gt_x2) / 2
        gt_center_y = (gt_y1 + gt_y2) / 2
        gt_width = gt_x2 - gt_x1
        gt_height = gt_y2 - gt_y1
        
        sigma_x = alpha * gt_width
        sigma_y = alpha * gt_height

        x_term = (pred_center_x - gt_center_x)**2 / (sigma_x**2)
        y_term = (pred_center_y - gt_center_y)**2 / (sigma_y**2)
        exponent = -0.5 * (x_term + y_term)
        point_reward = math.exp(exponent)
        point_reward = round(point_reward,3)
        return point_reward

    completions = [completions]

    contents = [completion for completion in completions]
    solution = [solution]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    write=0

    for i, (content, sol, img_size) in enumerate(zip(contents, solution, image_size)):
        image_grid_thw = grids
        reward = 0.0
        # print("content:")
        # print(content)
        try:
            sol = re.findall(answer_tag_pattern, sol, re.DOTALL)[-1]
            sol = eval(sol.strip())[0]
            gt_action=sol['action'].lower()
            gt_bbox=sol['gt_bbox']
            gt_input_text=sol['input_text']
            
            pred_action=extract_action(content).lower()
            pred_input_text=extract_input_text(content)
            pred_bbox,_=extract_coord(content)

            # Try symbolic verification first
            if pred_action!=gt_action:
                reward = 0.0

            if gt_action in ["click"]:
                if len(gt_bbox)==2:
                    if pred_bbox[0] < 1.1 and pred_bbox[1]<1.1:
                        pred_bbox[0]=pred_bbox[0]*img_size[0]
                        pred_bbox[1]=pred_bbox[1]*img_size[1]
                    else:
                        input_height = int(image_grid_thw[1]*14)
                        input_width = int(image_grid_thw[2]*14)
                        pred_bbox = resize_bbox(pred_bbox, input_height, input_width, img_size[1], img_size[0])
                    if gt_bbox[0] < 1.1 and gt_bbox[1] < 1.1:
                        gt_bbox[0]=gt_bbox[0]*img_size[0]
                        gt_bbox[1]=gt_bbox[1]*img_size[1]
                    if (pred_bbox[0]-gt_bbox[0])**2+(pred_bbox[1]-gt_bbox[1])**2<40**2:
                        reward = 1.0
                    else:
                        reward = 0.0
                elif len(gt_bbox)==4:

                    if len(pred_bbox)==2:
                        if gt_bbox[0] < 1.1 and gt_bbox[1] < 1.1 and gt_bbox[2]<1.1 and gt_bbox[3]<1.1:
                            gt_bbox[0]=gt_bbox[0]*img_size[0]
                            gt_bbox[1]=gt_bbox[1]*img_size[1]
                            gt_bbox[2]=gt_bbox[2]*img_size[0]
                            gt_bbox[3]=gt_bbox[3]*img_size[1]
                        if pred_bbox[0] < 1.1 and pred_bbox[1]<1.1:
                            pred_bbox[0]=pred_bbox[0]*img_size[0]
                            pred_bbox[1]=pred_bbox[1]*img_size[1]
                        else:
                            input_height = int(image_grid_thw[1]*14)
                            input_width = int(image_grid_thw[2]*14)
                            pred_bbox = resize_bbox(pred_bbox, input_height, input_width, img_size[1], img_size[0])
                        # if (gt_bbox[0]<pred_bbox[0]<gt_bbox[2]) and (gt_bbox[1]<pred_bbox[1]<gt_bbox[3]):
                        #     reward = 1.0
                        # else:
                        #     reward = 0.0
                        reward = g_point_reward(pred_bbox, gt_bbox)
                        # print(f"after: {pred_bbox}, gt_bbox: {gt_bbox}, reward: {reward}")
                    elif len(pred_bbox)==4:
                        if gt_bbox[0] < 1.1 and gt_bbox[1] < 1.1 and gt_bbox[2]<1.1 and gt_bbox[3]<1.1:
                            gt_bbox[0]=gt_bbox[0]*img_size[0]
                            gt_bbox[1]=gt_bbox[1]*img_size[1]
                            gt_bbox[2]=gt_bbox[2]*img_size[0]
                            gt_bbox[3]=gt_bbox[3]*img_size[1]
                        if pred_bbox[0] < 1.1 and pred_bbox[1] <1.1 and pred_bbox[2]<1.1 and pred_bbox[3]<1.1:
                            pred_bbox[0]=pred_bbox[0]*img_size[0]
                            pred_bbox[1]=pred_bbox[1]*img_size[1]
                            pred_bbox[2]=pred_bbox[2]*img_size[0]
                            pred_bbox[3]=pred_bbox[3]*img_size[1]
                        else:
                            input_height = int(image_grid_thw[1]*14)
                            input_width = int(image_grid_thw[2]*14)
                            pred_bbox = resize_bbox(pred_bbox, input_height, input_width, img_size[1], img_size[0])
                        def compute_iou(box1, box2):
                            # box: [x1, y1, x2, y2]
                            x_left = max(box1[0], box2[0])
                            y_top = max(box1[1], box2[1])
                            x_right = min(box1[2], box2[2])
                            y_bottom = min(box1[3], box2[3])

                            if x_right < x_left or y_bottom < y_top:
                                return 0.0  # 没有交集

                            intersection_area = (x_right - x_left) * (y_bottom - y_top)
                            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                            union_area = area1 + area2 - intersection_area

                            if union_area == 0:
                                return 0.0

                            return intersection_area / union_area
                        if compute_iou(pred_bbox, gt_bbox)>0.5:
                            reward=1.0
                        else:
                            reward=0.0
                    else:
                        reward = 0.0
                else:
                    reward = 0.0
            elif gt_action in ['type', 'select','scroll']:
                if calculate_f1_score(pred_input_text,gt_input_text)>=0.5:
                    reward = 1.0
                else:
                    reward = 0.0
            else:
                reward = 1.0

        except Exception as e:
            print(f"Exception occurred: {e}")
            pass  # Continue to next verification method if this fails
                
        rewards.append(reward)

    for reward_index in range(len(rewards)):
        rewards[reward_index] = reward_coeffs*rewards[reward_index]

    # print(f"action rewards: {rewards}")
    return rewards[0]

@staticmethod
def gaussian_plane_reward(completions, solution, reward_coeffs, image_size, step_id, **kwargs):
    """Calculate IoU reward between predicted bounding box from Qwen model and ground truth bounding box."""
    import re
    import os
    from datetime import datetime
    import json

    def extract_action(content):
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        action_pattern = r'["\']action["\']:\s*["\'](\w+)["\']'
        content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
        if content_answer_match:
            content_answer = content_answer_match.group(1).strip()
            action_match = re.search(action_pattern, content_answer)
            if action_match:
                return action_match.group(1)
        return "no action"

    def extract_input_text(content):
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        action_pattern = r"'input_text':\s*'(.*?)'"
        content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
        if content_answer_match:
            content_answer = content_answer_match.group(1).strip()
            action_match = re.search(action_pattern, content_answer)
            if action_match:
                return action_match.group(1)
        return "no input text"

    def extract_coord(content):
        # Try to find the bbox within <answer> tags, if can not find, return [0, 0, 0, 0]
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern = r'\{.*\[(\d+),\s*(\d+)]\s*.*\}'
        content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
        try:
            if content_answer_match:
                content_answer = content_answer_match.group(1).strip()
                coord_match = re.search(bbox_pattern, content_answer)
                if coord_match:
                    coord = [int(coord_match.group(1)), int(coord_match.group(2))]
                    return coord, True
                else:
                    bbox_pattern = r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]"
                    match = re.search(bbox_pattern, content_answer)
                    if match:
                        coord = [float(match.group(i)) for i in range(1, 5)]
                        return coord, True
                    else:
                        return [0,0,0,0], False
            else:
                coord_pattern = r'\{.*\((\d+),\s*(\d+))\s*.*\}'
                coord_match = re.search(coord_pattern, content)
                if coord_match:
                    coord = [int(coord_match.group(1)), int(coord_match.group(2))]
                    return coord, True
                else:
                    bbox_pattern = r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]"
                    match = re.search(bbox_pattern, content_answer)
                    if match:
                        coord = [float(match.group(i)) for i in range(1, 3)]
                        return coord, True
                    else:
                        return [0,0,0,0], False
            return [0, 0, 0, 0], False
        except:
            return [0, 0, 0, 0], False
    
    def resize_bbox(bbox, input_height, input_width, image_height, image_width):

        if len(bbox)==4:
            bbox[0] = bbox[0] / input_width * image_width
            bbox[1] = bbox[1] / input_height * image_height
            bbox[2] = bbox[2] / input_width * image_width
            bbox[3] = bbox[3] / input_height * image_height
        else:
            bbox[0] = bbox[0] / input_width * image_width
            bbox[1] = bbox[1] / input_height * image_height
        
        # print(f"input width: {input_width}, input height: {input_height}, image width: {image_width}, image height: {image_height}")

        return bbox
    
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
    
    def g_plane_reward(pred_bbox, gt_bbox):
        alpha = 0.5
        eps   = 1e-8
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_bbox
        gt_x1, gt_y1, gt_x2, gt_y2 = gt_bbox
        
        pred_center_x = (pred_x1 + pred_x2) / 2
        pred_center_y = (pred_y1 + pred_y2) / 2
        pred_width = pred_x2 - pred_x1
        pred_height = pred_y2 - pred_y1
        # pred_μ
        pred_mu = np.array([pred_center_x, pred_center_y])

        gt_center_x = (gt_x1 + gt_x2) / 2
        gt_center_y = (gt_y1 + gt_y2) / 2
        # gt_μ
        gt_mu = np.array([gt_center_x, gt_center_y])
        gt_width = gt_x2 - gt_x1
        gt_height = gt_y2 - gt_y1

        # 1 sigma
        pred_sigma_x = pred_width * alpha
        pred_sigma_y = pred_height * alpha
        gt_sigma_x   = gt_width * alpha
        gt_sigma_y = gt_height * alpha

        pred_cov = np.array([[pred_sigma_x**2, 0], 
                            [0, pred_sigma_y**2]])
        
        # Σ2 (ground truth distribution covariance matrix)  
        gt_cov = np.array([[gt_sigma_x**2, 0], 
                        [0, gt_sigma_y**2]])
        
        sigma_avg = (pred_cov + gt_cov) / 2
        # 
        mu_diff = pred_mu - gt_mu
        
        # (1/8) * (μ1 - μ2)^T * Σ^(-1) * (μ1 - μ2)
        sigma_avg_inv = np.linalg.inv(sigma_avg + eps * np.eye(2))
        term1 = (1/8) * np.dot(mu_diff.T, np.dot(sigma_avg_inv, mu_diff))
        
        # (1/2) * ln(det(Σ) / sqrt(det(Σ1) * det(Σ2)))
        det_sigma_avg = np.linalg.det(sigma_avg)
        det_pred_cov = np.linalg.det(pred_cov)
        det_gt_cov = np.linalg.det(gt_cov)
        try:
            term2 = 0.5 * np.log(det_sigma_avg / (np.sqrt(det_pred_cov * det_gt_cov + eps)))
        except:
            return 0.0
        bhattacharyya_distance = term1 + term2

        # 转换为奖励
        plane_reward = np.exp(-bhattacharyya_distance)
        plane_reward = round(plane_reward,3)
        return plane_reward

    completions = [completions]

    contents = [completion for completion in completions]
    solution = [solution]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    write=0

    for i, (content, sol, img_size) in enumerate(zip(contents, solution, image_size)):
        image_grid_thw = kwargs.get("image_grid_thw")[i]
        reward = 0.0
        # print("content:")
        # print(content)
        try:
            sol = re.findall(answer_tag_pattern, sol, re.DOTALL)[-1]
            sol = eval(sol.strip())[0]
            gt_action=sol['action'].lower()
            gt_bbox=sol['gt_bbox']
            gt_input_text=sol['input_text']
            
            pred_action=extract_action(content).lower()
            pred_input_text=extract_input_text(content)
            pred_bbox,_=extract_coord(content)

            # Try symbolic verification first
            if pred_action!=gt_action:
                reward = 0.0

            if gt_action in ["click"]:
                if len(gt_bbox)==2:
                    if pred_bbox[0] < 1.1 and pred_bbox[1]<1.1:
                        pred_bbox[0]=pred_bbox[0]*img_size[0]
                        pred_bbox[1]=pred_bbox[1]*img_size[1]
                    else:
                        input_height = int(image_grid_thw[1]*14)
                        input_width = int(image_grid_thw[2]*14)
                        pred_bbox = resize_bbox(pred_bbox, input_height, input_width, img_size[1], img_size[0])
                    if gt_bbox[0] < 1.1 and gt_bbox[1] < 1.1:
                        gt_bbox[0]=gt_bbox[0]*img_size[0]
                        gt_bbox[1]=gt_bbox[1]*img_size[1]
                    if (pred_bbox[0]-gt_bbox[0])**2+(pred_bbox[1]-gt_bbox[1])**2<40**2:
                        reward = 1.0
                    else:
                        reward = 0.0
                elif len(gt_bbox)==4:

                    if len(pred_bbox)==2:
                        if gt_bbox[0] < 1.1 and gt_bbox[1] < 1.1 and gt_bbox[2]<1.1 and gt_bbox[3]<1.1:
                            gt_bbox[0]=gt_bbox[0]*img_size[0]
                            gt_bbox[1]=gt_bbox[1]*img_size[1]
                            gt_bbox[2]=gt_bbox[2]*img_size[0]
                            gt_bbox[3]=gt_bbox[3]*img_size[1]
                        if pred_bbox[0] < 1.1 and pred_bbox[1]<1.1:
                            pred_bbox[0]=pred_bbox[0]*img_size[0]
                            pred_bbox[1]=pred_bbox[1]*img_size[1]
                        else:
                            input_height = int(image_grid_thw[1]*14)
                            input_width = int(image_grid_thw[2]*14)
                            pred_bbox = resize_bbox(pred_bbox, input_height, input_width, img_size[1], img_size[0])
                        # if (gt_bbox[0]<pred_bbox[0]<gt_bbox[2]) and (gt_bbox[1]<pred_bbox[1]<gt_bbox[3]):
                        #     reward = 1.0
                        # else:
                        #     reward = 0.0
                        reward = g_point_reward(pred_bbox, gt_bbox)
                        print(f"after: {pred_bbox}, gt_bbox: {gt_bbox}, reward: {reward}")
                    elif len(pred_bbox)==4:
                        if gt_bbox[0] < 1.1 and gt_bbox[1] < 1.1 and gt_bbox[2]<1.1 and gt_bbox[3]<1.1:
                            gt_bbox[0]=gt_bbox[0]*img_size[0]
                            gt_bbox[1]=gt_bbox[1]*img_size[1]
                            gt_bbox[2]=gt_bbox[2]*img_size[0]
                            gt_bbox[3]=gt_bbox[3]*img_size[1]
                        if pred_bbox[0] < 1.1 and pred_bbox[1] <1.1 and pred_bbox[2]<1.1 and pred_bbox[3]<1.1:
                            pred_bbox[0]=pred_bbox[0]*img_size[0]
                            pred_bbox[1]=pred_bbox[1]*img_size[1]
                            pred_bbox[2]=pred_bbox[2]*img_size[0]
                            pred_bbox[3]=pred_bbox[3]*img_size[1]
                        else:
                            input_height = int(image_grid_thw[1]*14)
                            input_width = int(image_grid_thw[2]*14)
                            pred_bbox = resize_bbox(pred_bbox, input_height, input_width, img_size[1], img_size[0])
                        def compute_iou(box1, box2):
                            # box: [x1, y1, x2, y2]
                            x_left = max(box1[0], box2[0])
                            y_top = max(box1[1], box2[1])
                            x_right = min(box1[2], box2[2])
                            y_bottom = min(box1[3], box2[3])

                            if x_right < x_left or y_bottom < y_top:
                                return 0.0  # 没有交集

                            intersection_area = (x_right - x_left) * (y_bottom - y_top)
                            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                            union_area = area1 + area2 - intersection_area

                            if union_area == 0:
                                return 0.0

                            return intersection_area / union_area
                        if compute_iou(pred_bbox, gt_bbox)>0.5:
                            reward=1.0
                        else:
                            reward=0.0
                    else:
                        reward = 0.0
                else:
                    reward = 0.0
            elif gt_action in ['type', 'select','scroll']:
                if calculate_f1_score(pred_input_text,gt_input_text)>=0.5:
                    reward = 1.0
                else:
                    reward = 0.0
            else:
                reward = 1.0

        except Exception as e:
            print(f"Exception occurred: {e}")
            pass  # Continue to next verification method if this fails
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
            problem = kwargs.get("problem")[0]
            if reward <= 1.0 and write==0:  # this condition can be changed for debug
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"image_path: {image_path}\n")
                    f.write(f"problem: {problem}\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")
                write=1
    for reward_index in range(len(rewards)):
        rewards[reward_index] = reward_coeffs*rewards[reward_index]

    return rewards

@staticmethod
def action_reward(completions, solution, reward_coeffs, image_size, grids):
    """Calculate IoU reward between predicted bounding box from Qwen model and ground truth bounding box."""
    import re
    import os
    from datetime import datetime
    import json

    def extract_action(content):
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        action_pattern = r'["\']action["\']:\s*["\'](\w+)["\']'
        content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
        if content_answer_match:
            content_answer = content_answer_match.group(1).strip()
            action_match = re.search(action_pattern, content_answer)
            if action_match:
                return action_match.group(1)
        return "no action"

    def extract_input_text(content):
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        action_pattern = r"'input_text':\s*'(.*?)'"
        content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
        if content_answer_match:
            content_answer = content_answer_match.group(1).strip()
            action_match = re.search(action_pattern, content_answer)
            if action_match:
                return action_match.group(1)
        return "no input text"

    def extract_coord(content):
        # Try to find the bbox within <answer> tags, if can not find, return [0, 0, 0, 0]
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern = r'\{.*\[(\d+),\s*(\d+)]\s*.*\}'
        content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
        try:
            if content_answer_match:
                content_answer = content_answer_match.group(1).strip()
                coord_match = re.search(bbox_pattern, content_answer)
                if coord_match:
                    coord = [int(coord_match.group(1)), int(coord_match.group(2))]
                    return coord, True
                else:
                    bbox_pattern = r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]"
                    match = re.search(bbox_pattern, content_answer)
                    if match:
                        coord = [float(match.group(i)) for i in range(1, 5)]
                        return coord, True
                    else:
                        return [0,0,0,0], False
            else:
                coord_pattern = r'\{.*\((\d+),\s*(\d+))\s*.*\}'
                coord_match = re.search(coord_pattern, content)
                if coord_match:
                    coord = [int(coord_match.group(1)), int(coord_match.group(2))]
                    return coord, True
                else:
                    bbox_pattern = r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]"
                    match = re.search(bbox_pattern, content_answer)
                    if match:
                        coord = [float(match.group(i)) for i in range(1, 3)]
                        return coord, True
                    else:
                        return [0,0,0,0], False
            return [0, 0, 0, 0], False
        except:
            return [0, 0, 0, 0], False
    
    def resize_bbox(bbox, input_height, input_width, image_height, image_width):

        if len(bbox)==4:
            bbox[0] = bbox[0] / input_width * image_width
            bbox[1] = bbox[1] / input_height * image_height
            bbox[2] = bbox[2] / input_width * image_width
            bbox[3] = bbox[3] / input_height * image_height
        else:
            bbox[0] = bbox[0] / input_width * image_width
            bbox[1] = bbox[1] / input_height * image_height
        
        # print(f"input width: {input_width}, input height: {input_height}, image width: {image_width}, image height: {image_height}")

        return bbox
    
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

    completions = [completions]

    contents = [completion for completion in completions]
    solution = [solution]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    write=0

    for i, (content, sol, img_size) in enumerate(zip(contents, solution, image_size)):
        image_grid_thw = grids
        reward = 0.0
        # print("content:")
        # print(content)
        try:
            sol = re.findall(answer_tag_pattern, sol, re.DOTALL)[-1]
            sol = eval(sol.strip())[0]
            gt_action=sol['action'].lower()
            gt_bbox=sol['gt_bbox']
            gt_input_text=sol['input_text']
            
            pred_action=extract_action(content).lower()
            pred_input_text=extract_input_text(content)
            pred_bbox,_=extract_coord(content)

            # Try symbolic verification first
            if pred_action!=gt_action:
                reward = 0.0

            if gt_action in ["click"]:
                if len(gt_bbox)==2:
                    if pred_bbox[0] < 1.1 and pred_bbox[1]<1.1:
                        pred_bbox[0]=pred_bbox[0]*img_size[0]
                        pred_bbox[1]=pred_bbox[1]*img_size[1]
                    else:
                        input_height = int(image_grid_thw[1]*14)
                        input_width = int(image_grid_thw[2]*14)
                        pred_bbox = resize_bbox(pred_bbox, input_height, input_width, img_size[1], img_size[0])
                    if gt_bbox[0] < 1.1 and gt_bbox[1] < 1.1:
                        gt_bbox[0]=gt_bbox[0]*img_size[0]
                        gt_bbox[1]=gt_bbox[1]*img_size[1]
                    if (pred_bbox[0]-gt_bbox[0])**2+(pred_bbox[1]-gt_bbox[1])**2<40**2:
                        reward = 1.0
                    else:
                        reward = 0.0
                elif len(gt_bbox)==4:

                    if len(pred_bbox)==2:
                        if gt_bbox[0] < 1.1 and gt_bbox[1] < 1.1 and gt_bbox[2]<1.1 and gt_bbox[3]<1.1:
                            gt_bbox[0]=gt_bbox[0]*img_size[0]
                            gt_bbox[1]=gt_bbox[1]*img_size[1]
                            gt_bbox[2]=gt_bbox[2]*img_size[0]
                            gt_bbox[3]=gt_bbox[3]*img_size[1]
                        if pred_bbox[0] < 1.1 and pred_bbox[1]<1.1:
                            pred_bbox[0]=pred_bbox[0]*img_size[0]
                            pred_bbox[1]=pred_bbox[1]*img_size[1]
                        else:
                            input_height = int(image_grid_thw[1]*14)
                            input_width = int(image_grid_thw[2]*14)
                            print(f"before: {pred_bbox}")
                            pred_bbox = resize_bbox(pred_bbox, input_height, input_width, img_size[1], img_size[0])
                        # if (gt_bbox[0]<pred_bbox[0]<gt_bbox[2]) and (gt_bbox[1]<pred_bbox[1]<gt_bbox[3]):
                        #     reward = 1.0
                        # else:
                        #     reward = 0.0
                        if (gt_bbox[0]<pred_bbox[0]<gt_bbox[2]) and (gt_bbox[1]<pred_bbox[1]<gt_bbox[3]):
                            center_x = (gt_bbox[0]+gt_bbox[2])/2
                            center_y = (gt_bbox[1]+gt_bbox[3])/2
                            if (pred_bbox[0]-center_x)**2+(pred_bbox[1]-center_y)**2<40**2:
                                reward = 1.0
                            else:
                                reward = 0.0
                        else:
                            reward = 0.0
                        print(f"after: {pred_bbox}, gt_bbox: {gt_bbox}, reward: {reward}")
                    elif len(pred_bbox)==4:
                        if gt_bbox[0] < 1.1 and gt_bbox[1] < 1.1 and gt_bbox[2]<1.1 and gt_bbox[3]<1.1:
                            gt_bbox[0]=gt_bbox[0]*img_size[0]
                            gt_bbox[1]=gt_bbox[1]*img_size[1]
                            gt_bbox[2]=gt_bbox[2]*img_size[0]
                            gt_bbox[3]=gt_bbox[3]*img_size[1]
                        if pred_bbox[0] < 1.1 and pred_bbox[1] <1.1 and pred_bbox[2]<1.1 and pred_bbox[3]<1.1:
                            pred_bbox[0]=pred_bbox[0]*img_size[0]
                            pred_bbox[1]=pred_bbox[1]*img_size[1]
                            pred_bbox[2]=pred_bbox[2]*img_size[0]
                            pred_bbox[3]=pred_bbox[3]*img_size[1]
                        else:
                            input_height = int(image_grid_thw[1]*14)
                            input_width = int(image_grid_thw[2]*14)
                            pred_bbox = resize_bbox(pred_bbox, input_height, input_width, img_size[1], img_size[0])
                        def compute_iou(box1, box2):
                            # box: [x1, y1, x2, y2]
                            x_left = max(box1[0], box2[0])
                            y_top = max(box1[1], box2[1])
                            x_right = min(box1[2], box2[2])
                            y_bottom = min(box1[3], box2[3])

                            if x_right < x_left or y_bottom < y_top:
                                return 0.0  # 没有交集

                            intersection_area = (x_right - x_left) * (y_bottom - y_top)
                            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                            union_area = area1 + area2 - intersection_area

                            if union_area == 0:
                                return 0.0

                            return intersection_area / union_area
                        if compute_iou(pred_bbox, gt_bbox)>0.5:
                            reward=1.0
                        else:
                            reward=0.0
                    else:
                        reward = 0.0
                else:
                    reward = 0.0
            elif gt_action in ['type', 'select','scroll']:
                if calculate_f1_score(pred_input_text,gt_input_text)>=0.5:
                    reward = 1.0
                else:
                    reward = 0.0
            else:
                reward = 1.0

        except Exception as e:
            print(f"Exception occurred: {e}")
            pass  # Continue to next verification method if this fails
                
        rewards.append(reward)

    for reward_index in range(len(rewards)):
        rewards[reward_index] = reward_coeffs*rewards[reward_index]

    # print(f"action rewards: {rewards}")
    return rewards[0]

def compute_score(prediction, ground_truth, extra_info):
    img_size = extra_info.get("img_size")
    grids = extra_info.get("grids")
    format_coeff = 0.1
    accuracy_coeff = 0.9
    score = format_reward_gui(prediction, format_coeff) + gaussian_point_reward(prediction, ground_truth, accuracy_coeff, img_size, grids)
    return score