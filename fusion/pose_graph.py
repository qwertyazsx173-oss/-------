import numpy as np
import torch
from SGHR.model.global_feature import compute_global_feature
from instance_pose import compute_instance_relative_pose, get_instance_inlier_ratio

def build_instance_aware_graph(processed_clouds, top_k=5):
    # 1. 计算所有视角的全局特征，用于估计视角重叠度
    global_features = []
    for cloud in processed_clouds:
        global_feat = compute_global_feature(cloud["yoho_features"])
        global_features.append(global_feat)
    
    # 2. 构建视角间边，过滤跨实例的无效边
    view_edges = []
    for i in range(len(processed_clouds)):
        # 计算与其他视角的重叠度
        overlaps = [
            (np.dot(global_features[i], global_features[j]) + 1) / 2
            for j in range(len(processed_clouds))
        ]
        # 选择前k个高重叠度的视角
        top_indices = np.argsort(overlaps)[-top_k:][::-1]
        
        for j in top_indices:
            if i == j:
                continue
            # 检查两个视角的实例匹配度
            instance_match = check_instance_overlap(
                processed_clouds[i]["instance_masks"],
                processed_clouds[j]["instance_masks"]
            )
            if instance_match > 0.3:  # 实例匹配度阈值
                # 边权重 = 视角重叠度 * 实例匹配度
                weight = overlaps[j] * instance_match
                view_edges.append((i, j, weight))
    
    # 3. 构建实例间边：同一实例在不同视角的对应边  
    instance_edges = []
    # 先统计所有实例ID
    all_instances = set()           
    for cloud in processed_clouds:
        instance_masks = cloud["instance_masks"]
        if instance_masks.is_cuda:
            instance_masks = instance_masks.cpu()
        all_instances.update(torch.unique(instance_masks).numpy())
    
    for instance_id in all_instances:
        # 找到包含该实例的所有视角
        instance_views = []
        for idx, cloud in enumerate(processed_clouds):
            instance_masks = cloud["instance_masks"]
            if instance_masks.is_cuda:
                instance_masks = instance_masks.cpu()
            if instance_id in instance_masks.numpy():
                instance_views.append(idx)
        
        # 为这些视角构建实例边
        for a in range(len(instance_views)):
            for b in range(a+1, len(instance_views)):
                i = instance_views[a]
                j = instance_views[b]
                # 计算该实例在两个视角的相对姿态
                relative_pose = compute_instance_relative_pose(
                    processed_clouds[i],
                    processed_clouds[j],
                    instance_id
                )
                # 边权重为实例配准的内点比例
                inlier_ratio = get_instance_inlier_ratio(
                    processed_clouds[i],
                    processed_clouds[j],
                    instance_id
                )
                instance_edges.append((i, j, inlier_ratio, relative_pose))
    
    return view_edges, instance_edges

def check_instance_overlap(mask1, mask2):
    # 计算两个视角的实例掩码交集比例
    if mask1.is_cuda:
        mask1 = mask1.cpu()
    if mask2.is_cuda:
        mask2 = mask2.cpu()
    mask1_np = mask1.numpy()
    mask2_np = mask2.numpy()
    
    # 统计共同实例的数量
    common_instances = set(mask1_np) & set(mask2_np)
    if not common_instances:
        return 0.0
    
    # 计算重叠比例
    count1 = np.sum([np.sum(mask1_np == c) for c in common_instances])
    count2 = np.sum([np.sum(mask2_np == c) for c in common_instances])
    return min(count1 / len(mask1_np), count2 / len(mask2_np))