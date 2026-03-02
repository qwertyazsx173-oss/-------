import numpy as np
import torch

#
def global_instance_optimize(processed_clouds, view_poses, instance_poses):
    # 1. 收集每个实例在所有视角中的姿态
    instance_global_poses = {}
    all_instances = set()
    for cloud in processed_clouds:
        # 处理CUDA张量的情况
        instance_masks = cloud["instance_masks"]
        if instance_masks.is_cuda:
            instance_masks = instance_masks.cpu()
        all_instances.update(torch.unique(instance_masks).numpy())

    for instance_id in all_instances:
        instance_poses_list = []
        for idx, cloud in enumerate(processed_clouds):
            instance_masks = cloud["instance_masks"]
            if instance_masks.is_cuda:
                instance_masks = instance_masks.cpu()
            if instance_id in instance_masks.numpy():
                # 计算该实例的全局姿态：视角姿态 * 实例相对姿态
                rel_pose = instance_poses.get((idx, idx), np.eye(4))  # 自身相对姿态为单位矩阵
                global_pose = view_poses[idx] @ rel_pose
                instance_poses_list.append(global_pose)

        # 计算实例的平均全局姿态
        if instance_poses_list:
            avg_pose = np.mean(instance_poses_list, axis=0)
            instance_global_poses[instance_id] = avg_pose

    # 2. 更新视角姿态，确保与实例全局姿态一致
    refined_view_poses = []
    for idx, pose in enumerate(view_poses):
        cloud = processed_clouds[idx]
        instance_masks = cloud["instance_masks"]
        if instance_masks.is_cuda:
            instance_masks = instance_masks.cpu()
        # 计算该视角下所有实例的期望姿态
        expected_poses = []
        for instance_id in torch.unique(instance_masks).numpy():
            if instance_id in instance_global_poses:
                # 期望的实例相对姿态 = 视角姿态逆 * 实例全局姿态，使用伪逆避免奇异矩阵问题
                expected_rel = np.linalg.pinv(pose) @ instance_global_poses[instance_id]
                expected_poses.append(expected_rel)

        # 微调视角姿态，最小化与所有实例期望姿态的残差
        if expected_poses:
            # 加权平均微调
            avg_rel = np.mean(expected_poses, axis=0)
            refined_pose = pose @ avg_rel
            refined_view_poses.append(refined_pose)
        else:
            refined_view_poses.append(pose)

    return refined_view_poses, instance_global_poses