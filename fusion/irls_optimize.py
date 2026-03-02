import numpy as np
from SGHR.TransSync.sync import rotation_sync, translation_sync

def instance_aware_irls(processed_clouds, view_edges, instance_edges,
                        max_iter=20, alpha=0.5, sigma=0.1):
    # 初始化视角姿态与实例姿态
    init_poses = [np.eye(4) for _ in range(len(processed_clouds))]
    instance_poses = {}
    # 先构建视角间的相对姿态映射
    view_rel_poses = {}
    for edge in instance_edges:
        i, j, weight, rel_pose = edge
        if (i, j) not in instance_poses:
            instance_poses[(i, j)] = rel_pose
        # 同时构建视角间的相对姿态
        view_rel_poses[(i, j)] = rel_pose
    
    # 迭代优化
    for iter in range(max_iter):
        # 1. 姿态同步：同时优化视角姿态与实例姿态
        # 旋转同步
        view_rotations = [pose[:3, :3] for pose in init_poses]
        instance_rotations = [rel_pose[:3, :3] for rel_pose in instance_poses.values()]
        refined_view_rots, refined_instance_rots = rotation_sync(
            view_edges, instance_edges, view_rotations, instance_rotations
        )
        
        # 平移同步
        view_trans = [pose[:3, 3] for pose in init_poses]
        instance_trans = [rel_pose[:3, 3] for rel_pose in instance_poses.values()]
        refined_view_trans, refined_instance_trans = translation_sync(
            view_edges, instance_edges, refined_view_rots,
            refined_instance_rots, view_trans, instance_trans
        )
        
        # 2. 更新权重：结合视角残差与实例残差
        new_view_weights = []
        # 重新映射实例姿态的旋转和平移
        updated_instance_poses = {}
        for idx, (key, rel_pose) in enumerate(instance_poses.items()):
            new_rel_pose = np.eye(4)
            new_rel_pose[:3,:3] = refined_instance_rots[idx]
            new_rel_pose[:3,3] = refined_instance_trans[idx]
            updated_instance_poses[key] = new_rel_pose
        
        for edge in view_edges:
            i, j, old_weight = edge
            # 获取视角间的相对姿态
            relative_pose = view_rel_poses.get((i, j), np.linalg.pinv(init_poses[i]) @ init_poses[j])
            # 计算视角姿态残差
            view_res = compute_pose_residual(init_poses[i], init_poses[j], relative_pose)
            
            # 计算实例姿态残差
            instance_res = 0.0
            if (i, j) in updated_instance_poses:
                instance_rel = updated_instance_poses[(i, j)]
                est_rel_rot = refined_view_rots[i].T @ refined_view_rots[j]
                instance_res = np.linalg.norm(est_rel_rot - instance_rel[:3, :3])
            
            # 总残差
            total_res = view_res + alpha * instance_res
            # 更新权重
            new_weight = old_weight * np.exp(-total_res**2 / (2 * sigma**2))
            new_view_weights.append(new_weight)
        
        # 3. 更新姿态
        for i in range(len(init_poses)):
            init_poses[i][:3, :3] = refined_view_rots[i]
            init_poses[i][:3, 3] = refined_view_trans[i]
        
        # 4. 检查收敛
        pose_change = np.mean([np.linalg.norm(init_poses[i] - old_pose)
                              for i, old_pose in enumerate(init_poses)])
        if pose_change < 1e-6:
            break
    
    return init_poses, updated_instance_poses

def compute_pose_residual(pose1, pose2, relative_pose):
    # 计算两个姿态的残差
    est_rel = np.linalg.inv(pose1) @ pose2
    return np.linalg.norm(est_rel - relative_pose)

def compute_instance_residual(instance_rel, rot1, rot2):
    # 计算实例姿态的残差
    est_rel_rot = rot1.T @ rot2
    return np.linalg.norm(est_rel_rot - instance_rel[:3, :3])