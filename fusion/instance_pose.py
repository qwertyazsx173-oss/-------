import torch
import open3d as o3d
import numpy as np
from SGHR.yoho.extract import extract_yoho_features

def get_instance_point_cloud(processed_cloud, instance_id):
    """从处理后的点云中提取指定实例的点云
    Args:
        processed_cloud: 预处理后的单视角点云数据，包含点云、实例掩码等信息
        instance_id: 目标实例的ID
    Returns:
        instance_pcd: 该实例的Open3D点云对象，若不存在则返回None
    """
    # 获取实例掩码的numpy数组
    instance_masks = processed_cloud["instance_masks"]
    if instance_masks.is_cuda:
        instance_masks = instance_masks.cpu()
    instance_mask_np = instance_masks.numpy()
    # 筛选出该实例对应的点的索引
    instance_indices = np.where(instance_mask_np == instance_id)[0]
    
    if len(instance_indices) == 0:
        return None
    
    # 从下采样后的点云中提取该实例的点云
    instance_pcd = processed_cloud["point_cloud"].select_by_index(instance_indices)
    return instance_pcd

def compute_instance_relative_pose(processed_cloud_i, processed_cloud_j, instance_id):
    """计算两个视角中同一个实例的相对姿态
    Args:
        processed_cloud_i: 视角i的预处理后点云数据
        processed_cloud_j: 视角j的预处理后点云数据
        instance_id: 目标实例的ID
    Returns:
        relative_pose: 4x4的相对姿态变换矩阵，将视角i的实例点云变换到视角j的坐标系；失败返回单位矩阵
    """
    # 提取两个视角中的目标实例点云
    instance_pcd_i = get_instance_point_cloud(processed_cloud_i, instance_id)
    instance_pcd_j = get_instance_point_cloud(processed_cloud_j, instance_id)
    
    if instance_pcd_i is None or instance_pcd_j is None:
        return np.eye(4)
    
    # 为实例点云提取YOHO特征，用于特征匹配
    feats_i = extract_yoho_features(instance_pcd_i)
    feats_j = extract_yoho_features(instance_pcd_j)
    
    # 构建特征匹配的特征对象
    feature_i = o3d.pipelines.registration.Feature()
    feature_i.data = feats_i.T
    feature_j = o3d.pipelines.registration.Feature()
    feature_j.data = feats_j.T
    
    # 使用基于特征匹配的RANSAC进行配准
    distance_threshold = 0.05  # 距离阈值，根据场景调整
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        instance_pcd_i,
        instance_pcd_j,
        feature_i,
        feature_j,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    
    return result.transformation

def get_instance_inlier_ratio(processed_cloud_i, processed_cloud_j, instance_id):
    """计算实例配准的内点比例，即配准成功的点占总点数的比例
    Args:
        processed_cloud_i: 视角i的预处理后点云数据
        processed_cloud_j: 视角j的预处理后点云数据
        instance_id: 目标实例的ID
    Returns:
        inlier_ratio: 内点比例，范围0~1；失败返回0
    """
    # 提取实例点云
    instance_pcd_i = get_instance_point_cloud(processed_cloud_i, instance_id)
    instance_pcd_j = get_instance_point_cloud(processed_cloud_j, instance_id)
    
    if instance_pcd_i is None or instance_pcd_j is None:
        return 0.0
    #
    # 计算该实例在两个视角间的相对姿态
    relative_pose = compute_instance_relative_pose(processed_cloud_i, processed_cloud_j, instance_id)
    # 将视角i的实例点云变换到视角j的坐标系
    transformed_pcd_i = instance_pcd_i.transform(relative_pose)
    
    # 构建KD树用于最近邻搜索
    pcd_tree = o3d.geometry.KDTreeFlann(instance_pcd_j)
    inlier_count = 0
    distance_threshold = 0.05  # 距离阈值，与配准阈值一致
    
    # 转换为numpy数组进行遍历
    transformed_points = np.asarray(transformed_pcd_i.points)
    target_points = np.asarray(instance_pcd_j.points)
    
    if len(transformed_points) == 0:
        return 0.0
    
    # 统计内点数量
    for point in transformed_points:
        [k, idx, _] = pcd_tree.search_knn_vector_3d(point, 1)
        if np.linalg.norm(point - target_points[idx[0]]) < distance_threshold:
            inlier_count += 1
    
    # 计算内点比例
    return inlier_count / len(transformed_points)