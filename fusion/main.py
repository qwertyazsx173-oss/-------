import argparse
import os
import shutil
import torch
import open3d as o3d
from preprocess import preprocess_point_clouds
from pose_graph import build_instance_aware_graph
from irls_optimize import instance_aware_irls
from global_optimize import global_instance_optimize

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to point cloud dataset")
    parser.add_argument("--miretr_ckpt", type=str, required=True, help="Path to MIRETR pretrained checkpoint")
    parser.add_argument("--top_k", type=int, default=5, help="Top k views for pose graph")
    parser.add_argument("--max_iter", type=int, default=20, help="Max IRLS iterations")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for results")
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 预处理点云：排序文件，确保处理顺序一致
    point_cloud_paths = sorted(
        [os.path.join(args.data_path, f) for f in os.listdir(args.data_path) if f.endswith(".ply")])
    processed_clouds = preprocess_point_clouds(point_cloud_paths, args.miretr_ckpt)

    # 2. 构建实例感知姿态图
    view_edges, instance_edges = build_instance_aware_graph(processed_clouds, args.top_k)

    # 3. 实例感知IRLS优化
    view_poses, instance_poses = instance_aware_irls(processed_clouds, view_edges, instance_edges, args.max_iter)

    # 4. 全局优化
    refined_view_poses, instance_global_poses = global_instance_optimize(processed_clouds, view_poses, instance_poses)

    # 5. 保存与可视化结果
    # 保存配准后的点云
    for idx, (cloud, pose) in enumerate(zip(processed_clouds, refined_view_poses)):
        transformed_pcd = cloud["point_cloud"].transform(pose)
        save_path = os.path.join(args.output_dir, f"registered_cloud_{idx}.ply")
        o3d.io.write_point_cloud(save_path, transformed_pcd)

    # 保存实例姿态
    pose_save_path = os.path.join(args.output_dir, "instance_poses.txt")
    with open(pose_save_path, "w") as f:
        for instance_id, pose in instance_global_poses.items():
            f.write(f"Instance {instance_id}:\n")
            f.write(f"{pose}\n\n")

    # 可视化结果
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for cloud in processed_clouds:
        vis.add_geometry(cloud["point_cloud"])
        # 更新几何对象
        vis.update_geometry(cloud["point_cloud"])
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()