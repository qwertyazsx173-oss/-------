import torch
import open3d as o3d
from SGHR.yoho.extract import extract_yoho_features
from MIRETR.vision3d.models import InstanceAwareTransformer

def preprocess_point_clouds(point_cloud_paths, miretr_ckpt_path):
    # 加载MIRETR实例分割预训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance_model = InstanceAwareTransformer.load_from_checkpoint(miretr_ckpt_path)
    instance_model = instance_model.to(device)
    instance_model.eval()
    
    processed_clouds = []
    for path in point_cloud_paths:
        # 读取点云
        pcd = o3d.io.read_point_cloud(path)
        points = torch.tensor(pcd.points, dtype=torch.float32).to(device)
        
        # SGHR原始预处理：下采样与YOHO特征提取
        downsampled = pcd.voxel_down_sample(voxel_size=0.05)
        yoho_feats = extract_yoho_features(downsampled)
        
        # MIRETR实例分割：提取实例掩码与实例特征
        with torch.no_grad():
            instance_masks, instance_feats = instance_model(points.unsqueeze(0))
        
        # 将掩码转回CPU，方便后续处理
        instance_masks = instance_masks[0].cpu()
        instance_feats = instance_feats[0].cpu()
        
        # 保存处理后的点云数据
        processed_clouds.append({
            "point_cloud": downsampled,
            "yoho_features": yoho_feats,
            "instance_masks": instance_masks,
            "instance_features": instance_feats
        })
    return processed_clouds