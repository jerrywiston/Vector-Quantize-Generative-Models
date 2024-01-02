import numpy as np
import torch

def get_batch(color, pose, obs_size=12, batch_size=32, to_torch=True):
    img_obs, pose_obs, img_query, pose_query = None, None, None, None
    for i in range(batch_size):
        batch_id = np.random.randint(0, color.shape[0])
        obs_id = np.random.randint(0, color.shape[1], size=obs_size)
        query_id = np.random.randint(0, color.shape[1])
        
        if img_obs is None:
            img_obs = color[batch_id:batch_id+1, obs_id].reshape(-1,color.shape[-3], color.shape[-2], color.shape[-1])
            pose_obs = pose[batch_id:batch_id+1, obs_id].reshape(-1,pose.shape[-1])
            img_query = color[batch_id:batch_id+1, query_id]
            pose_query = pose[batch_id:batch_id+1, query_id]
        else:
            img_obs = np.concatenate([img_obs, color[batch_id:batch_id+1, obs_id].reshape(-1,color.shape[-3], color.shape[-2], color.shape[-1])], 0)
            pose_obs = np.concatenate([pose_obs, pose[batch_id:batch_id+1, obs_id].reshape(-1,pose.shape[-1])], 0)
            img_query = np.concatenate([img_query, color[batch_id:batch_id+1, query_id]], 0)
            pose_query = np.concatenate([pose_query, pose[batch_id:batch_id+1, query_id]], 0)   

    if to_torch:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_obs = torch.FloatTensor(img_obs).permute(0,3,1,2).to(device)
        pose_obs = torch.FloatTensor(pose_obs).to(device)
        img_query = torch.FloatTensor(img_query).permute(0,3,1,2).to(device)
        pose_query = torch.FloatTensor(pose_query).to(device)

    return img_obs, pose_obs, img_query, pose_query

def get_batch_depth(color, pose, depth, obs_size=12, batch_size=32, to_torch=True):
    img_obs, pose_obs, depth_obs, img_query, pose_query, depth_query = None, None, None, None, None, None
    for i in range(batch_size):
        batch_id = np.random.randint(0, color.shape[0])
        obs_id = np.random.randint(0, color.shape[1], size=obs_size)
        query_id = np.random.randint(0, color.shape[1])
        
        if img_obs is None:
            img_obs = color[batch_id:batch_id+1, obs_id].reshape(-1,color.shape[-3], color.shape[-2], color.shape[-1])
            pose_obs = pose[batch_id:batch_id+1, obs_id].reshape(-1,pose.shape[-1])
            depth_obs = depth[batch_id:batch_id+1, obs_id].reshape(-1, 1, depth.shape[-2], depth.shape[-1])
            img_query = color[batch_id:batch_id+1, query_id]
            pose_query = pose[batch_id:batch_id+1, query_id]
            depth_query = depth[batch_id:batch_id+1, query_id].reshape(-1, 1, depth.shape[-2], depth.shape[-1])
        else:
            img_obs = np.concatenate([img_obs, color[batch_id:batch_id+1, obs_id].reshape(-1,color.shape[-3], color.shape[-2], color.shape[-1])], 0)
            pose_obs = np.concatenate([pose_obs, pose[batch_id:batch_id+1, obs_id].reshape(-1,pose.shape[-1])], 0)
            depth_obs = np.concatenate([depth_obs, depth[batch_id:batch_id+1, obs_id].reshape(-1, 1, depth.shape[-2], depth.shape[-1])], 0)
            img_query = np.concatenate([img_query, color[batch_id:batch_id+1, query_id]], 0)
            pose_query = np.concatenate([pose_query, pose[batch_id:batch_id+1, query_id]], 0)   
            depth_query = np.concatenate([depth_query, depth[batch_id:batch_id+1, query_id].reshape(-1, 1, depth.shape[-2], depth.shape[-1])], 0)   

    if to_torch:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_obs = torch.FloatTensor(img_obs).permute(0,3,1,2).to(device)
        pose_obs = torch.FloatTensor(pose_obs).to(device)
        depth_obs = torch.FloatTensor(depth_obs).to(device)
        img_query = torch.FloatTensor(img_query).permute(0,3,1,2).to(device)
        pose_query = torch.FloatTensor(pose_query).to(device)
        depth_query = torch.FloatTensor(depth_query).to(device)

    return img_obs, pose_obs, depth_obs, img_query, pose_query, depth_query