import numpy as np
import maze_env
import maze
import cv2

def gen_data_global(env, samp_size, load_path=None):
    color_list = []
    depth_list = []
    pose_list = []

    if load_path is None:
        state, info_query = env.reset()
    else:
        state, info_query = env.reset_from_file(load_path)

    for _ in range(samp_size):
        state, info = env.reset(gen_maze=False)
        color_list.append(np.expand_dims(info["color"],0))
        depth_list.append(np.expand_dims(info["depth"],0))
        pose = np.array([info["pose"][0], info["pose"][1], 0, np.sin(info["pose"][2]), np.cos(info["pose"][2]), 0, 0])
        pose_list.append(np.expand_dims(pose,0))

    color_list.append(np.expand_dims(info_query["color"],0))
    depth_list.append(np.expand_dims(info_query["depth"],0))
    pose = np.array([info_query["pose"][0], info_query["pose"][1], 0, np.sin(info_query["pose"][2]), np.cos(info_query["pose"][2]), 0, 0])
    pose_list.append(np.expand_dims(pose,0))

    color_np = np.concatenate(color_list, 0)
    depth_np = np.concatenate(depth_list, 0)
    pose_np = np.concatenate(pose_list, 0)
    return color_np, depth_np, pose_np

def gen_data_range(env, samp_range, samp_size, load_path=None):
    color_list = []
    depth_list = []
    pose_list = []

    if load_path is None:
        state, info_query = env.reset()
    else:
        state, info_query = env.reset_from_file(load_path)
    center_x, center_y = env.agent_pose[0], env.agent_pose[1]

    count = 0
    while True:
        y = center_y + (np.random.rand()*2-1) * samp_range
        x = center_x + (np.random.rand()*2-1) * samp_range
        th = np.pi * 2 * np.random.rand()
        agent_pose = (x, y, th)
        if env.maze_obj.collision_detect(agent_pose) == False:
            state, info = env.reset(gen_maze=False, init_agent_pose=agent_pose)
            color_list.append(np.expand_dims(info["color"],0))
            depth_list.append(np.expand_dims(info["depth"],0))
            pose = np.array([info["pose"][0], info["pose"][1], 0, np.sin(info["pose"][2]), np.cos(info["pose"][2]), 0, 0])
            pose_list.append(np.expand_dims(pose,0))
            count += 1
            if count >= samp_size:
                color_list.append(np.expand_dims(info_query["color"],0))
                depth_list.append(np.expand_dims(info_query["depth"],0))
                pose = np.array([info_query["pose"][0], info_query["pose"][1], 0, np.sin(info_query["pose"][2]), np.cos(info_query["pose"][2]), 0, 0])
                pose_list.append(np.expand_dims(pose,0))

                color_np = np.concatenate(color_list, 0)
                depth_np = np.concatenate(depth_list, 0)
                pose_np = np.concatenate(pose_list, 0)
                return color_np, depth_np, pose_np
            
def gen_dataset(env, scene_size, samp_range=2, samp_size=16):
    color_data, depth_data, pose_data = [], [], []
    for i in range(scene_size):
        print("\r", i+1, "/", scene_size, end="")
        if samp_range > 0:
            color_np, depth_np, pose_np = gen_data_range(env, samp_range=samp_range, samp_size=samp_size)
        else:
            color_np, depth_np, pose_np = gen_data_global(env, samp_size=samp_size)
        color_data.append(np.expand_dims(color_np, 0))
        depth_data.append(np.expand_dims(depth_np, 0))
        pose_data.append(np.expand_dims(pose_np, 0))
    
    color_data_np = np.concatenate(color_data, 0)
    depth_data_np = np.concatenate(depth_data, 0)
    pose_data_np = np.concatenate(pose_data, 0)

    return color_data_np, depth_data_np, pose_data_np

if __name__ == "__main__":
    # Select maze type.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', '-t', nargs='?', type=str, default="MazeBoardRandom", help='Maze type.')
    maze_type = parser.parse_args().type
    if maze_type == "MazeGridRoom":
        maze_obj = maze.MazeGridRoom()
    elif maze_type == "MazeGridRandom":
        maze_obj = maze.MazeGridRandom()
    elif maze_type == "MazeGridDungeon":
        maze_obj = maze.MazeGridDungeon()
    elif maze_type == "MazeBoardRoom":
        maze_obj = maze.MazeBoardRoom()
    elif maze_type == "MazeBoardRandom":
        maze_obj = maze.MazeBoardRandom()
    else:
        maze_obj = maze.MazeBoardRandom()

    # Initial Env
    import os
    save_path = "Datasets"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    res = 64
    env = maze_env.MazeBaseEnv(maze_obj, render_res=(res, res))
    dataset_path = os.path.join(save_path, maze_type)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    for i in range(100):
        color_data_np, depth_data_np, pose_data_np = gen_dataset(env, scene_size=128, samp_size=16)
        np.savez(os.path.join(dataset_path, str(i).zfill(3)+".npz"), color=color_data_np, depth=depth_data_np, pose=pose_data_np)
    