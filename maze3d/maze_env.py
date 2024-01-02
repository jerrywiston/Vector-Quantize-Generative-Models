import cv2
import glm
import numpy as np
import pyrender
import scene_render
import maze

COLLISION_SLIDE = True
FRICTION＿DISCOUNT = 0.5

class MazeBaseEnv:
    def __init__(self, maze_obj, render_res=(192,192), velocity=0.1, ang_velocity=np.pi/18, fov=80*np.pi/180, discrete_control=True):
        self.maze_obj = maze_obj
        self.render_res = render_res
        self.render_flags = pyrender.RenderFlags.SKIP_CULL_FACES | pyrender.RenderFlags.SHADOWS_DIRECTIONAL | pyrender.RenderFlags.SHADOWS_ALL | pyrender.RenderFlags.OFFSCREEN
        self.rend = pyrender.OffscreenRenderer(self.render_res[0],self.render_res[1])
        self.map_scale = 16
        self.velocity = velocity
        self.ang_velocity = ang_velocity
        self.fov = fov
        self.discrete_control = discrete_control
        
    def _gen_scene(self):
        floor_list, wall_list, obj_list = self.maze_obj.parse()
        self.scene = scene_render.gen_scene(floor_list, wall_list, obj_list)
        camera = pyrender.PerspectiveCamera(yfov=self.fov, aspectRatio=1.0)
        self.camera_node = pyrender.Node(camera=camera)
        self.scene.add_node(self.camera_node)
    
    def save_level(self, path):
        np.savez(path, \
            maze = self.maze_obj.maze, \
            maze_type = self.maze_obj.maze_type, \
            init_pose = self.traj[0]["pose"], \
        )

    def draw_traj(self, map_img, record=True):
        if record:  # Only draw the last step.
            map_img = cv2.flip(map_img.copy(), 0)
            if not hasattr(self, "map_traj_record") or len(self.traj) <= 1:
                self.map_traj_record = -1 * np.ones_like(map_img)
            if len(self.traj) >= 2:
                x1 = int(self.map_scale*self.traj[-2]["pose"][0])
                y1 = int(self.map_scale*self.traj[-2]["pose"][1])
                x2 = int(self.map_scale*self.traj[-1]["pose"][0])
                y2 = int(self.map_scale*self.traj[-1]["pose"][1])
                if "value" not in self.traj[-1]:
                    cv2.line(self.map_traj_record, (x1,y1), (x2,y2), (0,255,0), 1)
                else:
                    value = np.array(self.traj[-1]["value"]*255).astype(np.uint8)
                    color = cv2.applyColorMap(value, cv2.COLORMAP_JET)[0,0]
                    color = (int(color[0]), int(color[1]), int(color[2]))
                    cv2.line(self.map_traj_record, (x1,y1), (x2,y2), color, 1)
                map_img[self.map_traj_record!=-1] = self.map_traj_record[self.map_traj_record!=-1]
            map_img = cv2.flip(map_img, 0)

        else:   # Re-draw every times.
            map_img = cv2.flip(map_img.copy(), 0)
            for i in range(len(self.traj)-1):
                x1 = int(self.map_scale*self.traj[i]["pose"][0])
                y1 = int(self.map_scale*self.traj[i]["pose"][1])
                x2 = int(self.map_scale*self.traj[i+1]["pose"][0])
                y2 = int(self.map_scale*self.traj[i+1]["pose"][1])
                if "value" not in self.traj[i+1]:
                    cv2.line(map_img, (x1,y1), (x2,y2), (0,255,0), 1)
                else:
                    value = np.array(self.traj[i+1]["value"]*255).astype(np.uint8)
                    color = cv2.applyColorMap(value, cv2.COLORMAP_JET)[0,0]
                    color = (int(color[0]), int(color[1]), int(color[2]))
                    cv2.line(map_img, (x1,y1), (x2,y2), color, 1)    
            map_img = cv2.flip(map_img, 0)

        return map_img
    
    def draw_obs(self, map_img, obs_list=[]):
        map_img = cv2.flip(map_img, 0)
        for obs in obs_list:
            x, y, th = obs[0], obs[1], obs[2]
            temp_y = int(y*self.map_scale)
            temp_x = int(x*self.map_scale)
            cv2.circle(map_img, (temp_x, temp_y), int(self.map_scale/6), (255,100,100), 3)
            temp_y2 = int((y + 0.2*np.sin(th)) * self.map_scale)
            temp_x2 = int((x + 0.2*np.cos(th)) * self.map_scale)
            cv2.line(map_img, (temp_x, temp_y), (temp_x2, temp_y2), (100,100,255), 2)
        map_img = cv2.flip(map_img, 0)
        return map_img

    def render_frame(self, toRGB=True):
        m = scene_render.get_cam_pose(self.agent_pose[0], self.agent_pose[1], self.agent_pose[2])
        self.scene.set_pose(self.camera_node, m)
        color, depth = self.rend.render(self.scene, self.render_flags)
        if toRGB:
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        return color, depth
        
    def reset(self, gen_maze=True, init_agent_pose=None):
        if gen_maze:
            self.maze_obj.generate()
            self._gen_scene()
        # Set Camera 
        if init_agent_pose is None:
            self.agent_pose = self.maze_obj.random_pose()
        else:
            self.agent_pose = init_agent_pose
        self.traj = [{"pose":self.agent_pose}]
        color, depth = self.render_frame()
        # Return State / Info
        self.state = color
        #map_img = self.draw_traj(self.maze_obj.get_map(self.agent_pose, self.map_scale))
        map_img = self.maze_obj.get_map(self.agent_pose, self.map_scale)
        self.info = {"color":color, "depth":depth, "pose":self.agent_pose, "map":map_img, "collision":False}
        return self.state, self.info
    
    def reset_from_file(self, path):
        env_info = np.load(path)
        # Check type
        if self.maze_obj.maze_type != env_info["maze_type"]:
            print("Inconsistent maze type")
            exit(0)
        # Load Information
        self.maze_obj.maze = env_info["maze"]
        self.agent_pose = env_info["init_pose"]
        #
        self._gen_scene()
        self.traj = [{"pose":self.agent_pose}]
        color, depth = self.render_frame()
        # Return State / Info
        self.state = color
        #map_img = self.draw_traj(self.maze_obj.get_map(self.agent_pose, self.map_scale))
        map_img = self.maze_obj.get_map(self.agent_pose, self.map_scale)
        self.info = {"color":color, "depth":depth, "pose":self.agent_pose, "map":map_img, "collision":False}
        return self.state, self.info
        
    def step_discrete(self, action):
        agent_pose_new = list(self.agent_pose)
        if action == 0: # Forward (W)
            agent_pose_new[0] += self.velocity*np.cos(agent_pose_new[2])
            agent_pose_new[1] += self.velocity*np.sin(agent_pose_new[2])
        
        if action == 1: # Turn Left (Q)
            agent_pose_new[2] += self.ang_velocity
        
        if action == 2: # Turn Right (E)
            agent_pose_new[2] -= self.ang_velocity

        if action == 3: # Left + Forward 
            agent_pose_new[2] += self.ang_velocity
            agent_pose_new[0] += self.velocity*np.cos(agent_pose_new[2])
            agent_pose_new[1] += self.velocity*np.sin(agent_pose_new[2])
        
        if action == 4: # Right + Forward 
            agent_pose_new[2] -= self.ang_velocity
            agent_pose_new[0] += self.velocity*np.cos(agent_pose_new[2])
            agent_pose_new[1] += self.velocity*np.sin(agent_pose_new[2])
        
        if action == 5: # Backward (S)
            agent_pose_new[0] -= self.velocity*np.cos(agent_pose_new[2])
            agent_pose_new[1] -= self.velocity*np.sin(agent_pose_new[2])

        '''
        if action == 3: # Backward (S)
            agent_pose_new[0] -= self.velocity*np.cos(agent_pose_new[2])
            agent_pose_new[1] -= self.velocity*np.sin(agent_pose_new[2])
        
        if action == 4: # Left + Forward 
            agent_pose_new[2] += self.ang_velocity
            agent_pose_new[0] += self.velocity*np.cos(agent_pose_new[2])
            agent_pose_new[1] += self.velocity*np.sin(agent_pose_new[2])
        
        if action == 5: # Right + Forward 
            agent_pose_new[2] -= self.ang_velocity
            agent_pose_new[0] += self.velocity*np.cos(agent_pose_new[2])
            agent_pose_new[1] += self.velocity*np.sin(agent_pose_new[2])
        
        if action == 6: # Shift Left (A)
            agent_pose_new[0] -= self.velocity*np.sin(agent_pose_new[2])
            agent_pose_new[1] += self.velocity*np.cos(agent_pose_new[2])
        
        if action == 7: # Shift Right (D)
            agent_pose_new[0] += self.velocity*np.sin(agent_pose_new[2])
            agent_pose_new[1] -= self.velocity*np.cos(agent_pose_new[2])
        '''
        
        collision = self.maze_obj.collision_detect(agent_pose_new)
        if not collision:
            self.agent_pose = agent_pose_new
        elif COLLISION_SLIDE:
            # Slide X
            agent_pose_new1 = list(self.agent_pose)
            agent_pose_new1[2] = agent_pose_new[2]
            agent_pose_new1[0] += FRICTION＿DISCOUNT * (agent_pose_new[0] - agent_pose_new1[0])
            collision1 = self.maze_obj.collision_detect(agent_pose_new1)
            # Slide Y
            agent_pose_new2 = list(self.agent_pose)
            agent_pose_new2[2] = agent_pose_new[2]
            agent_pose_new2[1] += FRICTION＿DISCOUNT * (agent_pose_new[1] - agent_pose_new2[1])
            collision2 = self.maze_obj.collision_detect(agent_pose_new2)
            # 
            if not collision1 and collision2:
                self.agent_pose = agent_pose_new1
            elif collision1 and not collision2:
                self.agent_pose = agent_pose_new2
            elif collision1 and collision2:
                self.agent_pose[2] = agent_pose_new[2]
            elif not collision1 and not collision2:
                r = np.random.randint(2)
                if r == 0:
                    self.agent_pose = agent_pose_new1
                else:
                    self.agent_pose = agent_pose_new2
        else:
            self.agent_pose[2] = agent_pose_new[2]

        self.traj.append({"pose":self.agent_pose})
        color, depth = self.render_frame()
        
        # Return State / Reward / Done / Info
        self.state_next = color
        #map_img = self.draw_traj(self.maze_obj.get_map(self.agent_pose, self.map_scale))
        map_img = self.maze_obj.get_map(self.agent_pose, self.map_scale)
        self.info = {"color":color, "depth":depth, "pose":self.agent_pose, "map":map_img, "collision":collision}
        return self.state_next, 0.0, False, self.info
    
    def step_continuous(self, action):
        agent_pose_new = list(self.agent_pose)
        agent_pose_new[2] += action[1] * self.ang_velocity
        agent_pose_new[0] += (action[0]+1)/2 * self.velocity*np.cos(agent_pose_new[2])
        agent_pose_new[1] += (action[0]+1)/2 * self.velocity*np.sin(agent_pose_new[2])
        
        collision = self.maze_obj.collision_detect(agent_pose_new)
        if not collision:
            self.agent_pose = agent_pose_new
        elif COLLISION_SLIDE:
            # Slide X
            agent_pose_new1 = list(self.agent_pose)
            agent_pose_new1[2] = agent_pose_new[2]
            agent_pose_new1[0] += FRICTION＿DISCOUNT * (agent_pose_new[0] - agent_pose_new1[0])
            collision1 = self.maze_obj.collision_detect(agent_pose_new1)
            # Slide Y
            agent_pose_new2 = list(self.agent_pose)
            agent_pose_new2[2] = agent_pose_new[2]
            agent_pose_new2[1] += FRICTION＿DISCOUNT * (agent_pose_new[1] - agent_pose_new2[1])
            collision2 = self.maze_obj.collision_detect(agent_pose_new2)
            # 
            if not collision1 and collision2:
                self.agent_pose = agent_pose_new1
            elif collision1 and not collision2:
                self.agent_pose = agent_pose_new2
            elif collision1 and collision2:
                self.agent_pose[2] = agent_pose_new[2]
            elif not collision1 and not collision2:
                r = np.random.randint(2)
                if r == 0:
                    self.agent_pose = agent_pose_new1
                else:
                    self.agent_pose = agent_pose_new2
        else:
            self.agent_pose[2] = agent_pose_new[2]

        self.traj.append({"pose":self.agent_pose})
        color, depth = self.render_frame()
        
        # Return State / Reward / Done / Info
        self.state_next = color
        map_img = self.maze_obj.get_map(self.agent_pose, self.map_scale)
        self.info = {"color":color, "depth":depth, "pose":self.agent_pose, "map":map_img, "collision":collision}
        return self.state_next, 0.0, False, self.info
    
    def step(self, action):
        if self.discrete_control:
            state_next, reward, done, info = self.step_discrete(action)
        else:
            state_next, reward, done, info = self.step_continuous(action)
        return state_next, reward, done, info

    def render(self, res=(192,192), obs_list=[], render_depth=False, show=True):
        color = cv2.resize(self.info["color"], res, interpolation=cv2.INTER_NEAREST) 
        depth = (cv2.resize(self.info["depth"]/5, res, res, interpolation=cv2.INTER_NEAREST) * 255)
        depth[depth>255] = 255
        depth = depth.astype(np.uint8)[...,np.newaxis]
        depth = np.concatenate([depth, depth, depth], 2)
        # Draw Obs
        map_img = self.draw_obs(self.info["map"], obs_list)
        map_img = self.draw_traj(map_img, record=True)
        #map_img = self.draw_obs(self.info["map"], obs_list)
        #
        map_img = cv2.resize(map_img, res, res, interpolation=cv2.INTER_LINEAR)
        if render_depth:
            render_img = cv2.hconcat([color, depth, map_img])
        else:
            render_img = cv2.hconcat([color, map_img])
        if show:
            cv2.imshow("MazeEnv", render_img)
        return render_img

class MazeNavEnv(MazeBaseEnv):
    def __init__(self, maze_obj, render_res=(192,192), velocity=0.1, ang_velocity=np.pi/18, fov=80*np.pi/180, discrete_control=True):
        super().__init__(maze_obj, render_res, velocity, ang_velocity, fov, discrete_control)
        
    def _gen_scene(self):
        floor_list, wall_list, obj_list = self.maze_obj.parse()
        obj_list.append({"voff":(self.nav_target[0], self.nav_target[1], 0.5), "color_id":0, "mesh_id":2, "scale":0.5})
        self.scene = scene_render.gen_scene(floor_list, wall_list, [])
        self.obj_node = scene_render.gen_obj_mesh_node(obj_list)
        camera = pyrender.PerspectiveCamera(yfov=self.fov, aspectRatio=1.0)
        self.camera_node = pyrender.Node(camera=camera)
        self.scene.add_node(self.obj_node)
        self.scene.add_node(self.camera_node)
    
    def _draw_nav_target(self, map_img):
        map_img = cv2.flip(map_img.copy(), 0)
        x = int(self.map_scale*self.nav_target[0])
        y = int(self.map_scale*self.nav_target[1])
        cv2.circle(map_img, (x,y), int(0.2*self.map_scale), (255,255,0), 2)
        map_img = cv2.flip(map_img, 0)
        return map_img
    
    def save_level(self, path):
        np.savez(path, \
            maze = self.maze_obj.maze, \
            maze_type = self.maze_obj.maze_type, \
            init_pose = self.traj[0]["pose"], \
            nav_target = self.nav_target, \
        )

    def reset(self, gen_maze=True, change_nav_target=True, init_agent_pose=None):
        if gen_maze:
            self.maze_obj.generate()
        
        if change_nav_target or gen_maze or not hasattr(self, "nav_target"):
            self.nav_target = self.maze_obj.random_pose()

        self._gen_scene()

        # Set Camera 
        if init_agent_pose is None:
            self.agent_pose = self.maze_obj.random_pose()
        else:
            self.agent_pose = init_agent_pose
        self.traj = [{"pose":self.agent_pose}]
        color, depth = self.render_frame()
        # Return State / Info
        self.state = color
        map_img = self.draw_traj(self.maze_obj.get_map(self.agent_pose, self.map_scale))
        map_img = self._draw_nav_target(map_img)
        self.info = {"color":color, "depth":depth, "pose":self.agent_pose, "map":map_img, "target":self.nav_target, "goal":False, "collision":False}
        return self.state, self.info

    def reset_from_file(self, path):
        env_info = np.load(path)
        # Check type
        if self.maze_obj.maze_type != env_info["maze_type"]:
            print("Inconsistent maze type")
            exit(0)
        # Load Information
        self.maze_obj.maze = env_info["maze"]
        self.agent_pose = env_info["init_pose"]
        self.nav_target = env_info["nav_target"]
        self._gen_scene()
        #
        self.traj = [{"pose":self.agent_pose}]
        color, depth = self.render_frame()
        # Return State / Info
        self.state = color
        map_img = self.draw_traj(self.maze_obj.get_map(self.agent_pose, self.map_scale))
        map_img = self._draw_nav_target(map_img)
        self.info = {"color":color, "depth":depth, "pose":self.agent_pose, "map":map_img, "target":self.nav_target, "goal":False, "collision":False}
        return self.state, self.info

    def step(self, action):
        state_next, reward, done, info = super().step(action)
        self.info["map"] = self._draw_nav_target(self.info["map"])
        dist = (self.agent_pose[0] - self.nav_target[0])**2 + (self.agent_pose[1] - self.nav_target[1])**2
        dist = np.sqrt(dist)
        self.info["target"] = self.nav_target
        if dist < 0.5:
            done = True
            reward = 1.0
            self.info["goal"] = True
        else:
            self.info["goal"] = False
        return self.state_next, reward, done, self.info 

class MazeItemEnv(MazeBaseEnv):
    def __init__(self, maze_obj, render_res=(192,192), velocity=0.1, ang_velocity=np.pi/18, fov=80*np.pi/180, discrete_control=True, n_items=15):
        super().__init__(maze_obj, render_res, velocity, ang_velocity, fov, discrete_control)
        self.n_items = n_items
        self.score = 0

    def _gen_scene(self):
        # Gen Scene
        floor_list, wall_list, obj_list = self.maze_obj.parse()
        self.scene = scene_render.gen_scene(floor_list, wall_list, [])
        # Gen Obj Node
        obj_list = []
        for i in self.items:
            obj_list.append({"voff":(self.items[i][0], self.items[i][1], 0.5), "color_id":0, "mesh_id":2, "scale":0.5})
            #obj_list.append({"voff":(self.items[i][0], self.items[i][1], 0.5), "color_id":4, "mesh_id":1, "scale":0.5})
        self.obj_node = scene_render.gen_obj_mesh_node(obj_list)
        # Gen Camera Node
        camera = pyrender.PerspectiveCamera(yfov=self.fov, aspectRatio=1.0)
        self.camera_node = pyrender.Node(camera=camera)
        # Add Node to Scene
        if self.obj_node is not None:
            self.scene.add_node(self.obj_node)
        self.scene.add_node(self.camera_node)
    
    def _update_scene_items(self):
        self.scene.remove_node(self.obj_node)
        obj_list = []
        for i in self.items:
            obj_list.append({"voff":(self.items[i][0], self.items[i][1], 0.5), "color_id":0, "mesh_id":2, "scale":0.5})
        self.obj_node = scene_render.gen_obj_mesh_node(obj_list)
        if self.obj_node is not None:
            self.scene.add_node(self.obj_node)
    
    def _draw_map_items(self, map_img):
        map_img = cv2.flip(map_img.copy(), 0)
        for i in self.items:
            x = int(self.map_scale*self.items[i][0])
            y = int(self.map_scale*self.items[i][1])
            cv2.circle(map_img, (x,y), int(0.2*self.map_scale), (255,255,0), 2)
        map_img = cv2.flip(map_img, 0)
        return map_img
    
    def save_level(self, path):
        np.savez(path, \
            maze = self.maze_obj.maze, \
            maze_type = self.maze_obj.maze_type, \
            init_pose = self.traj[0]["pose"], \
            items = self.items, \
        )

    def reset(self, gen_maze=True, change_items=True, init_agent_pose=None):
        self.score = 0
        if gen_maze:
            self.maze_obj.generate()
        
        # Set Camera 
        if init_agent_pose is None:
            self.agent_pose = self.maze_obj.random_pose()
        else:
            self.agent_pose = init_agent_pose
        self.traj = [{"pose":self.agent_pose}]
        
        # Set items
        if change_items or gen_maze or not hasattr(self, "items"):
            self.items = {}
            for i in range(self.n_items):
                while True:
                    resample = False
                    temp_pos = self.maze_obj.random_pose()
                    # Distance between items and agent
                    dist_agent = (self.agent_pose[0] - temp_pos[0])**2 + (self.agent_pose[1] - temp_pos[1])**2
                    dist_agent  = np.sqrt(dist_agent )
                    if dist_agent < 1.0:
                        resample = True
                    # Distance between items
                    for j in self.items:
                        dist = (self.items[j][0] - temp_pos[0])**2 + (self.items[j][1] - temp_pos[1])**2
                        dist = np.sqrt(dist)
                        if dist < 1.0:
                            resample = True
                    if not resample:
                        break
                self.items[i] = temp_pos
        
        # Generate Scene and State
        self._gen_scene()
        color, depth = self.render_frame()
        
        # Return State / Info
        self.state = color
        map_img = self.draw_traj(self.maze_obj.get_map(self.agent_pose, self.map_scale))
        map_img = self._draw_map_items(map_img)
        self.info = {"color":color, "depth":depth, "pose":self.agent_pose, "map":map_img, \
            "items":self.items, "goal":False, "score":0, "collision":False}
        return self.state, self.info

    def reset_from_file(self, path):
        self.score = 0
        env_info = np.load(path, allow_pickle=True)
        # Check type
        if self.maze_obj.maze_type != env_info["maze_type"]:
            print("Inconsistent maze type")
            exit(0)
        # Load Information
        self.maze_obj.maze = env_info["maze"]
        self.agent_pose = env_info["init_pose"]
        items = env_info["items"][()]
        self.items = {}
        for i in range(self.n_items):
            self.items[i] = items[i]
        self._gen_scene()
        #
        self.traj = [{"pose":self.agent_pose}]
        color, depth = self.render_frame()
        # Return State / Info
        self.state = color
        map_img = self.draw_traj(self.maze_obj.get_map(self.agent_pose, self.map_scale))
        map_img = self._draw_map_items(map_img)
        self.info = {"color":color, "depth":depth, "pose":self.agent_pose, "map":map_img, \
            "items":self.items, "goal":False, "score":0, "collision":False}
        return self.state, self.info

    def step(self, action):
        state_next, reward, done, info = super().step(action)
        self.info["map"] = self._draw_map_items(self.info["map"])
        self.info["goal"] = False
        pop_list = []
        reward = 0.0
        for i in self.items:
            dist = (self.agent_pose[0] - self.items[i][0])**2 + (self.agent_pose[1] - self.items[i][1])**2
            dist = np.sqrt(dist)
            if dist < 0.5:
                pop_list.append(i)
                reward += 1.0
                self.score += 1
                self.info["goal"] = True
        for key in pop_list:
            self.items.pop(key)
        if len(pop_list) > 0:
            self._update_scene_items()
        self.info["items"] = self.items
        self.info["score"] = self.score
            
        return self.state_next, reward, done, self.info 

if __name__ == "__main__":
    # Select maze type.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', '-t', nargs='?', type=str, default="MazeGridDungeon2", help='Maze type.')
    maze_type = parser.parse_args().type
    if maze_type == "MazeGridRoom":
        maze_obj = maze.MazeGridRoom()
    elif maze_type == "MazeGridRandom":
        maze_obj = maze.MazeGridRandom2(size=(7,7), room_total=3)
    elif maze_type == "MazeGridDungeon":
        maze_obj = maze.MazeGridDungeon()
    elif maze_type == "MazeGridDungeon2":
        maze_obj = maze.MazeGridDungeon2()
    elif maze_type == "MazeBoardRoom":
        maze_obj = maze.MazeBoardRoom()
    elif maze_type == "MazeBoardRandom":
        maze_obj = maze.MazeBoardRandom()
    else:
        maze_obj = maze.MazeGridRandom()

    # Initial Env
    #env = MazeBaseEnv(maze_obj, render_res=(64,64), velocity=0.4, ang_velocity=np.pi/18*4)
    fov=90*np.pi/180
    #env = MazeNavEnv(maze_obj, render_res=(64,64), fov=fov)
    env = MazeItemEnv(maze_obj, render_res=(64,64), fov=fov, n_items=1)
    state, info = env.reset()
    env.render()

    while(True):
        # Control Handle
        k = cv2.waitKey(0)
        run_step = False
        # Exit
        if k == 27:
            break
        # Reset Maze
        if k == 13:
            state, info = env.reset()
            env.render()
            continue
        # Reset Pose
        if k == 32:
            state, info = env.reset(gen_maze=False)
            env.render()
            continue
        if k == ord('w'):
            action = 0
            run_step = True
        if k == ord('s'):
            action = 3
            run_step = True
        if k == ord('q'):
            action = 1
            run_step = True
        if k == ord('e'):
            action = 2
            run_step = True
        if k == ord('a'):
            action = 6
            run_step = True
        if k == ord('d'):
            action = 7
            run_step = True
        
        if run_step:
            state_next, reward, done, info = env.step(action)
            print("\r", info["pose"], reward, done, end="\t")
            #print(info["score"])
            #cv2.imshow("test", state_next)
            env.render()
            state = state_next.copy()

            if done:
                state, info = env.reset()
        