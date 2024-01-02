##############################################
# Multiple textures, solve the sampling problem.
##############################################
import os
import time
import random
import trimesh
import pyrender
import numpy as np
import glm
import cv2
import structure
import obj_loader

def add_struct(v, vn, f, vt, foff, struct_obj):
    struct_obj['v'] += v
    struct_obj['vn'] += vn
    struct_obj['f'] += f
    struct_obj['vt'] += vt
    struct_obj['foff'] += foff
    return struct_obj

def read_texture(path, flist, size=256):
    img_tex = None
    for fname in flist:
        img = cv2.imread(os.path.join(path,fname))
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        if img_tex is None:
            # Duplicate to solve the sampling problem.
            img_tex = cv2.hconcat([img, img, img])
        else:
            # Combine multiple textures.
            img_tex = cv2.hconcat([img_tex, img, img, img])

    img_tex = cv2.cvtColor(img_tex, cv2.COLOR_RGB2BGR)
    return img_tex

def gen_scene_mesh(floor_list, wall_list, obj_list, tex_shuffle=True):
    my_path = os.path.abspath(os.path.dirname(__file__))

    path_floor = os.path.join(my_path, './resource/texture/floor')
    flist_floor = os.listdir(path_floor)
    if tex_shuffle:
        random.shuffle(flist_floor)

    path_wall = os.path.join(my_path, './resource/texture/wall')
    flist_wall = os.listdir(path_wall)
    if tex_shuffle:
        random.shuffle(flist_wall)

    path_obj = os.path.join(my_path, './resource/texture/obj')
    flist_obj = os.listdir(path_obj)
    path_mesh = os.path.join(my_path, './resource/mesh')
    flist_mesh = os.listdir(path_mesh)

    struct_floor = {'v':[], 'vn':[], 'f':[], 'vt':[], 'foff':0}
    struct_wall = {'v':[], 'vn':[], 'f':[], 'vt':[], 'foff':0}
    struct_obj = {'v':[], 'vn':[], 'f':[], 'vt':[], 'foff':0}

    # Build Floor
    for data in floor_list:
        v, vn, f, vt, foff = structure.get_struct_floor(data["voff"], struct_floor['foff'], data["id"], len(flist_floor))
        struct_floor = add_struct(v, vn, f, vt, foff, struct_floor)
    
    # Build Wall
    for data in wall_list:
        v, vn, f, vt, foff = structure.get_struct_wall(data["type"], data["voff"], struct_wall['foff'], data["id"], len(flist_wall))
        struct_wall = add_struct(v, vn, f, vt, foff, struct_wall)
    
    # Build Object
    for data in obj_list:
        if "mesh_id" in data:
            mesh_id = data["mesh_id"]
        else:
            mesh_id = np.random.randint(len(flist_mesh))

        if "color_id" in data:
            color_id = data["color_id"]
        else:
            color_id = np.random.randint(len(flist_obj))
        
        if "scale" in data:
            scale = data["scale"]
        else:
            scale = np.random.uniform(0.2,0.6)
            
        v, vn, f, vt, foff = obj_loader.load_(os.path.join(path_mesh, flist_mesh[mesh_id]), \
                            data["voff"], struct_obj['foff'], scale, len(flist_obj), color_id)
        struct_obj = add_struct(v, vn, f, vt, foff, struct_obj)

    tex_size = 256
    # Texture Floor
    image_floor = read_texture(path_floor, flist_floor, tex_size)
    material_floor = trimesh.visual.texture.SimpleMaterial(image=image_floor)
    color_visuals_floor = trimesh.visual.TextureVisuals(uv=struct_floor['vt'], image=image_floor, material=material_floor)
    # Mesh Floor
    mesh_floor = trimesh.Trimesh(vertices=struct_floor['v'], vertex_normals=struct_floor['vn'], faces=struct_floor['f'], visual=color_visuals_floor)
    mesh_floor_pr = pyrender.Mesh.from_trimesh(mesh_floor)
    
    # Texture Wall
    image_wall = read_texture(path_wall, flist_wall, tex_size)
    material_wall = trimesh.visual.texture.SimpleMaterial(image=image_wall)
    color_visuals_wall = trimesh.visual.TextureVisuals(uv=struct_wall['vt'], image=image_wall, material=material_wall)
    # Mesh Wall
    mesh_wall = trimesh.Trimesh(vertices=struct_wall['v'], vertex_normals=struct_wall['vn'], faces=struct_wall['f'], visual=color_visuals_wall)
    mesh_wall_pr = pyrender.Mesh.from_trimesh(mesh_wall)
    
    if len(obj_list) > 0:
        # Texture Object
        image_obj = read_texture(path_obj, flist_obj, 256)
        material_obj = trimesh.visual.texture.SimpleMaterial(image=image_obj)
        color_visuals_obj = trimesh.visual.TextureVisuals(uv=struct_obj['vt'], image=image_obj, material=material_obj)
        # Mesh Object
        mesh_obj = trimesh.Trimesh(vertices=struct_obj['v'], faces=struct_obj['f'], visual=color_visuals_obj)
        mesh_obj_pr = pyrender.Mesh.from_trimesh(mesh_obj)
    else:
        mesh_obj_pr = None

    return mesh_floor_pr, mesh_wall_pr, mesh_obj_pr

def gen_obj_mesh_node(obj_list):
    if len(obj_list) == 0:
        return None
    my_path = os.path.abspath(os.path.dirname(__file__))

    path_obj = os.path.join(my_path, './resource/texture/obj')
    flist_obj = os.listdir(path_obj)
    path_mesh = os.path.join(my_path, './resource/mesh')
    flist_mesh = os.listdir(path_mesh)

    struct_obj = {'v':[], 'vn':[], 'f':[], 'vt':[], 'foff':0}

    # Build Object
    for data in obj_list:
        if "mesh_id" in data:
            mesh_id = data["mesh_id"]
        else:
            mesh_id = np.random.randint(len(flist_mesh))

        if "color_id" in data:
            color_id = data["color_id"]
        else:
            color_id = np.random.randint(len(flist_obj))
        
        if "scale" in data:
            scale = data["scale"]
        else:
            scale = np.random.uniform(0.2,0.6)
            
        v, vn, f, vt, foff = obj_loader.load_(os.path.join(path_mesh, flist_mesh[mesh_id]), \
                            data["voff"], struct_obj['foff'], scale, len(flist_obj), color_id)
        struct_obj = add_struct(v, vn, f, vt, foff, struct_obj)
    
    image_obj = read_texture(path_obj, flist_obj, 256)
    material_obj = trimesh.visual.texture.SimpleMaterial(image=image_obj)
    color_visuals_obj = trimesh.visual.TextureVisuals(uv=struct_obj['vt'], image=image_obj, material=material_obj)
    # Mesh Object
    mesh_obj = trimesh.Trimesh(vertices=struct_obj['v'], faces=struct_obj['f'], visual=color_visuals_obj)
    mesh_obj_pr = pyrender.Mesh.from_trimesh(mesh_obj)
    mesh_obj_node = pyrender.Node(mesh = mesh_obj_pr)
    return mesh_obj_node

def gen_scene(floor_list, wall_list, obj_list):
    # Initialize Scene
    amb_intensity = 1.0#0.2
    bg_color = np.array([160,200,255,0])
    scene = pyrender.Scene(ambient_light=amb_intensity*np.ones(3), bg_color=bg_color)
    
    # Generate Maze 
    mesh_floor_pr, mesh_wall_pr, mesh_obj_pr = gen_scene_mesh(floor_list, wall_list, obj_list)
    scene.add(mesh_floor_pr)
    scene.add(mesh_wall_pr)
    if mesh_obj_pr is not None:
        scene.add(mesh_obj_pr)
    
    # Add Light 
    dir_light = pyrender.DirectionalLight(color=np.ones(3), intensity=3)
    #m = glm.mat4_cast(glm.quat(glm.vec3(0.5,0.4,np.pi/2)))
    #m = glm.mat4_cast(glm.quat(glm.vec3(0.8,0,np.pi/2)))
    m = glm.mat4_cast(glm.quat(glm.vec3(0.2,0.2,np.pi/2)))
    light_node = pyrender.Node(light=dir_light, matrix=m)
    #scene.add_node(light_node)

    return scene

def get_cam_pose(x, y, th):
    r = glm.mat4_cast(glm.quat(glm.vec3(-np.pi/2,np.pi/2-th,0)))
    t = glm.translate(glm.mat4(1), glm.vec3(x,y,0.5))
    m = r * glm.transpose(t)
    m = np.array(m).transpose() #Add transpose if use new version GLM library.
    return m

#############################################
def run_dataset(rend, scene, maze, view_size=4., gen_size=16, show=False):
    images = []
    depths = []
    poses = []

    # Set Camera 
    camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0, aspectRatio=1.0)
    camera_node = pyrender.Node(camera=camera)
    scene.add_node(camera_node)

    flags = pyrender.RenderFlags.SKIP_CULL_FACES | pyrender.RenderFlags.SHADOWS_DIRECTIONAL
    count = 0
    x_min = np.random.uniform(0,float(maze.shape[1]-view_size))
    y_min = np.random.uniform(0,float(maze.shape[0]-view_size))
    #print(x_min, y_min)
    while True:
        x = np.random.uniform(x_min, x_min + view_size)
        y = np.random.uniform(y_min, y_min + view_size)
        th = np.random.uniform(0,np.pi*2)
        agent_info = {"x":x, "y":y, "theta":th}
        if not maze.collision_detect(agent_info):
            r = glm.mat4_cast(glm.quat(glm.vec3(-np.pi/2,np.pi/2-th,0)))
            t = glm.translate(glm.mat4(1), glm.vec3(x,y,0.5))
            m = r * glm.transpose(t)
            m = np.array(m)
            scene.set_pose(camera_node, m)
            color, depth = rend.render(scene, flags)
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

            images.append(color[np.newaxis,...])
            depths.append(depth[np.newaxis,...])
            poses.append(np.array([1, x, y, th]))

            if show:
                print("\rx={:.3f}, y={:.3f}, th={:.3f}\t".format(x, y, th*180/np.pi), end="")
                maze_draw = maze.get_map(agent_info)
                cv2.imshow("maze", maze_draw)
                cv2.imshow("color", color)
                cv2.waitKey(0)
            count += 1
        if count >= gen_size:
            break
    
    images = np.concatenate(images, 0)
    depths = np.concatenate(depths, 0)
    poses = np.concatenate(poses, 0)
    return images, depths, poses

def run_viewer(scene):
    render_flags = { \
        "flip_wireframe":False, #default:False
        "all_wireframe":False,  #default:False
        "all_solid":False,      #default:False
        "shadows":True,         #default:False
        "face_normals":False,   #default:False
        "cull_faces":True,     #default:True
        "point_size":1,         #default:1
    }
    pyrender.Viewer(scene, render_flags=render_flags, use_raymond_lighting=False)

def run_control(scene, maze):
    # Set Camera 
    agent_info = maze.random_pose()#{"x":1.5, "y":1.5, "theta":0}
    camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0, aspectRatio=1.0)
    camera_node = pyrender.Node(camera=camera)
    scene.add_node(camera_node)

    # Off-Screen Render
    render_frame = True
    render_res = (192, 192)
    flags = pyrender.RenderFlags.SKIP_CULL_FACES | pyrender.RenderFlags.SHADOWS_DIRECTIONAL | pyrender.RenderFlags.SHADOWS_ALL
    #flags = pyrender.RenderFlags.SHADOWS_DIRECTIONAL
    rend = pyrender.OffscreenRenderer(render_res[0],render_res[1])
    while(True):
        # Control Handling
        agent_info_new = agent_info.copy()
        k = cv2.waitKey(1)
        if k == 27:
            break
        if k == ord('a'):
            agent_info_new["x"] -= 0.1*np.sin(agent_info_new["theta"])
            agent_info_new["y"] += 0.1*np.cos(agent_info_new["theta"])
            render_frame = True
        if k == ord('d'):
            agent_info_new["x"] += 0.1*np.sin(agent_info_new["theta"])
            agent_info_new["y"] -= 0.1*np.cos(agent_info_new["theta"])
            render_frame = True
        if k == ord('w'):
            agent_info_new["x"] += 0.1*np.cos(agent_info_new["theta"])
            agent_info_new["y"] += 0.1*np.sin(agent_info_new["theta"])
            render_frame = True
        if k == ord('s'):
            agent_info_new["x"] -= 0.1*np.cos(agent_info_new["theta"])
            agent_info_new["y"] -= 0.1*np.sin(agent_info_new["theta"])
            render_frame = True
        if k == ord('q'):
            agent_info_new["theta"] += np.pi/18
            render_frame = True
        if k == ord('e'):
            agent_info_new["theta"] -= np.pi/18
            render_frame = True
        
        # Render Agent View
        if render_frame:
            # Collision Detection
            if not maze.collision_detect(agent_info_new):
                agent_info = agent_info_new

            # Rendering
            start = time.time()
            m = get_cam_pose(agent_info['x'], agent_info['y'], agent_info['theta'])
            scene.set_pose(camera_node, m)
            color, depth = rend.render(scene, flags)
            end = time.time()

            print("\rx={:.3f}, y={:.3f}, th={:.3f}, time={:.3f}\t"\
                .format(agent_info['x'], agent_info['y'], agent_info['theta']*180/np.pi, end - start), end="")

            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            cv2.imshow("camera", color)
            cv2.imshow("depth", depth/5)
            render_frame = False
        
        # Draw Maze Map
        maze_draw = maze.get_map(agent_info)
        cv2.imshow("maze", maze_draw)

    print()

if __name__ == "__main__":
    import maze
    #maze_obj = maze.MazeGridRoom()
    #maze_obj = maze.MazeGridRandom()
    #maze_obj = maze.MazeGridDungeon()
    #maze_obj = maze.MazeBoardRoom()
    maze_obj = maze.MazeBoardRandom()

    maze_obj.generate()
    floor_list, wall_list, obj_list = maze_obj.parse()
    scene = gen_scene(floor_list, wall_list, obj_list)
    #run_viewer(scene)
    run_control(scene, maze_obj) 
