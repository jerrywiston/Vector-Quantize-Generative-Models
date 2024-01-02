import trimesh
import pyrender
import numpy as np
import glm

def get_struct_floor(v_offset, f_offset, tex_id=0, tex_num=1):
    v = [[0.0+v_offset[0], 0.0+v_offset[1], 0.0], [1.0+v_offset[0], 0.0+v_offset[1], 0.0], 
        [0.0+v_offset[0], 1.0+v_offset[1], 0.0], [1.0+v_offset[0], 1.0+v_offset[1], 0.0]]
    vn = [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
    #f = [[0+f_offset,1+f_offset,3+f_offset], [0+f_offset,3+f_offset,2+f_offset]]
    f = [[0+f_offset,1+f_offset,3+f_offset], [0+f_offset,3+f_offset,2+f_offset],
        [0+f_offset,3+f_offset,1+f_offset], [0+f_offset,2+f_offset,3+f_offset]]
    temp1 = (1+tex_id*3) / (3*tex_num)
    temp2 = (2+tex_id*3) / (3*tex_num)
    vt = [[temp1, 0.0], [temp2, 0.0], [temp1, 1.0], [temp2, 1.0]]
    return v, vn, f, vt, 4

def get_struct_ceil(v_offset, f_offset, tex_id=0, tex_num=1):
    v = [[0.0+v_offset[0], 0.0+v_offset[1], 0.0], [1.0+v_offset[0], 0.0+v_offset[1], 1.0], 
        [0.0+v_offset[0], 1.0+v_offset[1], 0.0], [1.0+v_offset[0], 1.0+v_offset[1], 1.0]]
    vn = [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
    #f = [[0+f_offset,1+f_offset,3+f_offset], [0+f_offset,3+f_offset,2+f_offset]]
    f = [[0+f_offset,1+f_offset,3+f_offset], [0+f_offset,3+f_offset,2+f_offset],
        [0+f_offset,3+f_offset,1+f_offset], [0+f_offset,2+f_offset,3+f_offset]]
    temp1 = (1+tex_id*3) / (3*tex_num)
    temp2 = (2+tex_id*3) / (3*tex_num)
    vt = [[temp1, 0.0], [temp2, 0.0], [temp1, 1.0], [temp2, 1.0]]
    return v, vn, f, vt, 4

def get_struct_wall_left(v_offset, f_offset, tex_id=0, tex_num=1):
    v = [[0.0+v_offset[0], 0.0+v_offset[1], 0.0], [0.0+v_offset[0], 0.0+v_offset[1], 1.0], 
        [0.0+v_offset[0], 1.0+v_offset[1], 0.0], [0.0+v_offset[0], 1.0+v_offset[1], 1.0]]
    vn = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    #f = [[0+f_offset,3+f_offset,1+f_offset], [0+f_offset,2+f_offset,3+f_offset]]
    f = [[0+f_offset,3+f_offset,1+f_offset], [0+f_offset,2+f_offset,3+f_offset],
        [0+f_offset,1+f_offset,3+f_offset], [0+f_offset,3+f_offset,2+f_offset]]
    temp1 = (1+tex_id*3) / (3*tex_num)
    temp2 = (2+tex_id*3) / (3*tex_num)
    vt = [[temp1, 0.0], [temp1, 1.0], [temp2, 0.0], [temp2, 1.0]]
    return v, vn, f, vt, 4

def get_struct_wall_right(v_offset, f_offset, tex_id=0, tex_num=1):
    v = [[1.0+v_offset[0], 0.0+v_offset[1], 0.0], [1.0+v_offset[0], 0.0+v_offset[1], 1.0], 
        [1.0+v_offset[0], 1.0+v_offset[1], 0.0], [1.0+v_offset[0], 1.0+v_offset[1], 1.0]]
    vn = [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    #f = [[0+f_offset,1+f_offset,3+f_offset], [0+f_offset,3+f_offset,2+f_offset]]
    f = [[0+f_offset,1+f_offset,3+f_offset], [0+f_offset,3+f_offset,2+f_offset], \
        [0+f_offset,3+f_offset,1+f_offset], [0+f_offset,2+f_offset,3+f_offset]]
    temp1 = (1+tex_id*3) / (3*tex_num)
    temp2 = (2+tex_id*3) / (3*tex_num)
    vt = [[temp2, 0.0], [temp2, 1.0], [temp1, 0.0], [temp1, 1.0]]
    return v, vn, f, vt, 4

def get_struct_wall_buttom(v_offset, f_offset, tex_id=0, tex_num=1):
    v = [[0.0+v_offset[0], 0.0+v_offset[1], 0.0], [0.0+v_offset[0], 0.0+v_offset[1], 1.0], 
        [1.0+v_offset[0], 0.0+v_offset[1], 0.0], [1.0+v_offset[0], 0.0+v_offset[1], 1.0]]
    vn = [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    #f = [[0+f_offset,1+f_offset,3+f_offset], [0+f_offset,3+f_offset,2+f_offset]]
    f = [[0+f_offset,1+f_offset,3+f_offset], [0+f_offset,3+f_offset,2+f_offset],
        [0+f_offset,3+f_offset,1+f_offset], [0+f_offset,2+f_offset,3+f_offset]]
    temp1 = (1+tex_id*3) / (3*tex_num)
    temp2 = (2+tex_id*3) / (3*tex_num)
    vt = [[temp2, 0.0], [temp2, 1.0], [temp1, 0.0], [temp1, 1.0]]
    return v, vn, f, vt, 4

def get_struct_wall_top(v_offset, f_offset, tex_id=0, tex_num=1):
    v = [[0.0+v_offset[0], 1.0+v_offset[1], 0.0], [0.0+v_offset[0], 1.0+v_offset[1], 1.0], 
        [1.0+v_offset[0], 1.0+v_offset[1], 0.0], [1.0+v_offset[0], 1.0+v_offset[1], 1.0]]
    vn = [[0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0]]
    #f = [[0+f_offset,3+f_offset,1+f_offset], [0+f_offset,2+f_offset,3+f_offset]]
    f = [[0+f_offset,3+f_offset,1+f_offset], [0+f_offset,2+f_offset,3+f_offset],
        [0+f_offset,1+f_offset,3+f_offset], [0+f_offset,3+f_offset,2+f_offset]]
    temp1 = (1+tex_id*3) / (3*tex_num)
    temp2 = (2+tex_id*3) / (3*tex_num)
    vt = [[temp1, 0.0], [temp1, 1.0], [temp2, 0.0], [temp2, 1.0]]
    return v, vn, f, vt, 4

def get_struct_wall(wall_type, v_offset, f_offset, tex_id=0, tex_num=1):
    if wall_type == "B":
        return get_struct_wall_buttom(v_offset, f_offset, tex_id, tex_num)
    elif wall_type == "T":
        return get_struct_wall_top(v_offset, f_offset, tex_id, tex_num)
    elif wall_type == "L":
        return get_struct_wall_left(v_offset, f_offset, tex_id, tex_num)
    elif wall_type == "R":
        return get_struct_wall_right(v_offset, f_offset, tex_id, tex_num)
    else:
        return None 