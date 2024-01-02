import numpy as np
import random
import cv2 

##########################
# 
##########################
RENDER_MAZE = False
maze = None
room_id = 1
def gen_recursive(x,y,room_max,prob):
    global maze, RENDER_MAZE, room_id
    # Check Space
    if x[1]-x[0]<=1 or y[1]-y[0]<=1:
        return
    
    gen_room = np.random.rand() < prob 
    if x[1]-x[0]>room_max[0] or y[1]-y[0]>room_max[1] or gen_room==False:
        # Generate Wall
        wall_v = random.randrange(x[0]+1,x[1],2)
        wall_h = random.randrange(y[0]+1,y[1],2)
        maze[y[0]:y[1]+1, wall_v] = 255
        maze[wall_h, x[0]:x[1]+1] = 255
        
        # Generate Gap
        r1 = np.random.randint(0,3)
        if r1 != 0:
            r2 = random.randrange(x[0],wall_v,2)
            maze[wall_h,r2] = 0
        if r1 != 1:
            r2 = random.randrange(wall_v+1,x[1]+1,2)
            maze[wall_h,r2] = 0
        if r1 != 2:
            r2 = random.randrange(y[0],wall_h,2)
            maze[r2,wall_v] = 0
        if r1 != 3:
            r2 = random.randrange(wall_h+1,y[1]+1,2)
            maze[r2,wall_v] = 0
        
        # Draw
        if RENDER_MAZE:
            maze_re = cv2.resize(maze, (maze.shape[1]*10, maze.shape[0]*10), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("maze", 255-maze_re)
            cv2.waitKey(30)

        # Recursive
        gen_recursive((x[0],wall_v-1),(y[0],wall_h-1),room_max,prob)
        gen_recursive((wall_v+1,x[1]),(y[0],wall_h-1),room_max,prob)
        gen_recursive((x[0],wall_v-1),(wall_h+1,y[1]),room_max,prob)
        gen_recursive((wall_v+1,x[1]),(wall_h+1,y[1]),room_max,prob)
    else:
        # Generate Room
        maze[y[0]:y[1]+1, x[0]:x[1]+1] = room_id
        room_id += 1

def gen_maze(width=15, height=15, room_max=(7,7), prob=0.8):
    global maze
    # Generate Initial Map
    maze = np.zeros([height, width], dtype=np.uint8)
    maze[:,0] = 255
    maze[0,:] = 255
    maze[:,width-1] = 255
    maze[height-1,:] = 255
    # Start Recursive
    gen_recursive((1,width-2),(1,height-2),room_max,prob)
    return maze

if __name__ == "__main__":
    RENDER_MAZE = True
    for i in range(5):
        gen_maze(15,15,(7,7),0.8)
        print(maze)
        cv2.waitKey(0)