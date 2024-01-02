import numpy as np
import random
import cv2 

##########################
# 
##########################
RENDER_MAZE = False
maze = None
def gen_recursive(x,y):
    global maze, RENDER_MAZE
    # Check Space
    if x[1]-x[0]<=1 or y[1]-y[0]<=1:
        return
    
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
    gen_recursive((x[0],wall_v-1),(y[0],wall_h-1))
    gen_recursive((wall_v+1,x[1]),(y[0],wall_h-1))
    gen_recursive((x[0],wall_v-1),(wall_h+1,y[1]))
    gen_recursive((wall_v+1,x[1]),(wall_h+1,y[1]))

def gen_room(room_total=9, room_types=5):
    global maze
    for i in range(1,room_total+1):
        rx = np.random.randint(1,maze.shape[1]-2)
        ry = np.random.randint(1,maze.shape[0]-2)
        if rx % 2 == 0:
            rx -= 1
        if ry % 2 == 0:
            ry -= 1
        if room_types is not None:
            maze[ry:ry+3, rx:rx+3] = i%5
        else:
            maze[ry:ry+3, rx:rx+3] = i

def gen_maze(width=15, height=15, room_total=11):
    global maze
    # Generate Initial Map
    maze = np.zeros([height, width], dtype=np.uint8)
    maze[:,0] = 255
    maze[0,:] = 255
    maze[:,width-1] = 255
    maze[height-1,:] = 255
    # Start Recursive
    gen_recursive((1,width-2),(1,height-2))
    gen_room(room_total)
    return maze

if __name__ == "__main__":
    #RENDER_MAZE = True
    for i in range(5):
        maze = gen_maze(15,15)
        maze_re = cv2.resize(maze, (maze.shape[1]*10, maze.shape[0]*10), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("maze", 255-maze_re)
        print(maze)
        cv2.waitKey(0)