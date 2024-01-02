import numpy as np
import cv2

class Maze:
    def __init__(self):
        self.maze = None
        self.maze_type = "maze"

    def generate(self):
        raise NotImplementedError

    def parse(self):
        raise NotImplementedError
    
    def get_map(self):
        raise NotImplementedError
    
    def collision_detect(self):
        raise NotImplementedError
    
    def random_pose(self):
        raise NotImplementedError

#######################################
# Grid Maze
#######################################
class MazeGrid(Maze):
    def __init__(self, obj_prob=0.0):
        super().__init__()
        self.maze_type = "maze_grid"
        self.obj_prob = obj_prob

    def parse(self):
        floor_list = []
        wall_list = []
        obj_list = []
        for j in range(1,self.maze.shape[0]-1):
            for i in range(1,self.maze.shape[1]-1):
                if self.maze[j,i] < 255:
                    # Build Floor
                    floor_list.append({"voff":(i,j), "id":self.maze[j,i]})
                    # Build Wall Buttom
                    if self.maze[j-1,i]==255:
                        wall_list.append({"voff":(i,j), "id":self.maze[j,i], "type":"B"})
                    # Build Wall Top
                    if self.maze[j+1,i]==255:
                        wall_list.append({"voff":(i,j), "id":self.maze[j,i], "type":"T"})
                    # Build Wall Left
                    if self.maze[j,i-1]==255:
                        wall_list.append({"voff":(i,j), "id":self.maze[j,i], "type":"L"})
                    # Build Wall Right
                    if self.maze[j,i+1]==255:
                        wall_list.append({"voff":(i,j), "id":self.maze[j,i], "type":"R"})
                
                # Build Object
                if 0 < self.maze[j,i] < 255 and np.random.rand() < self.obj_prob:
                    obj_list.append({"voff":(i+0.5,j+0.5,0.5)})

        return floor_list, wall_list, obj_list
    
    def get_map(self, pose, map_scale=16):
        maze_re = cv2.cvtColor(self.maze, cv2.COLOR_GRAY2RGB)
        maze_re = 255-cv2.resize(maze_re, (self.maze.shape[1]*map_scale, self.maze.shape[0]*map_scale), interpolation=cv2.INTER_NEAREST)
        maze_draw = maze_re.copy()
        
        if pose is not None:
            x, y, th = pose[0], pose[1], pose[2]
            temp_y = int(y*map_scale)
            temp_x = int(x*map_scale)
            cv2.circle(maze_draw, (temp_x, temp_y), int(map_scale/5), (255,0,0), 3)
            temp_y2 = int((y + 0.3*np.sin(th)) * map_scale)
            temp_x2 = int((x + 0.3*np.cos(th)) * map_scale)
            cv2.line(maze_draw, (temp_x, temp_y), (temp_x2, temp_y2), (0,0,255), 3)
        maze_draw = cv2.flip(maze_draw,0)

        return maze_draw
    
    def collision_detect(self, pose, eps=0.15):
        collision = False
        # Check in map
        if pose[1] < 0 or pose[1] >= self.maze.shape[0]:
            return True
        if pose[0] < 0 or pose[0] >= self.maze.shape[1]:
            return True
        # Check Collision
        if self.maze[int(pose[1]),int(pose[0])] == 255:
            collision = True
        elif self.maze[int(pose[1]+eps),int(pose[0])] == 255:
            collision = True
        elif self.maze[int(pose[1]-eps),int(pose[0])] == 255:
            collision = True
        elif self.maze[int(pose[1]),int(pose[0]+eps)] == 255:
            collision = True
        elif self.maze[int(pose[1]),int(pose[0]-eps)] == 255:
            collision = True
        #
        elif self.maze[int(pose[1]+eps),int(pose[0]+eps)] == 255:
            collision = True
        elif self.maze[int(pose[1]+eps),int(pose[0]-eps)] == 255:
            collision = True
        elif self.maze[int(pose[1]-eps),int(pose[0]+eps)] == 255:
            collision = True
        elif self.maze[int(pose[1]-eps),int(pose[0]-eps)] == 255:
            collision = True
        #
        return collision
    
    def random_pose(self):
        while(True):
            y = self.maze.shape[0] * np.random.rand()
            x = self.maze.shape[1] * np.random.rand()
            th = np.pi * 2 * np.random.rand()
            pose = [x,y,th]
            if self.collision_detect(pose) == False:
                return  pose

class MazeGridRoom(MazeGrid):
    def __init__(self, obj_prob=0.0, size=(4,4)):
        super().__init__(obj_prob)
        self.maze_type = "maze_grid_room"
        self.size = size

    def generate(self):
        self.maze = np.ones((self.size[0]+2, self.size[1]+2), dtype=np.uint8)
        self.maze[:,0] = 255
        self.maze[0,:] = 255
        self.maze[self.size[0]+1,:] = 255
        self.maze[:,self.size[1]+1] = 255

class MazeGridRandom(MazeGrid):
    def __init__(self, obj_prob=0.0, size=(11,11), room_max=(5,5), room_prob=0.8):
        super().__init__(obj_prob)
        self.maze_type = "maze_grid_random"
        self.size = size
        self.room_max = room_max
        self.room_prob = room_prob

    def generate(self):
        from MazeGen import maze_gen_grid
        self.maze = maze_gen_grid.gen_maze(self.size[0],self.size[1],self.room_max,self.room_prob)

class MazeGridRandom2(MazeGrid):
    def __init__(self, obj_prob=0.0, size=(11,11), room_total=4):
        super().__init__(obj_prob)
        self.maze_type = "maze_grid_random2"
        self.size = size
        self.room_total = room_total

    def generate(self):
        from MazeGen import maze_gen_grid2
        self.maze = maze_gen_grid2.gen_maze(self.size[0],self.size[1],self.room_total)
    
    def parse(self):
        floor_list = []
        wall_list = []
        obj_list = []
        for j in range(1,self.maze.shape[0]-1):
            for i in range(1,self.maze.shape[1]-1):
                if self.maze[j,i] < 255:
                    # Build Floor
                    floor_list.append({"voff":(i,j), "id":0})
                    # Build Wall Buttom
                    if self.maze[j-1,i]==255:
                        wall_list.append({"voff":(i,j), "id":self.maze[j,i], "type":"B"})
                    # Build Wall Top
                    if self.maze[j+1,i]==255:
                        wall_list.append({"voff":(i,j), "id":self.maze[j,i], "type":"T"})
                    # Build Wall Left
                    if self.maze[j,i-1]==255:
                        wall_list.append({"voff":(i,j), "id":self.maze[j,i], "type":"L"})
                    # Build Wall Right
                    if self.maze[j,i+1]==255:
                        wall_list.append({"voff":(i,j), "id":self.maze[j,i], "type":"R"})
                
                # Build Object
                if 0 < self.maze[j,i] < 255 and np.random.rand() < self.obj_prob:
                    obj_list.append({"voff":(i+0.5,j+0.5,0.5)})

        return floor_list, wall_list, obj_list

class MazeGridDungeon(MazeGrid):
    def __init__(self, obj_prob=0.0, width=12, height=12, max_rooms=5, min_room_xy=3, \
            max_room_xy=5, rooms_overlap=False, random_connections=1, random_spurs=3):
        super().__init__(obj_prob)
        self.maze_type = "maze_grid_dungeon"
        self.width = width
        self.height = height
        self.max_rooms = max_rooms
        self.min_room_xy = min_room_xy
        self.max_room_xy = max_room_xy
        self.rooms_overlap = rooms_overlap
        self.random_connections = random_connections
        self.random_spurs = random_spurs

    def generate(self):
        from MazeGen import maze_gen_dungeon
        gen = maze_gen_dungeon.Generator(width=self.width, height=self.height, max_rooms=self.max_rooms, min_room_xy=self.min_room_xy, \
            max_room_xy=self.max_room_xy, rooms_overlap=self.rooms_overlap, random_connections=self.random_connections, random_spurs=self.random_spurs)
        gen.gen_level()
        self.maze = np.array(gen.level, dtype=np.uint8)

class MazeGridDungeon2(MazeGrid):
    def __init__(self, obj_prob=0.0, cellsX=3, cellsY=3, cellSize=5):
        super().__init__(obj_prob)
        self.cellsX = cellsX
        self.cellsY = cellsY
        self.cellSize = cellSize
        

    def generate(self):
        from MazeGen import maze_gen_dungeon2
        maze = maze_gen_dungeon2.generate(self.cellsX, self.cellsY, self.cellSize)
        self.maze = np.array(maze, dtype=np.uint8)
        

#######################################
# Board Maze
#######################################
class MazeBoard(Maze):
    def __init__(self):
        self.maze_type = "maze_board"

    def parse(self, tex_num=6):
        floor_list = []
        wall_list = []
        obj_list = []
        for j in range(self.maze.shape[0]):
            for i in range(self.maze.shape[1]):
                floor_list.append({"voff":(i,j), "id":0})
                # Build Wall Buttom
                if self.maze[j,i,0]==1:
                    tid = np.random.randint(0,tex_num)
                    wall_list.append({"voff":(i,j), "id":tid, "type":"B"})
                # Build Wall Top
                if self.maze[j,i,2]==1:
                    tid = np.random.randint(0,tex_num)
                    wall_list.append({"voff":(i,j), "id":tid, "type":"T"})
                # Build Wall Left
                if self.maze[j,i,1]==1:
                    tid = np.random.randint(0,tex_num)
                    wall_list.append({"voff":(i,j), "id":tid, "type":"L"})
                # Build Wall Right
                if self.maze[j,i,3]==1:
                    tid = np.random.randint(0,tex_num)
                    wall_list.append({"voff":(i,j), "id":tid, "type":"R"})

        return floor_list, wall_list, obj_list
    
    def get_map(self, pose, map_scale=16, wall_rate=0.4, flip=True):
        maze_draw  = 255*np.ones((self.maze.shape[0]*map_scale, self.maze.shape[1]*map_scale, 3), dtype=np.uint8)
        
        # Draw Wall
        wall_size = int(map_scale*wall_rate)
        for j in range(self.maze.shape[0]):
            for i in range(self.maze.shape[1]):
                if self.maze[j,i,2] == 1:
                    x1, y1 = (i)*map_scale, (j+1)*map_scale
                    x2, y2 = (i+1)*map_scale, (j+1)*map_scale
                    cv2.line(maze_draw, (x1, y1), (x2, y2), (0,0,0), wall_size)
                if self.maze[j,i,1] == 1:
                    x1, y1 = (i)*map_scale, (j)*map_scale
                    x2, y2 = (i)*map_scale, (j+1)*map_scale
                    cv2.line(maze_draw, (x1, y1), (x2, y2), (0,0,0), wall_size)
                if self.maze[j,i,0] == 1:
                    x1, y1 = (i)*map_scale, (j)*map_scale
                    x2, y2 = (i+1)*map_scale, (j)*map_scale
                    cv2.line(maze_draw, (x1, y1), (x2, y2), (0,0,0), wall_size)
                if self.maze[j,i,3] == 1:
                    x1, y1 = (i+1)*map_scale, (j)*map_scale
                    x2, y2 = (i+1)*map_scale, (j+1)*map_scale
                    cv2.line(maze_draw, (x1, y1), (x2, y2), (0,0,0), wall_size)

        # Draw Agent
        if pose is not None:
            x, y, th = pose[0], pose[1], pose[2]
            temp_y = int(y*map_scale)
            temp_x = int(x*map_scale)
            cv2.circle(maze_draw, (temp_x, temp_y), int(map_scale/5), (255,0,0), 3)
            temp_y2 = int((y + 0.3*np.sin(th)) * map_scale)
            temp_x2 = int((x + 0.3*np.cos(th)) * map_scale)
            cv2.line(maze_draw, (temp_x, temp_y), (temp_x2, temp_y2), (0,0,255), 3)
        if flip:
            maze_draw = cv2.flip(maze_draw,0)
        return maze_draw
    
    def collision_detect(self, pose):
        x_scale = int(pose[0] * self.map_scale)
        y_scale = int(pose[1] * self.map_scale)

        # Check in map
        if y_scale < 0 or y_scale >= self.collision_map.shape[0]:
            return True
        if x_scale < 0 or x_scale >= self.collision_map.shape[1]:
            return True
        # Check Collision
        if self.collision_map[y_scale, x_scale, 0] == 0:
            return True
        else:
            return False
    
    def random_pose(self):
        while(True):
            y = self.maze.shape[0] * np.random.rand()
            x = self.maze.shape[1] * np.random.rand()
            th = np.pi * 2 * np.random.rand()
            pose = [x, y, th]
            if self.collision_detect(pose) == False:
                return  pose

# T, L, B, R
class MazeBoardRoom(MazeBoard):
    def __init__(self):
        self.maze_type = "maze_board_room"

    def generate(self, room_size=(4,4)):
        self.maze = np.zeros((room_size[0], room_size[1], 4), dtype=np.uint8)
        self.maze[0,:,0] = 1
        self.maze[room_size[0]-1,:,2] = 1
        self.maze[:,0,1] = 1
        self.maze[:,room_size[1]-1,3] = 1
        self.map_scale = 16
        self.wall_rate = 0.4
        self.collision_map = self.get_map(None, self.map_scale, self.wall_rate, flip=False)

class MazeBoardRandom(MazeBoard):
    def __init__(self):
        self.maze_type = "maze_board_random"

    def generate(self, size=(9,9), room_size=3, room_num=3):
        from MazeGen import maze_gen_board
        M = maze_gen_board.Maze(size[0], size[1])
        M.dfs(0,0)
        M.imperfect2(size=room_size, num=room_num)
        #M.imperfect1()
        #M.render()
        self.maze = np.array(M.cell, dtype=np.uint8)
        self.map_scale = 16
        self.wall_rate = 0.4
        self.collision_map = self.get_map(None, self.map_scale, self.wall_rate, flip=False)

if __name__ == "__main__":
    maze_obj = MazeGridDungeon2()
    maze_obj.generate()
    print(maze_obj.maze)
    maze_draw = maze_obj.get_map((2,2,0))
    cv2.imshow("test", maze_draw)
    cv2.waitKey(0)
