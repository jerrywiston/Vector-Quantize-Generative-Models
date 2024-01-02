import cv2
import random
import numpy as np

class Maze:
    
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.cell = [[[1, 1, 1, 1] for _ in range(self.col)] for _ in range(self.row)]
        self.visit = [[False for _ in range(self.col)] for _ in range(self.row)]
        
    def initialize(self):
        self.visit = [[False for _ in range(self.col)] for _ in range(self.row)]
        
    def disable(self, row, col, direction):
        self.cell[row][col][direction] = 0
        
        if direction == 0 and row > 0: #top
            self.cell[row-1][col][2] = 0
        if direction == 1 and col > 0: #left
            self.cell[row][col-1][3] = 0
        if direction == 2 and row < self.row-1: #bottom
            self.cell[row+1][col][0] = 0
        if direction == 3 and col < self.col-1: #right
            self.cell[row][col+1][1] = 0
        
    def render(self, factor=30, key=0):
        
        img = np.zeros(((self.row+2)*factor, (self.col+2)*factor, 3), np.uint8)
        img.fill(255)
        
        for i in range(self.row):
            for j in range(self.col):
                
                y, y_next = (i+1)*factor, (i+2)*factor
                x, x_next = (j+1)*factor, (j+2)*factor
                
                if self.cell[i][j][0]: #top
                    cv2.line(img, (x, y), (x_next, y), (0, 0, 0), 1)
                if self.cell[i][j][1]: #left
                    cv2.line(img, (x, y), (x, y_next), (0, 0, 0), 1)
                if self.cell[i][j][2]: #bottom
                    cv2.line(img, (x, y_next), (x_next, y_next), (0, 0, 0), 1)
                if self.cell[i][j][3]: #right
                    cv2.line(img, (x_next, y), (x_next, y_next), (0, 0, 0), 1)
        
        img = cv2.flip(img,0)
        cv2.imshow('Maze', img)
        cv2.waitKey(key)
        
    def imperfect1(self, percentage = 0.05):
        
        for _ in range(int(self.row*self.col*4*percentage)):
            
            row = random.randint(1, self.row-2)
            col = random.randint(1, self.col-2)
            d = random.randint(0, 3)
            self.disable(row, col, d)
            #self.render(key=3)
        
    def imperfect2(self, size=7, num=10):
        p = [[random.randint(1, self.row-size-1), random.randint(1, self.row-size-1)] for i in range(num)]
        
        for x, y in p:
            
            r = random.randint(1, size)
            c = random.randint(1, size)
            
            for i in range(x, x+r):
                for j in range(y, y+c):
                    for k in range(4):
                        self.disable(i, j, k)
                        #self.cell[i][j][k] = 0
            
            #self.render(key=30)
            
            
    def dfs(self, row, col):

        self.visit[row][col] = True
        choices = []
        
        if row > 0 and self.visit[row-1][col] == False:
            choices.append(0)
        if col > 0 and self.visit[row][col-1] == False:
            choices.append(1)
        if row < self.row-1 and self.visit[row+1][col] == False:
            choices.append(2)
        if col < self.col-1 and self.visit[row][col+1] == False:
            choices.append(3)
        
        random.shuffle(choices)
        for choice in choices:
            
            if choice == 0 and self.visit[row-1][col] == False:
                self.disable(row, col, choice)
                #self.render(key=3)
                
                self.dfs(row-1, col)
            elif choice == 1 and self.visit[row][col-1] == False:
                self.disable(row, col, choice)
                #self.render(key=3)
                self.dfs(row, col-1)
            elif choice == 2 and self.visit[row+1][col] == False:
                self.disable(row, col, choice)
                #self.render(key=3)
                self.dfs(row+1, col)
            elif choice == 3 and self.visit[row][col+1] == False:
                self.disable(row, col, choice)
                #self.render(key=3)
                self.dfs(row, col+1)


if __name__ == "__main__":
    M = Maze(11, 11)
    M.dfs(10, 10)
    M.imperfect2(size=5, num=6)
    #M.imperfect1()
    M.render()