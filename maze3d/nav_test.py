import numpy as np
import cv2
import maze
import maze_env

def select_action(state):
    hsv = cv2.cvtColor(state, cv2.COLOR_BGR2HSV)
    red1 = np.array([0, 70, 50])
    red2 = np.array([10, 255, 255])
    red3 = np.array([170, 70, 50])
    red4 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, red1, red2)
    mask2 = cv2.inRange(hsv, red3, red4)
    mask = np.logical_or(mask1, mask2)
    
    width = mask.shape[1]
    left = mask[:,:int(width/2)].sum()
    right = mask[:,int(width/2):].sum()
    cv2.imshow("mask", mask.astype(float))
    if left < 10 and right < 10:
        return 3
    if left-right > 1000:
        return 2
    if left-right < -1000:
        return 3
    
    
    return 0

if __name__ == "__main__":
    # Initial Env
    maze_obj = maze.MazeGridRoom()
    env = maze_env.MazeNavEnv(maze_obj, render_res=(96, 96))
    state, info = env.reset()
    env.render()
    step = 0
    while(True):
        step += 1
        action = select_action(state)
        state_next, reward, done, info = env.step(action)
        print("\rStep:", step, "| Reward:", reward, end="\t")
        env.render()
        state = state_next.copy()

        if done is True or step>50 :
            state, info = env.reset(gen_maze=False)
            step = 0
            print()

        cv2.waitKey(100)