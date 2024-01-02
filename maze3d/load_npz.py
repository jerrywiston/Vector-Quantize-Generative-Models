import numpy as np
import cv2

with np.load("maze.npz") as data:
    for k in data:
        print(k)
    image_data = data["image"]
    depth_data = data["depth"]
    pose_data = data["pose"]
    maze_data = data["maze"]

for i in range(10):
    image = image_data[i].transpose(1,0,2,3).reshape(192,-1,3)
    image = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)))
    cv2.imshow("image", image)
    
    depth = depth_data[i].transpose(1,0,2).reshape(192,-1)
    depth = cv2.resize(depth, (int(depth.shape[1]/2), int(depth.shape[0]/2)))
    cv2.imshow("depth", depth/4)

    cv2.waitKey(0)
