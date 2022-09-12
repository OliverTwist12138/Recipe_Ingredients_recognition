import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('test_imgs/apple2.jfif',0)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
edges = cv2.Canny(img, 100, 200) #参数:图片，minval，maxval,kernel = 3

plt.subplot(131) #121表示行数，列数，图片的序号即共一行两列，第一张图
plt.imshow(img,cmap='gray') #cmap :colormap 设置颜色
plt.title('original image'),plt.xticks([]),plt.yticks([]) #坐标轴起点，终点的值
plt.subplot(132),plt.imshow(edges,cmap = 'gray')
plt.title('edge image'),plt.xticks([]),plt.yticks([])

height = len(img)
width = len(img[0])
new_img = img

for h in range(height):
    w = 0
    while w < width and edges[h][w] == 0:
        new_img[h][w] = 255
        w += 1
    r = width - 1
    while r > w and edges[h][r] == 0:
        new_img[h][r] = 255
        r -= 1

plt.subplot(133) #121表示行数，列数，图片的序号即共一行两列，第一张图
plt.imshow(new_img) #cmap :colormap 设置颜色
plt.title('new image'),plt.xticks([]),plt.yticks([]) #坐标轴起点，终点的值

plt.show()