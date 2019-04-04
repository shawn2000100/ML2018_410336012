import cv2
import numpy as np

# 讀取圖片準備開始處理
img1 = cv2.imread('C:\\Users\\shawn\\PycharmProjects\\20180501\\key1.png', 0)
img2 = cv2.imread('C:\\Users\\shawn\\PycharmProjects\\20180501\\key2.png', 0)
imgI = cv2.imread('C:\\Users\\shawn\\PycharmProjects\\20180501\\I.png', 0)
imgE = cv2.imread('C:\\Users\\shawn\\PycharmProjects\\20180501\\E.png', 0)
imgEprime = cv2.imread('C:\\Users\\shawn\\PycharmProjects\\20180501\\Eprime.png', 0)

# 設定初始權重
w = [1,1,1]
w = np.transpose(w)
# 設定圖片尺寸
W, H = np.shape(img1)
epoch = 1
a = np.zeros((300, 400))
e = np.zeros((300, 400))
I = np.zeros((300, 400))
r = 0.00001

while(epoch < 2):
        for i in range(W):
            for j in range(H):
                a[i, j] = np.dot( np.transpose(w) , np.transpose([ img1[i, j], img2[i, j], imgI[i, j]]) )
                e[i, j] = imgE[i, j] - a[i,j]
                w = w + r * e[i, j] * np.transpose([ img1[i, j], img2[i, j], imgI[i, j] ])
            epoch += 1

# 利用得出的W權重進行解密
for i in range(W):
    for j in range(H):
        I[i, j] = [imgEprime[i, j] - w[0] * img1[i, j] - w[1] * img2[i, j]] / w[2]
        if I[i, j] > 255:
            I[i, j] = 255
        elif I[i, j] < 0:
            I[i, j] = 0

# 最後將型態轉回圖片用的unit8型態
cv2.imwrite('ans.png', I)