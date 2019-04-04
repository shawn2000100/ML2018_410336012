# ML2018_410336012
## Assignment_1 Image_Decryption_by_Single_Layer_Neural_Network

Description:
Please download the description file of the assignment here:[Descriptions](http://www.elearn.ndhu.edu.tw/moodle/file.php/74252/Assignment_1-_Image_Decryption_by_Single_Layer_Neural_Network.pdf)

Dataset:
Please download the required image and data file here: [Image and Data](http://www.elearn.ndhu.edu.tw/moodle/file.php/74252/Image_and_ImageData.zip)
  
----------------------------------------------------
<ol>
  <li> 使用Python的opencv函式庫讀取圖檔 </li>
  <li> 加密公式: 𝐸 = 𝑤1𝐾1 + 𝑤2𝐾2 + 𝑤3𝐼 </li>
  <li> 解密公式: 𝐼 = 𝐸 − w1𝐾1 − 𝑤2𝐾2𝑤3 </li>
  <li> 使用single-layer neural network 來求出 weight vector (W) </li> 
  <li> Network architecture 以及 Training algorithm (based on gradient descent method) 參見於附檔 </li>
  <li> 解出 weight vector W後, 利用上課所提示的公式設計演算法進行圖像解密 </li>
  <li> 利用opencv函式庫將解密出之圖片匯出 </li>
</ol>

### 成果回顧 (Things I need to submit to the course website)

#### (1) Source codes with good comments on statements

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

#### (2) A 3~5-page report with
#### A. the way how you prepare the training samples
    Training Set是使用課堂上老師所提供的圖像檔
    
    
#### B. all parameters, such as MaxIterLimit, α, and 𝜖 , you used for the training algorithm
    
    
    MaxIterLimit = 2  
    
    α = 0.00001   
    
    𝜖 = 1e-05
    
    Adaptive Least Mean Square Error Method by Gradient Descent. 



#### C. the derived weight vector 𝐰
    權重 W = [0.24914331 0.6613819  0.08923953]
    
#### D. the printed image 𝐼’ decrypted from 𝐸’

     [見附檔 'ans.png']
     
     (https://github.com/shawn2000100/ML2018_410336012/blob/master/ans.png)
     
#### E. the problems you encountered   


  2018/04/30    
    
    1. 摸索markdown的語法以及GITHUB的相關操作
     
     
  2018/05/01
  
    1. 不知道如何安裝opencv與numpy.  google了很久之後問資工系的朋友才知道原來可以在pycharm的setting裡直接install package  
    
    2. 安裝好package後, 在操作cv2.imread上遇到問題, google後仍然不會解決, 只好再度詢問朋友 
   
   
  ##### CODE:
    img1 = cv2.imread("C:\\Users\\shawn\\PycharmProjects\\20180501.png")
    cv2.namedWindow("Image")
    cv2.imshow("Image", img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  
  
  ##### BUG:
    cv2.error: C:\projects\opencv-python\opencv\modules\highgui\src\window.cpp:331: error: (-215) size.width>0 && size.height>0 in function cv::imshow
  
  
  2018/05/02
      
      1. 了解一些基本的numpy函式庫操作
      
      2. 試著從老師的演算法以及課堂提示來寫出python程式
      

#### F. what you have learned from this work.
      
      1. 如何在PyCharm上安裝常用的package
      
      2. OpenCv讀寫圖檔, Numpy的一些基本操作
      
      3. 機器學習演算法的核心是數學, 以後一定要找時間把微積分、線性代數及機率複習一次
      
      4. GitHub的一些基本操作
      
      5. 培養version control的習慣


----------------------------------------------------
