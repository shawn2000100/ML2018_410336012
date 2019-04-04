# ML2018_410336012_HW2
## Assignment_2 Handwritten Character Recognition

### 文件說明
In this assignment, you are required to practice the application of handwritten character recognition using any method taught in the classroom. The steps to do this assignment include:

--------------------------------------------------------------

<ol>
 <li> downloading the dataset; </li>
 
 <li> preprocessing character images; </li>
 
 <li> reducing the dimension & choosing a classifier and training it </li>
 
 <li> evaluating the performance of the classifier; </li>
</ol>

--------------------------------------------------------------

### 成果回顧 (Things I need to submit to the course website)
#### 註：我參考了第一次作業的回答格式，來撰寫這次作業二的文件

#### (1) Source codes with good comments on statements

    import numpy as np #import numpy package
    import matplotlib.pyplot as plt #plotting package
    import scipy # 使用sklrean所需安裝的套件
    import sklearn.datasets # 引入mnist資料庫，之後可匯入數字...等圖檔
    import sklearn.svm # 作為classifier 用的 support vectorm machine
    import sklearn.metrics # 為了之後取得confusion matrix用


    #### A. downloading the dataset
    database = sklearn.datasets.load_digits() # 從MNIST database 中匯入數字的資料庫

    #### B. preprocessing character images
    total = len(database.images) # 紀錄樣本數量, 之後要將dataset分成前後兩半
    data = database.images.reshape( (total, -1) ) # reshape 1-D

    #### C. reducing the dimension & choosing a classifier and training it
    classifier = sklearn.svm.SVC( gamma = 0.001) # 使用SVM作為classifier
    classifier.fit(data[:total // 2], database.target[:total // 2]) # 將前半部的dataset送入SVC進行訓練

    #### D. evaluating the performance of the classifier.
    expected = database.target[total // 2:] # 預期得到的結果
    predicted = classifier.predict(data[total // 2:]) # 訓練完後預測的結果

    # 將正確樣本數字顯示在上排以供參考對照
    img_sample = list( zip(database.images, database.target) ) # 讀取數字圖像資料
    for start, (image, label) in enumerate(img_sample[:8]):
        plt.subplot(2, 8, start + 1)
        plt.imshow(image, cmap = plt.cm.gray_r, interpolation = 'none')
        plt.title('digit:%i' % label)
    # 將訓練完後的機器之預測結果顯示在下排
    img_predict = list(zip(database.images[total // 2:], predicted))
    for start, (image, prediction) in enumerate(img_predict[:8]):
        plt.subplot(2, 8, start + 9)
        plt.imshow(image, cmap = plt.cm.gray_r, interpolation = 'none')
        plt.title('predict:%i' % prediction)

    plt.show()

    # 完工，print出預測結果
    print( "Report for classifier %s\n" % (sklearn.metrics.classification_report(expected, predicted)) )
    print("Confusion matrix:\n%s" % sklearn.metrics.confusion_matrix(expected, predicted))


#### (2) A 3~5-page report with
#### A. downloading the dataset

這次的作業要從MNIST資料庫來下載數字的圖片

使用sklearn函式庫即可辦到


code:
 
    database = sklearn.datasets.load_digits() # 從MNIST database 中匯入數字的資料庫
    
    
#### B. preprocessing character images

使用sklearn套件裡面的函式 images.reshape( (total, -1) ) # reshape 1-D

預設就會將圖片處理好了


#### C. reducing the dimension & choosing a classifier and training it

使用SVM (support vector machine) 作為classifier 的效果據說挺不錯的, 之後將dataset拆分為前後兩半

前半部送入svc裡面進行訓練, 後半部則拿來做為訓練完後的預測分析用

code:

    classifier = sklearn.svm.SVC( gamma = 0.001) # 使用SVM作為classifier
    classifier.fit(data[:total // 2], database.target[:total // 2]) # 將前半部的dataset送入SVC進行訓練


#### D. evaluating the performance of the classifier.


使用sklearn.metrics函式庫即可輕鬆列印出report


![Confusion Matrix](https://github.com/shawn2000100/ML2018_410336012_HW2/blob/master/Confusion%20Matrix.png)


X軸為預測為該數字的結果數量

Y軸依序表示012345...，沒有特別的意義

從圖中可看出辨識的結果是相當不錯的，最難清楚辨認的數字是 3 ，可能是因為跟 8, 2, 5都長得有一點點相似吧?




![Performance](https://github.com/shawn2000100/ML2018_410336012_HW2/blob/master/Performance.png)

辨識的效能，圖中可看出平均準確度高達97%




![Result](https://github.com/shawn2000100/ML2018_410336012_HW2/blob/master/result.png)

上排為正確的數字

下排則為預測出的結果 (截錄一部分)

     
     
#### E. the problems you encountered   



  2018/06/10：
  
    
   1. 在讀完了說明文件以及規格後，一時間沒有頭緒，決定先google找資料作為參考
    
    
  2018/06/24：
  
   1. 遇到一個小問題，由於我剛換電腦，用的是Macbook的作業系統，需安裝sklear, scipy...等套件，
       然而Mac預設的Python版本為二代，而非三代，且有一些預設路徑也需要做修改，於是Google找尋指令教學
       
   2. 問題解決, 套件成功安裝完成, 環境設定似乎沒有問題了, 開始思考如何從Google到的資料中東拼西湊出完整程式碼
   
     
  2018/06/29：
     
   1. 先前已找到許多豐富的參考資料，也大致釐清了本次的作業要求以及問題，程式碼應該快要能生出來了，開始著手撰寫文件
   
  2018/06/30：
  
  不知不覺就天亮了。完成後發現其實程式碼不長，關鍵就是下面三行code而已，因為我都是用函式庫跟套件，以及Google東拼西湊的結果，所以Code其實不長。
  
  
  Code:
  
    database = sklearn.datasets.load_digits() # 從MNIST database 中匯入數字的資料庫
    
    classifier = sklearn.svm.SVC( gamma = 0.001) # 使用SVM作為classifier
    classifier.fit(data[:total // 2], database.target[:total // 2])
     
     
#### G. what you have learned from this work.
      
      
   1. 如何在Macbook的作業系統環境下安裝scipy, matplotlib...等套件函式庫
      
   2. 更了解MNIST database是什麼, 以及一些基本操作
      
   3. 利用numpy搭配matplotlib來顯示及處理圖片
   
   4. 利用sklearn套件來使用svm做訓練
      
   5. 培養version control的習慣及練習Markdown語法
      
   6. 最重要的是, 不要把事情拖到最後一刻才做!!!


----------------------------------------------------
