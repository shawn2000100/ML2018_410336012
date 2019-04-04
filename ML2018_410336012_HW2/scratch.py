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



