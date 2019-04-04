# ML2018_410336012_Final
## Term_Project Face Recognizer System

### 說明

我使用python的face_recognition套件來做這次期末作業，這個套件主要的實踐技術用到了：Convolutional Neural Network (CNN)、k-means clustering演算法以及Support Vector Machine (SVM)。

我使用以下幾行程式碼來讀取人臉資料庫作為input

    for fn in os.listdir(path): #fn 表示的是文檔名
      # print(fn)
      total_face_encoding.append( face_recognition.face_encodings( face_recognition.load_image_file(path + "/" + fn) )[0] )
      fn = fn[:(len(fn) - 4)] #截取圖片名（這裏應該把images文檔中的圖片名命名為為人物名）
      # print(fn）
      total_image_name.append(fn) #圖片名字列表
      


而訓練模型我則交給內建的函式來處理，將欲辨識的圖片(Unknown_Image)與已知的人臉資料庫(Known_Image)做比對辨識由face_encodings這個函式將圖片進行編碼，再由compare_faces此函式來做人臉比對，最後將比對的結果進行輸出即可。


    unknown_image = face_recognition.load_image_file("/Users/JayChen/Downloads/Face Database/" + test + ".jpg")
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
 
    known_image = face_recognition.load_image_file(path + "/" + fn)
    known_encoding = face_recognition.face_encodings(known_image)[0]

    results = face_recognition.compare_faces([known_encoding], unknown_encoding)
    


不過最後要測試時我還是遇到了Bug，不管我輸入哪一張圖片作為比對，他都只會輸出同一種辨識結果。即，s01_01與s03_01、s09_01的辨識結果都是一樣的，我不太清楚這個Bug是怎麼形成的，可能要去研究函式庫的原始碼了，不過我目前還沒有這麼厲害... 這次專案只完成了一半，實在有點難過... 如果有組員可以一起討論以及督促即時做作業可能這次的產出會好很多。



#### 參考資料

https://hk.saowen.com/a/deab0643d3768999b75231985a9ea8cb67521a2a4029206d1e4db86a7ae5b231
https://github.com/ageitgey/face_recognition
https://xiaozhuanlan.com/topic/0273148596


