import face_recognition
import cv2
import os


total_image = []
total_image_name = []
total_face_encoding = []

path ="/Users/JayChen/Downloads/Face Database" #在同級目錄下的images文檔中放需要被識別出的人物圖
for fn in os.listdir(path): #fn 表示的是文檔名
  # print(fn)
  total_face_encoding.append( face_recognition.face_encodings( face_recognition.load_image_file(path + "/" + fn) )[0] )
  fn = fn[:(len(fn) - 4)] #截取圖片名（這裏應該把images文檔中的圖片名命名為為人物名）
  # print(fn)
  total_image_name.append(fn) #圖片名字列表


# 開始測試人臉辨識
for case in range(5):
  test = input(' 輸入一個欲比對之圖片名稱 ex:s01_01 (請勿輸入不存在的名稱): ')
  unknown_image = face_recognition.load_image_file("/Users/JayChen/Downloads/Face Database/" + test + ".jpg")
  unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

  count = 0
  for fn in os.listdir(path):
    known_image = face_recognition.load_image_file(path + "/" + fn)
    known_encoding = face_recognition.face_encodings(known_image)[0]

    results = face_recognition.compare_faces([known_encoding], unknown_encoding)
    if(results):
        fn = fn[:(len(fn) - 4)]
        print(fn, end = '  ')
        count += 1

    if(count >= 5):
        print()
        break