import cv2
import numpy as np

class TakeAPicture():
    def shotAndShow():
        print("TakeAPicture Activate...")
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
        # 定義加入文字的函式
        def putText(source, x, y, text, scale=2.5, color=(255,255,255)):
            org = (x,y)
            fontFace = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = scale
            thickness = 5
            lineType = cv2.LINE_AA
            cv2.putText(source, text, org, fontFace, fontScale, color, thickness, lineType)
        a = 0
        n = 0
        f = 0

        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        while True:

            f +=1
            ret, img = cap.read()

            if not ret:
                print("Cannot receive frame")
                break
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            w = int(img.shape[1]*0.5)
            h = int(img.shape[0]*0.5)
            img = cv2.resize(img,(w,h))

            white = 255 - np.zeros((h,w,4), dtype='uint8')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            photo = img.copy()
            cv2.putText(img, "Taking Photo", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (134, 25, 12), 1, cv2.LINE_AA)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi = photo[y:y+h+10, x:x+w+10]
            key = cv2.waitKey(1)
            #if key == 32:
            if f ==100:
                a = 1
                sec = 4  # 加入倒數秒數
            elif key == ord('q'):
                break
            if a == 0:
                output = img.copy()
            else:
                output = img.copy()  # 設定 output 和 photo 變數
                #photo = img.copy()
                sec = sec - 0.05     # sec 不斷減少 0.05 ( 根據個人電腦效能設定，使其搭配 while 迴圈看起來像倒數一秒 )
                putText(img, 10, 70, str(int(sec)))  # 加入文字
                # 如果秒數小於 1
                if sec < 1:
                    output = cv2.addWeighted(white, a, photo, 1-a, 0)
                    a = a - 0.1
                    if a<=0:
                        a = 0
                        n = n + 1
                        #cv2.imwrite(f'photo_{n}.jpg', photo)
                        cv2.imwrite(f'ROI_{n}.jpg',roi)
                        break
            #cv2.putText(img, "Taking Photo..", (w, h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow('Take A Photo', img)

        cap.release()
        cv2.destroyAllWindows()
        print("TakeAPicture Progress Over")
        TakeAPicture.show()
    def show():
        myPhoto = cv2.imread("./ROI_1.jpg")
        cv2.imshow("This is you!",myPhoto)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
if __name__ == '__main__':
    TakeAPicture.shotAndShow()