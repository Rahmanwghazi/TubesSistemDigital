import cv2

facecascade = cv2.CascadeClassifier("D:\\Telyuu\\Sistem Digital\\Tubes\\haarcascade_frontalface_default.xml")
nosecascade = cv2.CascadeClassifier("D:\\Telyuu\\Sistem Digital\\Tubes\\Nariz.xml")
mouthcascade = cv2.CascadeClassifier("D:\\Telyuu\\Sistem Digital\\Tubes\\Mouth.xml")


video_capture = cv2.VideoCapture(0)
mask_on = False

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = facecascade.detectMultiScale(gray, 1.1,5)

    for (x,y,w,h) in wajah:
        roi_color = frame [y:y+h, x:x+w]


        hidung = nosecascade.detectMultiScale(roi_color,1.18,35,)
        for (sx, sy, sw, sh) in hidung:
            cv2.rectangle(roi_color,(sh, sy),(sx+sw, sy+sh), (255,0,0),2)
            cv2.putText(frame, 'Hidung', (x + sx, y + sy), 1, 1, (0,255,0),1)

        mulut = mouthcascade.detectMultiScale(roi_color,1.18,35,)
        for (sx, sy, sw, sh) in mulut:
            cv2.rectangle(roi_color,(sh, sy),(sx+sw, sy+sh), (255,0,0),2)
            cv2.putText(frame, 'mulut', (x + sx, y + sy), 1, 1, (0,255,0),1)
        
        if len(hidung)>0:
            mask_on = False

        elif len(mulut)>0:
            mask_on = False
        else:
            mask_on = True

        if mask_on:
            cv2.rectangle(frame,(x,y), (x+w, y+h), (0,255,0),3)
            cv2.putText(frame, 'Menggunakan masker', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),3)
        else:
            cv2.rectangle(frame,(x,y), (x+w, y+h), (0,0,255),3)
            cv2.putText(frame, 'Tidak menggunakan masker', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),3)


        cv2.putText(frame,'Jumlah wajah terdeteksi: ' + str(len(wajah)),(80,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord ('q'):
        break
video_capture.release()
cv2.destroyAllWindows()