import cv2
import numpy as np

# สร้างหน้าต่าง
cv2.namedWindow("Real-time Character Display", cv2.WINDOW_NORMAL)

# กำหนดแบบอักษร (font) และขนาด
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 5
font_thickness = 8

# สร้างสีข้อความ
text_color = (0, 0, 0)  # สีดำ (RGB)

# สร้างพื้นหลังสีขาว
background_color = (255, 255, 255)  # สีขาว (RGB)

# เริ่มการเปิดกล้องเว็บแคม
cap = cv2.VideoCapture(0)  # 0 คือเลือกกล้องเริ่มต้น

while True:
    # อ่านภาพจากกล้อง
    ret, frame = cap.read()

    # กำหนดข้อความที่จะแสดง (ตัวอักษร "ก")
    text = "ก"

    # คำนวณขนาดข้อความ
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_width, text_height = text_size

    # คำนวณตำแหน่งที่ต้องการแสดงข้อความอยู่กลางภาพ
    x = (frame.shape[1] - text_width) // 2
    y = (frame.shape[0] + text_height) // 2

    # สร้างภาพตัวอักษร "ก" บนภาพ
    image = np.ones_like(frame) * background_color
    cv2.putText(image, text, (x, y), font, font_scale, text_color, font_thickness)

    # แปลงภาพให้เป็นชนิดข้อมูล unsigned 8-bit integer (CV_8U)
    image = cv2.convertScaleAbs(image)

    # แสดงภาพแบบ Real-time
    cv2.imshow("Real-time Character Display", image)

    # หากกด 'q' ให้ออกจากลูป
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้องเว็บแคมและหน้าต่าง
cap.release()
cv2.destroyAllWindows()
