import cv2
import os

camera_id = 0
cap = cv2.VideoCapture(camera_id)

if not cap.isOpened():
    print("can not open camera")
    exit()

save_dir = f"./camera{camera_id}/"
os.mkdir(save_dir)

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow(str(camera_id), frame)
    
    frame_id += 1
    save_name = save_dir + str(frame_id) + ".jpg"
    cv2.imwrite(save_name, frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()