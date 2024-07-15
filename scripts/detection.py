import cv2
from ultralytics import YOLOv10
import streamlit as st
import numpy as np
from PIL import Image
from playsound import playsound # pip install playsound=1.2.2
import os

print('getcwd:      ', os.getcwd())
print('__file__:    ', __file__)


model = YOLOv10.from_pretrained("jameslahm/yolov10n") 

backgroundColor = "#F0F0F0"
font = "Helvetica Neue"

st.title("監視カメラ（YOLOv10）")
# st.divider()

label_names_st = st.empty()
scores_st = st.empty()
image_loc = st.empty()



cap = cv2.VideoCapture(0)

stack = 0
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Run YOLOv10 inference on the frame
        results = model(frame,imgsz = 640, conf=0.5)
        
        flag = 0
        classes = results[0].boxes.cls
        for cls in classes:
            if int(cls) == 0:
                flag = 1
        print(flag)
        const = 0.07
        stack = stack*(1-const) +const*flag
        print(stack)
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        # print(results[0].names)
        if stack > 0.8:
            label_names_st.markdown('<b style="color:red; font-size: 36px;">ﾐ ﾂ ｹ ﾀ!</b>', unsafe_allow_html=True)
            playsound("./wav/Warning-Siren04-02_Low-Long_.wav")
        else:
            label_names_st.markdown('<b style="color:red; font-size: 36px;">人間検出中...</b>', unsafe_allow_html=True)
        
        # scores_st.markdown('<p style="font-family:monospace; color:purple; font-size: 36px;">monospace</p>', unsafe_allow_html=True)
        image_loc.image(annotated_frame)
        
        # Display the annotated frame
        # cv2.imshow("YOLOv8 Inference", annotated_frame)

        # # Break the loop if 'q' is pressed
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()