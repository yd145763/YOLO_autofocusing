# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 07:22:52 2024

@author: limyu
"""

from ultralytics import YOLO
from pypylon import pylon
import cv2
import numpy as np
import devices_TMC_USB as de
import matplotlib.pyplot as plt
import numpy as np
import time


def moveup1():
    
    # 创建设备
    TMC1 = de.TMC_USB()
    
    try:
        # 打开，并连接硬件
        enum_dev = TMC1.tmc_device_count()
        print(f'设备数量为:{enum_dev}')
        handle = TMC1.tmc_open(enum_dev-1)
        print(f'设备句柄为:{handle}')
        
        # 读取设备状态
        status = TMC1.tmc_get_status(handle, 0x00)
        print(f'设备状态为:{status}')
    
        #读取设备错误状态原因
        buff1 = TMC1.tmc_get_error(handle, 128)
        print(
            f'设备错误状态原因：{buff1}')
    
    
        # 设置 板卡名称
        TMC1.tmc_set_board_name(handle, 'waterfall',9)
        # 读取 板卡名称
        name1 = TMC1.tmc_get_board_name(handle, 9)
        print(f'板卡名称:{name1}')
    
        # 加载/重置 轴参数
        TMC1.tmc_init(handle, 0x00, "TSA100-B")
    
        # 设置轴控制状态
        issuccess = TMC1.tmc_set_axis_enable(handle, 0x00, True)
        print(f'设置轴控制状态:{issuccess}')
    
        # 设置 运动单位
        TMC1.tmc_set_unit(handle, 0x00, 2)
        
        # 读取 运动单位
        unit=TMC1.tmc_get_unit(handle, 0x00)
        print(f'运动单位为：{unit}')
        
        #set initial speed
        TMC1.tmc_set_init_speed(handle, 0x00, 0.05)
    
        # get initial speed
        speed=TMC1.tmc_get_init_speed(handle, 0x00)
        print(f'initial speed is：{speed}')
        
        #set speed
        TMC1.tmc_set_speed(handle, 0x00, 0.05)
    
        # get speed
        speed=TMC1.tmc_get_speed(handle, 0x00)
        print(f'speed is：{speed}')
        
        # h归零
        #TMC1.tmc_go_home(handle, 0x00)
    
        # 绝对移动
        issuccess = TMC1.tmc_relative_move(handle, 0x00, 0.001)
        print(f'设备移动：{issuccess}')
    
        # 停止移动
        TMC1.tmc_stop(handle, 0x00)



    finally:
        # 关闭设备
        TMC1.tmc_close(handle)
        print("关闭设备")

def movedown1():
    
    # 创建设备
    TMC1 = de.TMC_USB()
    
    try:
        # 打开，并连接硬件
        enum_dev = TMC1.tmc_device_count()
        print(f'设备数量为:{enum_dev}')
        handle = TMC1.tmc_open(enum_dev-1)
        print(f'设备句柄为:{handle}')
        
        # 读取设备状态
        status = TMC1.tmc_get_status(handle, 0x00)
        print(f'设备状态为:{status}')
    
        #读取设备错误状态原因
        buff1 = TMC1.tmc_get_error(handle, 128)
        print(
            f'设备错误状态原因：{buff1}')
    
    
        # 设置 板卡名称
        TMC1.tmc_set_board_name(handle, 'waterfall',9)
        # 读取 板卡名称
        name1 = TMC1.tmc_get_board_name(handle, 9)
        print(f'板卡名称:{name1}')
    
        # 加载/重置 轴参数
        TMC1.tmc_init(handle, 0x00, "TSA100-B")
    
        # 设置轴控制状态
        issuccess = TMC1.tmc_set_axis_enable(handle, 0x00, True)
        print(f'设置轴控制状态:{issuccess}')
    
        # 设置 运动单位
        TMC1.tmc_set_unit(handle, 0x00, 2)
        
        # 读取 运动单位
        unit=TMC1.tmc_get_unit(handle, 0x00)
        print(f'运动单位为：{unit}')
        
        #set initial speed
        TMC1.tmc_set_init_speed(handle, 0x00, 0.05)
    
        # get initial speed
        speed=TMC1.tmc_get_init_speed(handle, 0x00)
        print(f'initial speed is：{speed}')
        
        #set speed
        TMC1.tmc_set_speed(handle, 0x00, 0.05)
    
        # get speed
        speed=TMC1.tmc_get_speed(handle, 0x00)
        print(f'speed is：{speed}')
        
        # h归零
        #TMC1.tmc_go_home(handle, 0x00)
    
        # 绝对移动
        issuccess = TMC1.tmc_relative_move(handle, 0x00, -0.001)
        print(f'设备移动：{issuccess}')
    
        # 停止移动
        TMC1.tmc_stop(handle, 0x00)



    finally:
        # 关闭设备
        TMC1.tmc_close(handle)
        print("关闭设备")


def moveup():
    print('move up')

def movedown():
    print('move down')
    


# Load the model
model = YOLO("C:\\Users\\yudian.lim\\Desktop\\tmc-usb sdk release\\Python\\mixed_power\\weights_01validation\\weights\\best.pt")

# Define colors for 10 classes
colors = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 0),    # Maroon
    (0, 128, 0),    # Dark Green
    (0, 0, 128),    # Navy
    (128, 128, 0)   # Olive
]


    

# Connect to the camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

# Set parameters, if needed
camera.Width = camera.Width.Max
camera.Height = camera.Height.Max

camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

# Prepare an OpenCV window
cv2.namedWindow('YOLOv8 Real-time Detection', cv2.WINDOW_NORMAL)

dist_per_pixel = 579 / camera.Width.Value  # Update this based on your camera's resolution

def findwidth(cropped_array):
    col_shortened = cropped_array.shape[1]//3
    shorted_array = cropped_array[:, col_shortened:2*col_shortened]
    max_index = np.argmax(shorted_array)
    row, col = np.unravel_index(max_index, shorted_array.shape)
    e = shorted_array[:, col]
    max_indices1 = np.where(e == e.max())[0]
    max_indices2 = np.where(e == int(e.max()*0.80))[0]
    max_indices3 = np.where(e == int(e.max()*0.85))[0]
    max_indices4 = np.where(e == int(e.max()*0.90))[0]
    max_indices5 = np.where(e == int(e.max()*0.95))[0]
    max_indices6 = np.where(e == int(e.max()*0.99))[0]

    max_indices = np.concatenate((max_indices1, max_indices2, max_indices3, max_indices4, max_indices5, max_indices6))
    if len(max_indices) >1:
        beam_width = max(max_indices) - min(max_indices)
    if len(max_indices) ==1:
        beam_width = 0
    if len(max_indices) ==0:
        beam_width = 0
    beam_width = beam_width*dist_per_pixel
    return beam_width



while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    
    
    
    if grabResult.GrabSucceeded():
        # Access the image data
        image = grabResult.Array

        # Convert grayscale image to BGR for display
        if len(image.shape) == 2:  # Check if the image is grayscale
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = image

        # Perform detection
        results = model(image_bgr)
        
        region_list = []
        
        # Process results and draw bounding boxes
        for result in results:
            for box in result.boxes:
                # Convert tensor values to Python scalars
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{model.names[cls]}:{conf:.2f}"
                
                region_list.append(model.names[cls])

                # Select color based on class index
                color = colors[cls % len(colors)]

                # Crop the detected region from the image for width calculation
                cropped_region = image[y1:y2, x1:x2]  # Use original grayscale image for analysis
                width = findwidth(cropped_region)
                
                # Draw the bounding box and label on the frame
                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 5)  # Adjust the thickness as needed
                cv2.putText(image_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 4, color, 5)  # Adjust font size and thickness

                # Display the calculated width on the image
                cv2.putText(image_bgr, f'{width:.2f}um', (x2-10, y2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 4, color, 5)

        # Display the resulting frame
        cv2.imshow('YOLOv8 Real-time Detection', image_bgr)
        print(region_list)


        region_a_count = region_list.count('Region A')
        region_b_count = region_list.count('Region B')
        region_c_count = region_list.count('Region C')
        region_d_count = region_list.count('Region D')
        

        if region_a_count > int(len(region_list)/2) and region_b_count !=len(region_list):
            moveup1()
            # Get the text size to position it correctly
            (text_width, text_height), _ = cv2.getTextSize('moving up...', cv2.FONT_HERSHEY_SIMPLEX, 4, 5)
            text_x = 20  # X position for the text (10 pixels from the left)
            text_y = image_bgr.shape[0] - 200  # Y position for the text (10 pixels from the bottom)

            # Put the text on the image
            cv2.putText(image_bgr, 'moving up...', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 255, 255), 8, cv2.LINE_AA)

            if region_b_count < int(len(region_list)*0.8):
                moveup1()
                # Get the text size to position it correctly
                (text_width, text_height), _ = cv2.getTextSize('moving up...', cv2.FONT_HERSHEY_SIMPLEX, 4, 5)
                text_x = 20  # X position for the text (10 pixels from the left)
                text_y = image_bgr.shape[0] - 200  # Y position for the text (10 pixels from the bottom)
    
                # Put the text on the image
                cv2.putText(image_bgr, 'moving up...', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 255, 255), 8, cv2.LINE_AA)



        if region_c_count > int(len(region_list)/2) or region_d_count > int(len(region_list)/2) and region_b_count !=len(region_list):
            movedown1()
            # Get the text size to position it correctly
            (text_width, text_height), _ = cv2.getTextSize('moving down...', cv2.FONT_HERSHEY_SIMPLEX, 4, 5)
            text_x = 20  # X position for the text (10 pixels from the left)
            text_y = image_bgr.shape[0] - 200  # Y position for the text (10 pixels from the bottom)

            # Put the text on the image
            cv2.putText(image_bgr, 'moving down...', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 255, 255), 8, cv2.LINE_AA)

            if region_b_count < int(len(region_list)*0.8):
                movedown1()
                # Get the text size to position it correctly
                (text_width, text_height), _ = cv2.getTextSize('moving down...', cv2.FONT_HERSHEY_SIMPLEX, 4, 5)
                text_x = 20  # X position for the text (10 pixels from the left)
                text_y = image_bgr.shape[0] - 200  # Y position for the text (10 pixels from the bottom)
    
                # Put the text on the image
                cv2.putText(image_bgr, 'moving down...', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 255, 255), 8, cv2.LINE_AA)
        

        
        # Display the resulting frame
        cv2.imshow('YOLOv8 Real-time Detection', image_bgr)
        

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    grabResult.Release()

# Release the camera and close all OpenCV windows
camera.StopGrabbing()
camera.Close()
cv2.destroyAllWindows()


