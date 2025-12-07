import cv2
import numpy as np
import sys
import os
import time


# 获取可执行文件所在目录
def resource_path(absolute_path):
    """获取资源文件的绝对路径"""
    try:
        # PyInstaller创建临时文件夹，将路径存储在_MEIPASS中
        temporary_path = sys._MEIPASS
    except Exception:
        temporary_path = os.path.abspath(".")

    return os.path.join(temporary_path, absolute_path)


# HSV颜色范围 [H_min, S_min, V_min, H_max, S_max, V_max]
myclolors = [[5, 107, 0, 19, 255, 255],  # 橙色
             [133, 56, 0, 159, 156, 255],  # 紫色
             [57, 76, 0, 100, 255, 255],  # 绿色
             [90, 48, 0, 118, 255, 255]]  # 蓝色

mycolorvalues = [[51, 153, 255],  # 橙色
                 [255, 0, 255],  # 紫色
                 [0, 255, 0],  # 绿色
                 [255, 0, 0]]  # 蓝色

mypoints = []  # [x, y, colorID, time]


def findColor(img, myclolors, mycolorvalues):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    count = 0
    newPoints = []
    for color in myclolors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)

        # 显示mask以便调试
        # cv2.imshow(str(count), mask)

        x, y = getContours(mask, img)  # 传递img参数
        # 只有当坐标有效时才绘制和添加点
        if x != 0 and y != 0:
            # 记录坐标和时间戳
            newPoints.append([x, y, count, time.time()])
        count += 1
    return newPoints


def getContours(img, imgResult):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = 0, 0, 0, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 可以调整这个阈值以适应不同大小的物体
        if area > 500:
            cv2.drawContours(imgResult, cnt, -1, (0, 255, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            # 返回中心点坐标
            return x + w // 2, y + h // 2

    # 如果没找到合适的轮廓，返回(0,0)
    return 0, 0


# FIXME
# def drawpoints(mypoints, mycolorvalues, imgResult):
#     for point in mypoints:
#         cv2.circle(imgResult, (point[0], point[1]), 10, mycolorvalues[point[2]], cv2.FILLED)


# 添加一个函数用于校准颜色
def checkcolor(cap):
    print("enter 's' get fps start check，enter 'q' exit")
    while True:
        success, img = cap.read()
        if not success:
            print("无法读取摄像头画面")
            break

        imgResult = img.copy()
        cv2.imshow("enter s, 'q' exit", imgResult)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # 在画面中央取一个小区域进行颜色分析
            h, w = img.shape[:2]
            center_x, center_y = w // 2, h // 2
            roi = img[center_y - 20:center_y + 20, center_x - 20:center_x + 20]

            # 转换为HSV
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # 计算平均HSV值
            avg_hsv = np.mean(hsv_roi, axis=(0, 1))
            print(f"中心区域平均HSV值: {avg_hsv}")

            # 建议的颜色范围（可以手动调整）
            h_min, s_min, v_min = np.maximum(avg_hsv - [10, 50, 50], [0, 0, 0])
            h_max, s_max, v_max = np.minimum(avg_hsv + [10, 50, 50], [179, 255, 255])

            print(f"建议颜色范围: [{int(h_min)}, {int(s_min)}, {int(v_min)}, {int(h_max)}, {int(s_max)}, {int(v_max)}]")

        elif key == ord('q'):
            break


# 主循环
def main():
    print("c分析画面c键点完s键分析颜色给出建议颜色,只检测橙紫绿蓝色其他色不检测，按 'q' 键退出")
    frameWidth = 640
    frameHeight = 480
    cap = cv2.VideoCapture(0)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
    cap.set(10, 150)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    mypoints = []
    while True:
        success, img = cap.read()
        if not success:
            print("无法读取摄像头画面")
            break

        imgResult = img.copy()
        newPoints = findColor(img, myclolors, mycolorvalues)
        nowtime = time.time()

        # 添加新的点
        for point in newPoints:
            mypoints.append(point)

        # 过滤并绘制点
        updatepoints = []
        for point in mypoints:
            if nowtime - point[3] < 3:  # 保留3秒内的点
                updatepoints.append(point)
                cv2.circle(imgResult, (point[0], point[1]), 10, mycolorvalues[point[2]], cv2.FILLED)
        mypoints[:] = updatepoints

        cv2.imshow("colorcheck", imgResult)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            checkcolor(cap)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
