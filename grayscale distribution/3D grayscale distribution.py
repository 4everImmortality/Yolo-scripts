import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
import plotly.graph_objects as go
import os

# 全局变量
rois = []
selecting = False
start_x, start_y = -1, -1
img_display = None
original_gray = None
scale_factor = 1.0  # 缩放比例
offset_x, offset_y = 0, 0  # 平移偏移量
image_filename = ""  # 保存图片文件名（不含路径和扩展名）

# 鼠标回调函数，用于选择 ROI 和拖动图像
def mouse_callback(event, x, y, flags, param):
    global selecting, start_x, start_y, rois, img_display, offset_x, offset_y

    if event == cv2.EVENT_LBUTTONDOWN:
        if not selecting:
            selecting = True
            start_x, start_y = x, y
            cv2.circle(img_display, (x, y), 2, (0, 255, 0), -1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if selecting:
            img_temp = img_display.copy()
            cv2.rectangle(img_temp, (start_x, start_y), (x, y), (0, 255, 0), 1)
            cv2.imshow("Select ROI", img_temp)
        elif flags & cv2.EVENT_FLAG_LBUTTON:  # 拖动图像
            new_offset_x = offset_x + (x - start_x)
            new_offset_y = offset_y + (y - start_y)
            offset_x, offset_y = new_offset_x, new_offset_y
            update_display()

    elif event == cv2.EVENT_LBUTTONUP:
        if selecting:
            selecting = False
            end_x, end_y = x, y
            x1, x2 = min(start_x, end_x), max(start_x, end_x)
            y1, y2 = min(start_y, end_y), max(start_y, end_y)
            rois.append((x1, y1, x2, y2))
            cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.imshow("Select ROI", img_display)

    elif event == cv2.EVENT_MOUSEWHEEL:  # 鼠标滚轮缩放
        global scale_factor
        if flags > 0:  # 滚轮向上，放大
            scale_factor *= 1.1
        else:  # 滚轮向下，缩小
            scale_factor /= 1.1
        scale_factor = max(0.1, min(scale_factor, 10.0))  # 限制缩放范围
        update_display()

# 更新显示图像（缩放和平移）
def update_display():
    global img_display, original_gray, scale_factor, offset_x, offset_y
    height, width = original_gray.shape
    new_width, new_height = int(width * scale_factor), int(height * scale_factor)
    resized_gray = cv2.resize(original_gray, (new_width, new_height), interpolation=cv2.INTER_AREA)
    img_display = cv2.cvtColor(resized_gray, cv2.COLOR_GRAY2BGR)

    # 绘制已选 ROI（根据缩放和平移调整坐标）
    for (x1, y1, x2, y2) in rois:
        scaled_x1 = int(x1 * scale_factor) + offset_x
        scaled_y1 = int(y1 * scale_factor) + offset_y
        scaled_x2 = int(x2 * scale_factor) + offset_x
        scaled_y2 = int(y2 * scale_factor) + offset_y
        cv2.rectangle(img_display, (scaled_x1, scaled_y1), (scaled_x2, scaled_y2), (0, 255, 0), 1)

    # 限制显示区域
    h, w = img_display.shape[:2]
    if offset_x > 0: offset_x = 0
    if offset_y > 0: offset_y = 0
    if offset_x < -(w - width): offset_x = -(w - width)
    if offset_y < -(h - height): offset_y = -(h - height)

    cv2.imshow("Select ROI", img_display[offset_y:offset_y+height, offset_x:offset_x+width])

# 重置 ROI
def reset_rois():
    global rois, img_display, scale_factor, offset_x, offset_y
    rois = []
    scale_factor = 1.0
    offset_x, offset_y = 0, 0
    img_display = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)
    cv2.imshow("Select ROI", img_display)

# 绘制并保存交互式三维灰度分布图到 result/ 文件夹
def plot_3d_gray_distribution(image, roi, roi_index):
    x1, y1, x2, y2 = roi
    orig_x1 = int(x1 / scale_factor - offset_x / scale_factor)
    orig_y1 = int(y1 / scale_factor - offset_y / scale_factor)
    orig_x2 = int(x2 / scale_factor - offset_x / scale_factor)
    orig_y2 = int(y2 / scale_factor - offset_y / scale_factor)
    roi_image = image[orig_y1:orig_y2, orig_x1:orig_x2]
    roi_normalized = roi_image.astype(float) / 255.0  # 归一化到 [0, 1]

    height, width = roi_normalized.shape
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    Z = roi_normalized

    # 创建 Plotly 三维表面图
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Inferno', showscale=True)])

    # 设置 Z 轴范围
    fig.update_layout(
        scene=dict(
            zaxis=dict(range=[0, 1]),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5)
        )
    )

    # 确保 result 文件夹存在，若不存在则创建
    result_dir = "result"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 使用图片文件名生成 HTML 文件名
    global image_filename
    output_file = os.path.join(result_dir, f'{image_filename}_roi_{roi_index + 1}.html')
    fig.write_html(output_file)
    print(f"交互式三维图已保存为: {output_file}")

# 使用资源管理器选择图片文件
def select_image_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="选择图片文件",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("All files", "*.*")]
    )
    root.destroy()
    return file_path

# 主函数
def main():
    global img_display, original_gray, image_filename

    # 使用 GUI 选择图片文件
    image_path = select_image_file()
    if not image_path:
        print("未选择图片文件，程序退出。")
        return

    image = cv2.imread(image_path)
    if image is None:
        print("错误：无法加载图片，请检查路径是否正确！")
        return

    # 提取图片文件名（不含路径和扩展名）
    image_filename = os.path.splitext(os.path.basename(image_path))[0]

    # 如果不是灰度图，转换为灰度图
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    original_gray = gray_image.copy()
    img_display = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    # 创建窗口并绑定鼠标回调
    cv2.namedWindow("Select ROI")
    cv2.setMouseCallback("Select ROI", mouse_callback)

    print("操作说明：")
    print("- 鼠标左键按住并拖动选择 ROI")
    print("- 鼠标滚轮缩放图像（向上放大，向下缩小）")
    print("- 按 'r' 重置 ROI 和缩放")
    print("- 按 'q' 结束选择并生成三维图")

    while True:
        cv2.imshow("Select ROI", img_display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 退出
            break
        elif key == ord('r'):  # 重置
            reset_rois()
        elif key == ord('+'):  # 放大
            scale_factor *= 1.1
            update_display()
        elif key == ord('-'):  # 缩小
            scale_factor /= 1.1
            update_display()

    cv2.destroyAllWindows()

    # 对每个 ROI 生成三维灰度分布图并保存为 HTML
    for i, roi in enumerate(rois):
        plot_3d_gray_distribution(original_gray, roi, i)

if __name__ == "__main__":
    main()