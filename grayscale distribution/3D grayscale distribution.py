import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
import plotly.graph_objects as go
import os

# 全局变量：所有 ROI 坐标均以原图坐标存储
rois = []               # 存储格式为 (x1, y1, x2, y2)
current_roi = None      # 当前 ROI，格式为 (中心x, 中心y, 边长)（单位：原图像素）
scale_factor = 1.0      # 缩放比例
offset_x, offset_y = 0, 0  # 平移偏移量（如果需要平移时使用，目前保持 0）
img_display = None
original_gray = None
image_filename = ""     # 图片文件名（不含路径和扩展名）

# 鼠标回调函数：左键点击时转换坐标并创建 ROI，鼠标滚轮缩放图像
def mouse_callback(event, x, y, flags, param):
    global current_roi, rois, scale_factor, offset_x, offset_y

    if event == cv2.EVENT_LBUTTONDOWN:
        # 将窗口坐标 (x, y) 转换为原图坐标
        orig_x = int((x + offset_x) / scale_factor)
        orig_y = int((y + offset_y) / scale_factor)
        # 若当前已有 ROI，则先确定它
        if current_roi is not None:
            cx, cy, s = current_roi
            half = s // 2
            finalized = (cx - half, cy - half, cx + half, cy + half)
            rois.append(finalized)
            print("Finalized ROI:", finalized)
        # 以原图坐标为中心创建默认边长为 15px 的 ROI
        current_roi = (orig_x, orig_y, 15)
        print("Created new ROI at ({}, {}) with side length 15".format(orig_x, orig_y))
        update_display()

    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:  # 滚轮向上，放大
            scale_factor *= 1.1
        else:          # 滚轮向下，缩小
            scale_factor /= 1.1
        scale_factor = max(0.1, min(scale_factor, 10.0))
        update_display()

# 更新显示图像：根据当前缩放和平移参数绘制图像和 ROI（均由原图坐标转换到显示坐标）
def update_display():
    global img_display, original_gray, scale_factor, offset_x, offset_y, rois, current_roi
    height, width = original_gray.shape
    new_width, new_height = int(width * scale_factor), int(height * scale_factor)
    resized_gray = cv2.resize(original_gray, (new_width, new_height), interpolation=cv2.INTER_AREA)
    img_display = cv2.cvtColor(resized_gray, cv2.COLOR_GRAY2BGR)

    # 绘制已确定的 ROI（绿色）
    for (x1, y1, x2, y2) in rois:
        scaled_x1 = int(x1 * scale_factor) + offset_x
        scaled_y1 = int(y1 * scale_factor) + offset_y
        scaled_x2 = int(x2 * scale_factor) + offset_x
        scaled_y2 = int(y2 * scale_factor) + offset_y
        cv2.rectangle(img_display, (scaled_x1, scaled_y1), (scaled_x2, scaled_y2), (0, 255, 0), 1)

    # 绘制当前 ROI（绿色）
    if current_roi is not None:
        cx, cy, s = current_roi
        half = s // 2
        x1, y1, x2, y2 = cx - half, cy - half, cx + half, cy + half
        scaled_x1 = int(x1 * scale_factor) + offset_x
        scaled_y1 = int(y1 * scale_factor) + offset_y
        scaled_x2 = int(x2 * scale_factor) + offset_x
        scaled_y2 = int(y2 * scale_factor) + offset_y
        cv2.rectangle(img_display, (scaled_x1, scaled_y1), (scaled_x2, scaled_y2), (0, 255, 0), 1)

    # 显示原图大小区域（cropped显示）
    cv2.imshow("Select ROI", img_display[offset_y:offset_y+height, offset_x:offset_x+width])

# 重置所有 ROI 与缩放参数
def reset_rois():
    global rois, current_roi, img_display, scale_factor, offset_x, offset_y
    rois = []
    current_roi = None
    scale_factor = 1.0
    offset_x, offset_y = 0, 0
    img_display = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)
    cv2.imshow("Select ROI", img_display)

# 绘制并保存交互式三维灰度分布图到 result/ 文件夹
def plot_3d_gray_distribution(image, roi, roi_index):
    # 获取 ROI
    x1, y1, x2, y2 = roi
    roi_image = image[y1:y2, x1:x2]
    roi_normalized = roi_image.astype(float) / 255.0  # 归一化到 [0, 1]

    height, width = roi_normalized.shape
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    Z = roi_normalized

    # 创建 Plotly 三维表面图，并固定颜色范围
    fig = go.Figure(data=[go.Surface(
        z=Z, x=X, y=Y, colorscale='Inferno', showscale=True, cmin=0, cmax=1  # 固定颜色范围
    )])
    fig.update_layout(
        scene=dict(
            zaxis=dict(range=[0, 1]),  # 统一 Z 轴范围
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5)
        )
    )

    # 确保 result 目录存在
    result_dir = "result"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    global image_filename
    output_file = os.path.join(result_dir, f'{image_filename}_roi_{roi_index + 1}.html')
    fig.write_html(output_file)
    print(f"交互式三维图已保存为: {output_file}")


# 使用文件对话框选择图片文件
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
    global img_display, original_gray, image_filename, current_roi
    image_path = select_image_file()
    if not image_path:
        print("未选择图片文件，程序退出。")
        return
    image = cv2.imread(image_path)
    if image is None:
        print("错误：无法加载图片，请检查路径是否正确！")
        return
    image_filename = os.path.splitext(os.path.basename(image_path))[0]

    # 转换为灰度图
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    original_gray = gray_image.copy()
    img_display = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    cv2.namedWindow("Select ROI")
    cv2.setMouseCallback("Select ROI", mouse_callback)

    print("操作说明：")
    print("- 鼠标左键点击图像：以点击位置为中心创建默认边长15px的 ROI（坐标转换到原图）；若已有 ROI，则先确定前一个 ROI。")
    print("- 使用上/下方向键调整当前 ROI 的边长（每次按键会打印出更新后的 ROI 信息）。")
    print("- 鼠标滚轮缩放图像。")
    print("- 按 'r' 重置所有 ROI 和缩放。")
    print("- 按 'q' 结束选择并生成三维图。")

    while True:
        cv2.imshow("Select ROI", img_display)
        key = cv2.waitKeyEx(1)
        if key == ord('q'):
            break
        elif key == ord('r'):
            reset_rois()
        # 上方向键：增大 ROI 边长（扩展按键返回码通常为 2490368）
        elif key == 2490368:
            if current_roi is not None:
                cx, cy, s = current_roi
                s += 1
                current_roi = (cx, cy, s)
                print("Updated ROI at ({}, {}) with new side length: {}".format(cx, cy, s))
                update_display()
        # 下方向键：减小 ROI 边长（扩展按键返回码通常为 2621440）
        elif key == 2621440:
            if current_roi is not None:
                cx, cy, s = current_roi
                if s > 1:
                    s -= 1
                    current_roi = (cx, cy, s)
                    print("Updated ROI at ({}, {}) with new side length: {}".format(cx, cy, s))
                    update_display()

    cv2.destroyAllWindows()

    # 若当前有未确定的 ROI，则最终确定它
    if current_roi is not None:
        cx, cy, s = current_roi
        half = s // 2
        finalized = (cx - half, cy - half, cx + half, cy + half)
        rois.append(finalized)
        print("Finalized ROI:", finalized)

    # 对每个 ROI 生成三维灰度分布图并保存为 HTML 文件
    for i, roi in enumerate(rois):
        plot_3d_gray_distribution(original_gray, roi, i)

if __name__ == "__main__":
    main()
