import os
import cv2
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse

classes = ['ore carrier', 'fishing boat', 'passenger ship', 'general cargo ship', 'bulk cargo carrier', 'container ship']

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', default='',type=str, help="path of images")
parser.add_argument('--label_path', default='',type=str, help="path of labels .txt")
parser.add_argument('--save_path', type=str,default='data.json', help="if not split the dataset, give a path to a json file")
arg = parser.parse_args()

def yolo2coco(arg):
    print("Loading data from ", arg.image_path, arg.label_path)

    assert os.path.exists(arg.image_path)
    assert os.path.exists(arg.label_path)
    
    originImagesDir = arg.image_path
    originLabelsDir = arg.label_path
    indexes = os.listdir(originImagesDir)

    dataset = {'categories': [], 'annotations': [], 'images': []}
    for i, cls in enumerate(classes, 0):
        dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
    
    # 标注的id从0开始
    ann_id_cnt = 0
    for k, index in enumerate(tqdm(indexes)):
        # 支持 png, jpg, irp 等格式的图片.
        # k 将作为这张图片的唯一整数ID
        image_id = k
        
        # 构造标签文件名
        txtFile = f'{index[:index.rfind(".")]}.txt'

        # 读取图像的宽和高
        try:
            im = cv2.imread(os.path.join(originImagesDir, index))
            height, width, _ = im.shape
        except Exception as e:
            print(f'Error reading {os.path.join(originImagesDir, index)}: {e}')
            continue # 如果图片读取失败，跳过这张图

        # 将图片信息首先添加进去，并使用整数k作为id
        dataset['images'].append({
            'file_name': index,
            'id': image_id,  # 使用整数 ID
            'width': width,
            'height': height
        })

        # 如果对应的标签文件不存在，则跳过后续的标注处理步骤
        if not os.path.exists(os.path.join(originLabelsDir, txtFile)):
            continue
            
        with open(os.path.join(originLabelsDir, txtFile), 'r') as fr:
            labelList = fr.readlines()
            for label in labelList:
                label = label.strip().split()
                x = float(label[1])
                y = float(label[2])
                w = float(label[3])
                h = float(label[4])

                # convert x,y,w,h to x1,y1,x2,y2
                H, W, _ = im.shape
                x1 = (x - w / 2) * W
                y1 = (y - h / 2) * H
                x2 = (x + w / 2) * W
                y2 = (y + h / 2) * H
                
                cls_id = int(label[0])
                bbox_width = max(0, x2 - x1)
                bbox_height = max(0, y2 - y1)

                # 在annotations中，使用整数k作为image_id
                dataset['annotations'].append({
                    'area': bbox_width * bbox_height,
                    'bbox': [x1, y1, bbox_width, bbox_height],
                    'category_id': cls_id,
                    'id': ann_id_cnt,
                    'image_id': image_id,  # 使用和上面images中对应的整数 ID
                    'iscrowd': 0,
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })
                ann_id_cnt += 1

    # 保存结果
    with open(arg.save_path, 'w') as f:
        json.dump(dataset, f)
        print('Save annotation to {}'.format(arg.save_path))

if __name__ == "__main__":
    yolo2coco(arg)
