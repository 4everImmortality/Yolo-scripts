# evaluate_detr.py
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from tqdm import tqdm
import os

# DETR的输出后处理器，用于将模型输出转换为COCO评估格式
# 来源于DETR官方实现：https://github.com/facebookresearch/detr/blob/main/models/detr.py
class PostProcess(torch.nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the original image size
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = torch.nn.functional.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

# COCO数据集的标准化图像变换
# 来源于DETR官方实现
def make_coco_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')
    
def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = torch.stack(batch[0], 0)
    return tuple(batch)


def main(args):
    # ---- 1. 环境与设备设置 ----
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    # ---- 2. 模型加载 ----
    print("正在加载模型...")
    # 加载一个标准的DETR resnet-50模型结构
    model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=False, num_classes=args.num_classes)
    
    # 加载你自己的权重
    print(f"正在从 '{args.weights_path}' 加载权重...")
    checkpoint = torch.load(args.weights_path, map_location='cpu')
    
    # DETR训练脚本通常会将模型保存在'model'键下
    # 如果你的.pth文件就是模型state_dict本身，使用: model.load_state_dict(checkpoint)
    model.load_state_dict(checkpoint['model'])
    
    model.to(device)
    model.eval()
    
    postprocessor = PostProcess().to(device)
    print("模型加载完成。")

    # ---- 3. 数据集准备 ----
    val_ann_path = os.path.join(args.coco_path, 'annotations', 'instances_val2017.json')
    val_img_folder = os.path.join(args.coco_path, 'val2017')

    print(f"正在从 '{val_img_folder}' 加载COCO验证集...")
    dataset_val = CocoDetection(
        root=val_img_folder, 
        annFile=val_ann_path,
        transform=make_coco_transforms('val')
    )
    
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )
    print("数据集加载完成。")

    # ---- 4. 推理与结果收集 ----
    coco_gt = COCO(val_ann_path)
    coco_results = []
    
    print("开始在验证集上进行推理...")
    for images, targets in tqdm(data_loader_val):
        images = images.to(device)
        
        # 模型推理
        with torch.no_grad():
            outputs = model(images)
        
        # 获取原始图像尺寸
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        
        # 后处理，将模型输出转换为COCO格式
        results = postprocessor(outputs, orig_target_sizes.to(device))
        
        # 格式化为COCO评估工具所需的JSON格式
        for i, res in enumerate(results):
            image_id = targets[i]['image_id'].item()
            for score, label, box in zip(res['scores'], res['labels'], res['boxes']):
                coco_results.append({
                    "image_id": image_id,
                    "category_id": coco_gt.getCatIds()[label.item()], # 将模型标签索引转为COCO类别ID
                    "bbox": [box[0].item(), box[1].item(), (box[2] - box[0]).item(), (box[3] - box[1]).item()], # [x, y, w, h]
                    "score": score.item(),
                })

    if not coco_results:
        print("没有生成任何预测结果，请检查模型或代码。")
        return
        
    print("推理完成，开始进行评估...")

    # ---- 5. 整体COCO指标评估 ----
    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    print("\n" + "="*50)
    print("COCO 整体评估结果:")
    print("="*50)
    coco_eval.summarize()
    
    # ---- 6. 分类别COCO指标评估 ----
    print("\n" + "="*50)
    print("各类别 AP 评估结果 (IoU=0.50:0.95):")
    print("="*50)
    
    cat_ids = coco_gt.getCatIds()
    cats = coco_gt.loadCats(cat_ids)
    cat_names = [cat['name'] for cat in cats]

    for i, cat_id in enumerate(cat_ids):
        # 针对每个类别创建一个新的评估器
        cat_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        cat_eval.params.catIds = [cat_id]
        cat_eval.evaluate()
        cat_eval.accumulate()
        cat_eval.summarize() # 只需从中提取AP值
        ap = cat_eval.stats[0] # stats[0] is AP @ IoU=0.50:0.95
        print(f"  - 类别 '{cat_names[i]}' (ID: {cat_id}) 的 AP: {ap:.4f}")

    print("\n评估脚本执行完毕。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained DETR model on COCO.")
    parser.add_argument('--weights_path', type=str, default='output/dfine_hgnetv2_n_llvip/best_stg1.pth', help="Path to the .pth weights file.")
    parser.add_argument('--coco_path', type=str, default='/root/autodl-tmp/infrared_power-6/annotations/instances_val.json', help="Root path to the COCO dataset directory.")
    parser.add_argument('--num_classes', type=int, default=6, help="Number of classes in the model (including background). For COCO, it's 80+1=91.")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for evaluation.")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use for evaluation ('cuda' or 'cpu').")
    
    args = parser.parse_args()
    main(args)
