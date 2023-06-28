import os
import cv2
import torch
import numpy as np
from torchvision.transforms import functional as F
from matplotlib import pyplot as plt
from dataset import get_data_transforms, MVTecDataset
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from scipy.ndimage import gaussian_filter
from test import cvt2heatmap, show_cam_on_image, min_max_norm, cal_anomaly_map

def path_test(input_img_path):
    # Extract class, cause, and num from input_img_path
    components = input_img_path.split('/')
    _class_ = components[-4]
    _cause_ = components[-2]
    _num_ = components[-1].split('.')[0]

    # Rest of the code...
    print("class: ",_class_)
    print("cause: ",_cause_)
    print("num: ",_num_)

def extract_info(input_img_path):
    components = input_img_path.split('/')
    _class_ = components[-4]
    _defect_type_ = components[-2]
    _num_ = components[-1].split('.')[0]
    return _class_,_defect_type_,_num_

def visualizationS3(input_img_path):
    # Extract class, defect_type, and num from input_img_path
    _class_, _defect_type_, _num_ = extract_info(input_img_path)

    #print("class: ",_class_)
    #print("defect_type: ",_defect_type_)
    #print("num: ",_num_)

    data_transform, gt_transform = get_data_transforms(256, 256)
    test_path = './mvtec/' + _class_
    ckp_path = './checkpoints/' + 'wres50_' + _class_ + '.pth'
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform,\
                              phase="test", defect_type=_defect_type_, num=_num_)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    # Load model
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)

    # Load checkpoint
    ckp_path = f'./checkpoints/wres50_{_class_}.pth' 
    ckp = torch.load(ckp_path,map_location=torch.device('cpu'))
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])

    with torch.no_grad():
        for img, gt, label, _ in test_dataloader:
            if (label.item() == 0): # good 이면 anomaly segmentation 하지 않음
                #print(f'{_class_}_{_num_} -> good product')
                continue
            decoder.eval()
            bn.eval()
            inputs = encoder(img)
            outputs = decoder(bn(inputs))

            anomaly_map, amap_list = cal_anomaly_map([inputs[-1]], [outputs[-1]], img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            ano_map = min_max_norm(anomaly_map)
            ano_map = cvt2heatmap(ano_map*255)
            
            img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            img = np.uint8(min_max_norm(img)*255)
            ano_map = show_cam_on_image(img, ano_map)

            # Save anomaly map image
            output_dir = './seg_result'  
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'{_class_}{_num_}_caused_by_{_defect_type_}.png')
            cv2.imwrite(output_path, ano_map)


