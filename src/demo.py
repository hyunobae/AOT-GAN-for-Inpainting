import cv2
import os
import importlib
import numpy as np
from glob import glob 

import torch
from torchvision.transforms import ToTensor

from utils.option import args
from utils.painter import Sketcher
from PIL import Image



def postprocess(image):
    image = torch.clamp(image, -1., 1.)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return image



def demo(args):
    # load images 
    img_list = []
    for ext in ['*.jpg', '*.png']: 
        img_list.extend(glob(os.path.join("images", ext)))
    img_list.sort()

    mask_list = []
    for ext in ['*.jpg', '*.png']:
        mask_list.extend(glob(os.path.join("mask", ext)))

    # Model and version
    net = importlib.import_module('model.'+args.model)
    model = net.InpaintGenerator(args)
    model.load_state_dict(torch.load("../experiments/G0000000.pt", map_location='cpu'))
    model.eval()

    for (fn, mk) in zip(img_list, mask_list):
        filename = os.path.basename(fn).split('.')[0]
        orig_img = cv2.imread(fn, cv2.IMREAD_COLOR)
        img_tensor = (ToTensor()(orig_img) * 2.0 - 1.0).unsqueeze(0)
        h, w, c = orig_img.shape
        mask = np.zeros([h, w, 1], np.uint8)

        while True:
            print('[**] inpainting ... ')
            print(f"{mask.shape}")

            with torch.no_grad():
                print(f"mask info: {type(mask), mask.shape}")
                mname = os.path.basename(mk).split('.')[0]
                mask = Image.open(mname)
                mask_tensor = (ToTensor()(mask)).unsqueeze(0)
                masked_tensor = (img_tensor * (1 - mask_tensor).float()) + mask_tensor

                pred_tensor = model(masked_tensor, mask_tensor)
                comp_tensor = (pred_tensor * mask_tensor + img_tensor * (1 - mask_tensor))

                pred_np = postprocess(pred_tensor[0])
                masked_np = postprocess(masked_tensor[0])
                comp_np = postprocess(comp_tensor[0])

                cv2.imwrite(filename+"_inpainted.png", comp_np)
                print('inpainting finish!')
                print('[**] save successfully!')


if __name__ == '__main__':
    demo(args)
