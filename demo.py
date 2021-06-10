import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
# from utils import flow_viz
from utils.utils import InputPadder
from utils.frame_utils import writeFlow


DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    os.makedirs(os.path.join(args.path, 'flow_fw'), exist_ok=True)
    os.makedirs(os.path.join(args.path, 'flow_bw'), exist_ok=True)
    print('Predicting optical flow ...')

    with torch.no_grad():
        images = sorted(glob.glob(os.path.join(args.path, 'images/*')))
        
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_fw = model(image1, image2, iters=20, test_mode=True)
            _, flow_bw = model(image2, image1, iters=20, test_mode=True)

            writeFlow(imfile1.replace('images', 'flow_fw')[:-3]+'flo', flow_fw[0].permute(1,2,0).cpu().numpy())
            writeFlow(imfile2.replace('images', 'flow_bw')[:-3]+'flo', flow_bw[0].permute(1,2,0).cpu().numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
