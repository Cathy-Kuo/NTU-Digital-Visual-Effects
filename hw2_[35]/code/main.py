import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import os
from argparse import ArgumentParser, Namespace

from create_cylindrical_projection import *
from RANSAC import *
from stitch import *
from feature import *

def main(args):
    #load images
    img_dir = args.image_dir
    files = os.listdir(img_dir)
    files.sort()
    img_filenames = ['./'+img_dir+'/'+f for f in files if ('JPG' in f or 'jpg' in f)]

    imgs = np.array([cv2.imread(img_filename) for img_filename in img_filenames])
    img_size, height, width, _ = imgs.shape
    
    print(f'Create cylindrical projection of {img_size} images ...')
    cyl_projs = create_cyli_projs(img_dir, imgs)

    # feature matching
    cache = []
    all_matched_pairs = []
    for i in range(img_size - 1):
        print(f'[epoch: {i+1}/{img_size-1}]')
        if i == 0:
            print(f'process image #{i+1}')
            img1 = cv2.imread(f'cylindrical{i}.jpg')[:,:,::-1]
            keypoint1 = harris_detector(img1, n=i)
            descriptor1 = sift_descriptor(img1, keypoint1)
        else:
            img1, keypoint1, descriptor1 = cache

        print(f'process image #{i+2}')
        img2 = cv2.imread(f'cylindrical{i+1}.jpg')[:,:,::-1]
        keypoint2 = harris_detector(img2, n=i+1)
        descriptor2 = sift_descriptor(img2, keypoint2)
        cache = [img2, keypoint2, descriptor2]

        print(f'match image #{i+1}, image #{i+2}')
        matched = sift_matching(keypoint1, descriptor1, keypoint2, descriptor2)
        match_visualization(img1, img2, matched, n=i)
        matched = [[[m[0][1], m[0][0]], [m[1][1], m[1][0]]] for m in matched]
        all_matched_pairs.append(matched)
    
    print('Calculate shifts via RANSAC ... ')
    shifts = [[0,0]]
    for i in range(len(all_matched_pairs)):
        shift = RANSAC(all_matched_pairs[i])
        shifts.append(shift)
    
    print(f'Stitching {img_size} images ...')
    stitch_img = stitch(shifts, cyl_projs, img_size, height, width)
    cv2.imwrite('pano.jpg', stitch_img)
    

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--image_dir",
        type=str,
        help="Path to the image file.",
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
