import numpy as np
import cv2
import math

def read_focal_lens(img_dir):
    focal_lens = []
    with open(img_dir+'/pano.txt', 'r') as f:
        pano_txt = [line for line in f.read().splitlines() if line.strip()!='']
        focal_lens.append(float(pano_txt[-1]))
        for i, line in enumerate(pano_txt):
            if i != 0 and line[0]=='C':
                focal_lens.append(float(pano_txt[i-1]))
        
    return np.array(focal_lens)

def create_cyli_projs(img_dir, imgs):
    focal_lens = read_focal_lens(img_dir)
    cyl_projs = []
    for i, img in enumerate(imgs):
        print(f'Create cylindrical projection of image {i+1}')
        cyl_proj = np.zeros(shape=img.shape, dtype=int)
        f = focal_lens[i]
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                rgb = img[y][x]
                Xc = int(img.shape[1]/2)
                Yc = int(img.shape[0]/2)
                x_hat = x-Xc
                y_hat = y-Yc
                proj_x = round(f * math.atan(x_hat/f)) + Xc
                proj_y = round(f*y_hat/(math.sqrt(x_hat**2+f**2))) + Yc

                if proj_x >= 0 and proj_y >= 0 and proj_x < img.shape[1] and proj_y < img.shape[0]:
                    cyl_proj[proj_y][proj_x] = rgb
        cv2.imwrite("cylindrical"+str(i)+".jpg",cyl_proj)
        cyl_projs.append(cyl_proj)
    
    return cyl_projs
