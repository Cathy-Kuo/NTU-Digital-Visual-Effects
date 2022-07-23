import cv2
import numpy as np
import math
from argparse import ArgumentParser, Namespace
np.seterr(divide='ignore',invalid='ignore')

def gaussian(img, sigma):
    return 1 / (sigma * math.sqrt(2 * math.pi)) * np.exp(-(img ** 2 / ((sigma ** 2) * 0.5)))

def bilateral_filter(hdr, sigma_s, sigma_r, kernel_size):
    img_filter = np.zeros_like(hdr)
    hdr_x, hdr_y = hdr.shape

    for x in range(kernel_size // 2, hdr_x - kernel_size // 2):
        for y in range(kernel_size // 2, hdr_y - kernel_size // 2):
            Wp = 0
            val = 0
            for i in range(kernel_size):
                for j in range(kernel_size):
                    nb_x = (x - kernel_size//2 + i)
                    nb_y = (y - kernel_size//2 + j)

                    Gs = gaussian(np.sqrt((x-nb_x) ** 2 + (y-nb_y) ** 2), sigma_s)
                    Gr = gaussian(hdr[nb_x][nb_y] - hdr[x][y], sigma_r)

                    w = Gs * Gr
                    val += Gs * Gr * hdr[nb_x][nb_y]
                    Wp += w
                    
            val = val / Wp
            img_filter[x][y] = val

    return img_filter
 

def main(args):
    filename = args.hdr_filename
    compression_factor = args.compression_factor
    ldr_scale_factor = args.ldr_scale_factor
    sigma_s = args.sigma_s
    sigma_r = args.sigma_r
    kernel_size = args.kernel_size

    hdr = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)
    ldr = np.zeros_like(hdr)
    b, g, r = 0.1, 0.65, 0.25
        
    Lw = (hdr[:, :, 0] * b + hdr[:, :, 1] * g + hdr[:, :, 2] * r)
    large_scale = bilateral_filter(Lw, sigma_s, sigma_r, kernel_size)
    detail = Lw / large_scale

    large_scale_reduce = large_scale * compression_factor
    Ld = large_scale_reduce * detail

    ldr[:, :, 0] = (hdr[:, :, 0] / Lw) * Ld
    ldr[:, :, 1] = (hdr[:, :, 1] / Lw) * Ld
    ldr[:, :, 2] = (hdr[:, :, 2] / Lw) * Ld

    ldr_out = np.clip((ldr * ldr_scale_factor) * 255, 0, 255)

    cv2.imwrite("out/ldr-bilateral.jpg",ldr_out)



def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--hdr_filename",
        type=str,
        help="Path to the hdr file.",
    )
    parser.add_argument(
        "--compression_factor",
        type=float,
        help="Reduce contrast factor",
        default=0.1,
    )
    parser.add_argument(
        "--ldr_scale_factor",
        type=int,
        help="The factor multiply by ldr for final output",
        default=60,
    )
    parser.add_argument(
        "--sigma_s",
        type=int,
        help="The sigma of space.",
        default=1,
    )
    parser.add_argument(
        "--sigma_r",
        type=int,
        help="The sigma of intensity.",
        default=1,
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        help="The kernel (window) size",
        default=5,
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)