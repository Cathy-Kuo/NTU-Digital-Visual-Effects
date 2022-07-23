import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import argparse
import random
import cv2
import os

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [54, 183, 19]) / 256

def binarize(im, threshold):
    im = rgb2gray(np.array(im)).astype('uint8')
    h, w = im.shape
    mask = np.ones(im.shape).astype('uint8')
    for i in range(h):
        for j in range(w):
            if im[i][j] > threshold - 10 and im[i][j] < threshold + 10:
                mask[i][j] = 0
            im[i][j] = 1 if (im[i][j] >= threshold) else 0
    return im, mask

def calculate_loss(im1, im2, mask, x, y):
    loss = 0
    h, w = im1.shape
    for i in range(h):
        for j in range(w):
            if i+x < 0 or i+x >= h or j+y < 0 or j+y >= w: # out of bounds
                continue
            loss += (im1[i][j] ^ im2[i+x][j+y]) & mask[i][j]
    return loss

def offset(im, x, y):
    im = np.array(im)
    h, w, c = im.shape
    out = np.zeros(im.shape).astype('uint8')
    for i in range(h):
        for j in range(w):
            if i+x < 0 or i+x >= h or j+y < 0 or j+y >= w: # out of bounds
                continue
            out[i, j, :] = im[i+x, j+y, :]
    return Image.fromarray(out)

def mtb_alignment(im1, im2):
    x, y = 0, 0 # offset
    for scale in [32, 16, 8, 4, 2, 1]:
        print(f'scale: 1/{scale}')
        # binarize image in a certain scale
        h, w = im1.size[0] // scale, im1.size[1] // scale
        im1_bin = im1.resize((h, w))
        im1_bin, mask = binarize(im1_bin, np.median(im1_bin))
        im2_bin = im2.resize((h, w))
        im2_bin, _ = binarize(im2_bin, np.median(im2_bin))

        # update offset
        x, y = x*2, y*2 # image size is doubled after each iteration
        best_x, best_y, best_loss = x, y, float('inf')
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                loss = calculate_loss(im1_bin, im2_bin, mask, x+dx, y+dy)
                print(f'({dx}, {dy}): \t{loss}')
                if loss < best_loss:
                    best_x, best_y, best_loss = x+dx, y+dy, loss

        x, y = best_x, best_y
        print(f'-------({x}, {y})-------')
    print(f'offset: ({x}, {y})')
    return offset(im2, x, y)

def sample_points(images, npoints):
    random.seed(0) # for reproducibility
    images_np = []
    for image in images:
        images_np.append(np.array(image))

    Z = np.zeros((npoints, len(images), 3), dtype='uint8')
    for i in range(npoints):
        x = random.randint(0, images_np[0].shape[0])
        y = random.randint(0, images_np[0].shape[1])
        for j in range(len(images)):
            Z[i, j, :] = images_np[j][x, y, :]
    return Z

def plot_sampled_points(Z, B, args):
    for i in range(args.npoints):
        plt.scatter(B, Z[i, :, 0], s=20)
    plt.xlabel('log exposure X')
    plt.ylabel('pixel value Z')
    plt.title('Red')
    plt.savefig(os.path.join(args.outdir, 'scatter_red.jpg'))

    plt.clf()
    for i in range(args.npoints):
        plt.scatter(B, Z[i, :, 1], s=20)
    plt.xlabel('log exposure X')
    plt.ylabel('pixel value Z')
    plt.title('Green')
    plt.savefig(os.path.join(args.outdir, 'scatter_green.jpg'))

    plt.clf()
    for i in range(args.npoints):
        plt.scatter(B, Z[i, :, 2], s=20)
    plt.xlabel('log exposure X')
    plt.ylabel('pixel value Z')
    plt.title('Blue')
    plt.savefig(os.path.join(args.outdir, 'scatter_blue.jpg'))

def get_weight_function():
    w = np.zeros((256))
    for i in range(256):
        w[i] = i if i <= 127 else 255 - i
    return w

def plot_weight_function(w, args):
    plt.clf()
    plt.plot(range(256), w,)
    plt.xlabel('pixel value Z')
    plt.ylabel('w(Z)')
    plt.savefig(os.path.join(args.outdir, 'weight_function.jpg'))

def gsolve(Z, B, l, w):
    # solve response curve
    n = 256
    npoints, nimages = Z.shape
    A = np.zeros((npoints*nimages+n+1, n+npoints))
    b = np.zeros((A.shape[0], 1))

    k = 0
    for i in range(npoints):
        for j in range(nimages):
            wij = w[Z[i, j]]
            A[k, Z[i, j]] = wij
            A[k, n+i] = -wij
            b[k, 0] = wij * B[j]
            k += 1

    A[k, 128] = 1
    k += 1

    for i in range(n-2):
        A[k, i] = l * w[i+1]
        A[k, i+1] = -2 * l * w[i+1]
        A[k, i+2] = l * w[i+1]
        k += 1

    # Ax = b, x = A_pinv * b (A_pinv: pseudo inverse of A)
    A_pinv = np.linalg.pinv(A)
    x = np.matmul(A_pinv, b).reshape(-1)
    g, lE = x[:n], x[n:]
    return g, lE

def plot_response_curve(g_red, g_green, g_blue, args):
    plt.clf()
    plt.plot(g_red, range(256), c='red')
    plt.plot(g_green, range(256), c='green')
    plt.plot(g_blue, range(256), c='blue')
    plt.xlabel('log exposure X')
    plt.ylabel('pixel value Z')
    plt.title('RGB curves')
    plt.xlim(-6, 3)
    plt.savefig(os.path.join(args.outdir, 'g.jpg'))

    plt.clf()
    plt.plot(g_red, range(256), c='red')
    plt.xlabel('log exposure X')
    plt.ylabel('pixel value Z')
    plt.title('Red')
    plt.xlim(-6, 3)
    plt.savefig(os.path.join(args.outdir, 'g_red.jpg'))

    plt.clf()
    plt.plot(g_green, range(256), c='green')
    plt.xlabel('log exposure X')
    plt.ylabel('pixel value Z')
    plt.title('Green')
    plt.xlim(-6, 3)
    plt.savefig(os.path.join(args.outdir, 'g_green.jpg'))

    plt.clf()
    plt.plot(g_blue, range(256), c='blue')
    plt.xlabel('log exposure X')
    plt.ylabel('pixel value Z')
    plt.title('Blue')
    plt.xlim(-6, 3)
    plt.savefig(os.path.join(args.outdir, 'g_blue.jpg'))

def get_radiance_map(images, B, w, g):
    images_np = []
    for image in images:
        images_np.append(np.array(image))

    radiance_map = np.zeros(images_np[0].shape)
    for x in range(images_np[0].shape[0]):
        for y in range(images_np[0].shape[1]):
            for z in range(images_np[0].shape[2]):
                num, den = 0, 0
                for j in range(len(images)):
                    Zij = images_np[j][x, y, z]
                    num += w[Zij] * (g[z][Zij] - B[j])
                    den += w[Zij]
                radiance_map[x, y, z] = num / den
    return np.exp(radiance_map)

def plot_radiance_map(radiance_map, args):
    plt.clf()
    plt.imshow(np.log(rgb2gray(radiance_map)), cmap='jet')
    plt.colorbar()
    plt.savefig(os.path.join(args.outdir, 'radiance_map.jpg'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='.txt file that lists image paths and shutter speeds')
    parser.add_argument('outdir', type=str, help='output directory')
    parser.add_argument('--alignment', dest='alignment', action='store_true', help='whether to align images')
    parser.add_argument('--npoints', type=int, default=50, help='number of pixels for solving g-curve')
    parser.add_argument('--l', type=float, default=30., help='smoothness constraint for solving g-curve')
    parser.set_defaults(alignment=False)
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    # read images
    images, log_exposure = [], []
    with open(args.file) as f:
        lines = f.readlines()
        for line in lines:
            image_path, shutter_speed = line.split(' ')
            image = Image.open(image_path)
            images.append(image)
            log_exposure.append(np.log(1/float(shutter_speed)))

    # image alignment
    if args.alignment:
        for i in range(1, len(images)):
            images[i] = mtb_alignment(images[i-1], images[i])

    # solve response curve
    Z = sample_points(images, args.npoints)
    # plot_sampled_points(Z, log_exposure, args)
    w = get_weight_function()
    # plot_weight_function(w, args)
    g_red,   _ = gsolve(Z[:, :, 0], log_exposure, args.l, w) # red g-curve
    g_green, _ = gsolve(Z[:, :, 1], log_exposure, args.l, w) # green g-curve
    g_blue,  _ = gsolve(Z[:, :, 2], log_exposure, args.l, w) # blue g-curve
    plot_response_curve(g_red, g_green, g_blue, args)

    # construct HDR radiance map
    radiance_map = get_radiance_map(images, log_exposure, w, [g_red, g_green, g_blue])
    plot_radiance_map(radiance_map, args)
    cv2.imwrite(os.path.join(args.outdir, 'radiance_map.hdr'), radiance_map[:, :, ::-1])

if __name__ == '__main__':
    main()