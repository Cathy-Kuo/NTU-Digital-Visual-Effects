import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_gradients(img, ksize=5, sigma=3):
    h, w, _ = img.shape

    gaussian = cv2.getGaussianKernel(ksize, sigma)
    img = cv2.filter2D(img, -1, gaussian)

    I = np.dot(img[...,:3], [54, 183, 19]) / 256 # rgb to gray

    Ix = np.zeros((h, w))
    for j in range(w):
        if not (j == 0 or j == w-1):
            Ix[:,j] = (I[:,j+1] - I[:,j-1]) / 2

    Iy = np.zeros((h, w))
    for i in range(h):
        if not (i == 0 or i == h-1):
            Iy[i,:] = (I[i+1,:] - I[i-1,:]) / 2

    return Ix, Iy

def harris_detector(img, ksize=5, sigma=3, k=0.04, threshold=10, n=0):
    h, w, _ = img.shape

    Ix, Iy = get_gradients(img)
    Ix2 = np.multiply(Ix, Ix)
    Iy2 = np.multiply(Iy, Iy)
    Ixy = np.multiply(Ix, Iy)

    gaussian = cv2.getGaussianKernel(ksize, sigma)
    Sx2 = cv2.filter2D(Ix2, -1, gaussian)
    Sy2 = cv2.filter2D(Iy2, -1, gaussian)
    Sxy = cv2.filter2D(Ixy, -1, gaussian)

    # corner response map
    Rmap = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            M = np.array([[Sx2[i, j], Sxy[i, j]], [Sxy[i, j], Sy2[i, j]]])
            R = np.linalg.det(M) - k * (np.trace(M) ** 2)
            Rmap[i, j] = R if R > threshold else 0

    cv2.imwrite(f'corner_response_map_{n}.png', Rmap)

    # local maximum
    def isMaximum(Rmap, i, j, distance=25):
        for dx in range(-distance, distance+1):
            for dy in range(-distance, distance+1):
                if i+dx < 0 or i+dx > h-1 or j+dy < 0 or j+dy > w-1 or (dx == dy == 0):
                    continue
                if Rmap[i, j] <= Rmap[i+dx, j+dy]:
                    return False
        return True

    keypoint = []
    for i in range(h):
        for j in range(w):
            if Rmap[i, j] > 0 and isMaximum(Rmap, i, j):
                keypoint.append([i, j])

    print(f'num of keypoints: {len(keypoint)}')
    x = [k[1] for k in keypoint]
    y = [k[0] for k in keypoint]
    plt.clf()
    plt.imshow(img)
    plt.scatter(x, y, c='red', s=1)
    plt.savefig(f'keypoint_{n}.png')

    return keypoint

def sift_descriptor(img, keypoint, eps=1e-6, num_bins=8):
    h, w, _ = img.shape
    Ix, Iy = get_gradients(img)

    magnitude = (np.multiply(Ix, Ix) + np.multiply(Iy, Iy)) ** 0.5
    theta = np.arctan(Iy / (Ix + eps)) * (180 / np.pi)
    theta[Ix < 0] += 180 # x > 0, -90 < theta < 90; x < 0, 90 < theta < 270
    theta = (theta + 360) % 360 # -90 < theta < 270 -> 0 < theta < 360

    bin_size = 360 / num_bins
    orientation = (theta // bin_size).astype('int')

    def patch_to_histogram(magnitude, orientation, i, j):
        hist = np.zeros((num_bins))
        for dx in range(4):
            for dy in range(4):
                if i+dx < 0 or i+dx > h-1 or j+dy < 0 or j+dy > w-1:
                    continue
                bin = orientation[i+dx, j+dy]
                hist[bin] += magnitude[i+dx, j+dy]
        return hist

    descriptor = []
    for i, j in keypoint:
        d = []
        for patch_i in range(4):
            for patch_j in range(4):
                start_r = i + 4 * (patch_i - 2)
                start_c = j + 4 * (patch_j - 2)
                hist = patch_to_histogram(magnitude, orientation, start_r, start_c)
                d.append(hist)
        d = np.array(d).reshape(-1) # 128-d vector
        # normalized, clip values larger than 0.2, renormalize
        d = d / np.sqrt(np.sum(d**2))
        d[d > 0.2] = 0.2
        d = d / np.sqrt(np.sum(d**2))
        descriptor.append(d)

    return descriptor

def sift_matching(k1, d1, k2, d2):
    # for each descriptor in d1, find a descriptor in d2 that best matches d1
    matched = []
    for i in range(len(d1)):
        distances = []
        for j in range(len(d2)):
            dist = ((d1[i] - d2[j]) ** 2).sum()
            distances.append(dist)
        idx1, idx2 = np.argsort(distances)[:2]
        if distances[idx1] / distances[idx2] < 0.8:
            matched.append([k1[i], k2[idx1]])
    print(f'num of matched features: {len(matched)}')
    return matched

def match_visualization(img1, img2, matched, n=0):
    img = np.concatenate((img1, img2), axis=1)
    plt.clf()
    plt.imshow(img)
    for k1, k2 in matched:
        x_values = [k1[1], k2[1] + img1.shape[1]]
        y_values = [k1[0], k2[0]]
        plt.plot(x_values, y_values, linewidth=0.5)
    plt.savefig(f'match_{n}.png', dpi=300)

def main():
    print('process image #1')
    img1 = cv2.imread('cylindrical0.jpg')[:,:,::-1]
    keypoint1 = harris_detector(img1, n=0)
    descriptor1 = sift_descriptor(img1, keypoint1)

    print('process image #2')
    img2 = cv2.imread('cylindrical1.jpg')[:,:,::-1]
    keypoint2 = harris_detector(img2, n=1)
    descriptor2 = sift_descriptor(img2, keypoint2)

    print('match image #1, image #2')
    matched = sift_matching(keypoint1, descriptor1, keypoint2, descriptor2)
    match_visualization(img1, img2, matched)

if __name__ == '__main__':
    main()