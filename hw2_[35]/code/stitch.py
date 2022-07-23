import numpy as np

#cover over 100 blend 100 else blend all
def stitch(shifts, cyl_projs, img_size, height, width):
    shifts = np.asarray(shifts)
    new_img_x = width - shifts[:,0].sum()
    abs_y = []
    abs_x = []
    for i in range(shifts[:,1].shape[0]):
        abs_y.append(shifts[:i+1,1].sum())
        abs_x.append(-shifts[:i+1,0].sum())
        
    #let min y on top
    new_img_y = height + abs(max(abs_y)) + abs(min(abs_y))
    if min(abs_y) < 0:
        abs_y = abs_y + abs(min(abs_y))
    new_img = np.zeros((new_img_y, new_img_x,3))
    
    #first_pic
    new_img[abs_y[0]:abs_y[0]+height][:, :width] = cyl_projs[0].copy()
    
    for img_idx in range(1, img_size):
        print(f'Stitching image {img_idx} and image {img_idx+1}')
        img_with_blend = cyl_projs[img_idx].copy()
        shift_x = shifts[img_idx][0]
        shift_y = shifts[img_idx][1]
        if shift_y >= 0:
            blend_prev = new_img[abs_y[img_idx-1]:abs_y[img_idx-1]+height][:, abs_x[img_idx-1]:abs_x[img_idx-1]+width][shift_y:][:,-shift_x:].copy()
            blend_now = cyl_projs[img_idx][:height-shift_y][:,:width+shift_x].copy()
        else:
            blend_prev = new_img[abs_y[img_idx-1]:abs_y[img_idx-1]+height][:, abs_x[img_idx-1]:abs_x[img_idx-1]+width][:height+shift_y][:,-shift_x:].copy()
            blend_now = cyl_projs[img_idx][-shift_y:][:,:width+shift_x].copy()
        
        covered = shift_x+width
        blend = np.zeros(blend_prev.shape)
        
        if covered < 500:
            for blend_y in range(blend.shape[0]):
                for blend_x in range(blend.shape[1]):
                    alpha = (blend.shape[1]-blend_x)/blend.shape[1]
                    blend[blend_y][blend_x] += (alpha*blend_prev[blend_y][blend_x] + (1-alpha)*blend_now[blend_y][blend_x]).astype(np.uint8)
        else:
            
            left = int((covered-500)/2)
            right = int(covered-(covered-500)/2)
            for blend_y in range(blend.shape[0]):
                for blend_x in range(blend.shape[1]):
                    if blend_x <= left:
                        blend[blend_y][blend_x] += blend_prev[blend_y][blend_x]
                    elif blend_x > right:
                        blend[blend_y][blend_x] += blend_now[blend_y][blend_x]
                    else:
                        
                        alpha = (blend_x-left)/(right-left)
                        blend[blend_y][blend_x] += ((1-alpha)*blend_prev[blend_y][blend_x] + alpha*blend_now[blend_y][blend_x]).astype(int)
        
        if shift_y >= 0:
            img_with_blend[:height-shift_y][:,:width+shift_x] = blend.copy()
        else:
            img_with_blend[-shift_y:][:,:width+shift_x] = blend.copy()
        
        new_img[abs_y[img_idx]:abs_y[img_idx]+height][:, abs_x[img_idx]:abs_x[img_idx]+width] = img_with_blend.copy()
        
    return new_img
    