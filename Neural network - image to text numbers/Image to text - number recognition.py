# Import libraries
print("Importing libraries...")
import cv2 as cv
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random as rng
import skimage.measure
matplotlib.use('TkAgg')

# To do
print("""
To do:
    Investigate why prediction of a number 9 is so bad
    """)

# User defined functions

def image_circles(cent, rad, image, col_type):
    for i in range(len(cent)):
        if col_type is 'rand':
            color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        elif col_type is 'black':
            color = (0,0,0)
        cv.rectangle(image, (cent[i][0]-rad[i],cent[i][1]-rad[i]), (cent[i][0]+rad[i],cent[i][1]+rad[i]), color, 3)
    return image

def enclosing_circles(image):
    canny_output = cv.Canny(image, 0, 255)
    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours_poly = [None]*len(contours)
    cent = [None]*len(contours)
    rad = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect = cv.boundingRect(contours_poly[i])
        cent[i]=(int(boundRect[0]+0.5*boundRect[2]),int(boundRect[1]+0.5*boundRect[3]))
        rad[i] = int(np.ceil(0.5*max(boundRect[2:4])))
    return cent, rad, canny_output
    
def intersection_area(dist, rad1, rad2):
    if dist <= abs(rad1-rad2):
        area = np.pi * min(rad1, rad2)**2
    elif dist >= rad2 + rad1:
        area = 0
    else:
        rad2_2, rad1_2, dist_2 = rad2**2, rad1**2, dist**2
        alpha = np.arccos((dist_2+rad2_2-rad1_2)/(2*dist*rad2))
        beta=np.arccos((dist_2+rad1_2-rad2_2)/(2*dist*rad1))
        area=rad2_2*alpha+rad1_2*beta-0.5*(rad2_2*np.sin(2*alpha)+rad1_2*np.sin(2*beta))
    return area

def remove_overlapping_circles(cent, rad):
    prop_overlap = np.zeros((len(cent),len(cent)))
    for i in range(len(cent)):
        base_cent, base_rad = cent[i], rad[i]
        for ii in range(len(cent)):
            if ii != i:
                comp_cent, comp_rad = cent[ii], rad[ii]
                dist = np.linalg.norm(np.array(base_cent)-np.array(comp_cent))
                area_overlap = intersection_area(dist, base_rad, comp_rad)
                area_comp = np.pi * comp_rad**2
                prop_overlap[i,ii] = area_overlap/area_comp
    circ_no=0
    while circ_no<=len(rad)-1:
        rad_smallest_ind_arr = np.argsort(rad)
        ind=rad_smallest_ind_arr[circ_no]
        remove = max(prop_overlap[:,ind])>0.5
        if remove:
            prop_overlap = np.delete(prop_overlap, ind, 0)
            prop_overlap = np.delete(prop_overlap, ind, 1)
            del cent[ind]
            del rad[ind]
        else:
            circ_no+=1
    return cent, rad

def item_extract(cent,rad, image):
    x_min, x_max, y_min, y_max = (cent[0]-rad), (cent[0]+rad), (cent[1]-rad), (cent[1]+rad)
    image_item = image[y_min:y_max,x_min:x_max].copy()
    image_item[image_item>120] = 255
    image_item[image_item<=120] = 0
    return image_item

def item_vary(image, AR_fact_arr, bord_fact_arr, rot_arr):
    item_variants = []
    orig_sz = image.shape
    for bord_fact in bord_fact_arr:
        bord_sz = tuple([int(bord_fact*x) for x in orig_sz])
        image_bord = np.zeros(bord_sz)+255
        bord_wid=int((bord_sz[0]-orig_sz[0])/2)
        image_bord[bord_wid:bord_wid+orig_sz[0],bord_wid:bord_wid+orig_sz[1]] = image
        for rot in rot_arr:
            rows,cols = bord_sz
            image_bord_rot = 255-cv.warpAffine(255-image_bord,cv.getRotationMatrix2D((cols/2,rows/2),rot,1), (cols,rows))
            for hor_fact, vert_fact in zip([1]*(len(AR_fact_arr)+1)+AR_fact_arr, AR_fact_arr+[1]*(len(AR_fact_arr)+1)):
                new_size = (int(image_bord_rot.shape[0]*vert_fact), int(image_bord_rot.shape[1]*hor_fact))
                image_bord_rot_AR = cv.resize(image_bord_rot, new_size)
                item_variants.append(image_bord_rot_AR)
    return item_variants

def item_predict_prep(image_in):
    image_out = []
    for image in image_in:
        kernal_size=max([1, int(np.floor(image.shape[0]/28))])
        image_pool = skimage.measure.block_reduce(image, (kernal_size,kernal_size), np.min)
        image_downsize = cv.resize(image_pool, (28,28))
        image_downsize[image_downsize<255] = 0
        image_out.append(image_downsize)
    return image_out

def item_predict(model, image_list):
    image_list_arr=np.transpose(np.stack(image_list, axis=2).reshape(-1,len(image_list)))
    conf_arr_variants = model.predict(255-image_list_arr)
    conf_arr = conf_arr_variants[np.argmax(np.max(conf_arr_variants, axis=1))]
    sort_ind = np.flipud(np.argsort(conf_arr))
    conf_arr = conf_arr[sort_ind]
    conf_tot_ind=(np.cumsum(conf_arr)<0.9).tolist().index(False)
    preds = sort_ind[0:conf_tot_ind+1]
    pred_vals = conf_arr[0:conf_tot_ind+1]
    return preds, pred_vals

# Load trained neural network
model = load_model('model.h5')
model.summary()

# Convert grayscale image and blur it
filename='test.jpg'
# filename='test_full.jpg'
image_raw = cv.imread(filename)
image_blur = cv.blur(cv.imread(filename,0), (3,3))

# Create enclosing circles of each item to be processed
circ_cent, circ_rad, canny = enclosing_circles(image_blur)
image_circ_all = image_circles(circ_cent, circ_rad, np.zeros((image_blur.shape[0], image_blur.shape[1], 3), dtype=np.uint8),'rand')

# Remove overlapping circles
circ_cent, circ_rad = remove_overlapping_circles(circ_cent, circ_rad)
image_circ_fin = image_circles(circ_cent, circ_rad, image_raw.copy(),'black')

# Label each item
predictions = []
for cent, rad in zip(circ_cent, circ_rad):
    image_item_full = item_extract(cent, rad, image_blur)
    image_item_full_variants = item_vary(image_item_full, [0.7, 0.8, 0.9],[1.3, 1.5],[-10, 0, 10])
    image_item_variants = item_predict_prep(image_item_full_variants)
    labels, confs = item_predict(model, image_item_variants)
    predictions.append((image_item_full, labels, confs))

# Display results
figure1, axes1 = plt.subplots(nrows=2, ncols=2)
axes1[0][0].imshow(image_raw)
axes1[0][1].imshow(canny, cmap='gray')
axes1[1][0].imshow(image_circ_all)
axes1[1][1].imshow(image_circ_fin)
figure1.tight_layout()

no_col=4
figure2, axes2 = plt.subplots(nrows=3, ncols=no_col)
for i, (image, labels, confs) in enumerate(predictions):
    print(labels)
    print(confs)
    axes2[(i // no_col)][i % no_col].imshow(image, cmap='gray')
    axes2[(i // no_col)][i % no_col].set_title(",".join([f"{i}" for i in labels]) + " (" + ",".join([f"{100*i:.0f}" for i in confs]) + ")")
figure2.tight_layout()

# no_col=7
# figure3, axes3 = plt.subplots(nrows=6, ncols=no_col)
# for i, image in enumerate(image_item_variants):
#     axes3[(i // no_col)][i % no_col].imshow(image, cmap='gray')
# figure3.tight_layout()

plt.show()