
# author: Cheng Peng
# create date: April,2014
# brief : research for Minutiae Based Fingerprint Recognition Algorithm

# -----------------------------------------------------------------------------
# import modules

from __future__ import division
import cv2
import cv
import os
import sys
import argparse
import numpy as np
import pylab as pl
from matplotlib import pyplot as plt
from numpy.lib import pad

# -----------------------------------------------------------------------------
# Implementation Methods

def read_image(image_name):
# Load a subject's right hand, second finger image and risize to 352*352

    fingerprint = cv2.imread(image_name, 0)
    fingerprint = cv2.resize(fingerprint,(352,352))
    fpcopy = fingerprint[:]
    row, col = fingerprint.shape

    return row, col, fingerprint, fpcopy


def segment(r, c, finger_print, fp_copy):
# Image segmentation based on variance and mathematical morphology

    W = 16
    threshold = 1600
    A = np.zeros((r,c), np.uint8)

    for i in np.arange(0,r-1,W):
        for j in np.arange(0,c-1,W):
            Mw = (1/(W*W)) * (sum(finger_print[i:i+W,j:j+W]))
            Vw = (1/(W*W)) * (sum((finger_print[i:i+W,j:j+W] - Mw)**2))
            
            if (Vw < threshold).all():
                finger_print[i:i+W,j:j+W] = 0
                A[i:i+W,j:j+W] = 0
            else:
                A[i:i+W,j:j+W] = 1

    kernel = np.ones((44,44),np.uint8)
    closing_mask = cv2.morphologyEx(A, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((88,88),np.uint8)
    opening_mask = cv2.morphologyEx(closing_mask, cv2.MORPH_OPEN, kernel)

    for i in np.arange(0,r-1,W):
        for j in np.arange(0,c-1,W):
            if ((sum(finger_print[i:i+W,j:j+W])) != (sum(opening_mask[i:i+W,j:j+W]))).all():
                if np.mean(opening_mask[i:i+W,j:j+W]) == 1:
                    finger_print[i:i+W,j:j+W] = fp_copy[i:i+W,j:j+W]
                elif  np.mean(opening_mask[i:i+W,j:j+W]) == 0:
                    finger_print[i:i+W,j:j+W] = 0

    return finger_print, opening_mask


def normal(fingerprint, r, c):
# image normalization

    M0 = 100
    VAR0 = 1000
    M = np.mean(fingerprint[:])
    VAR = np.var(fingerprint[:])
    normalization = np.zeros((r,c), np.uint8)

    for i in range(r):
        for j in range(c):
            if (fingerprint[i,j] > M):
                normalization[i,j] = M0 + np.sqrt(VAR0 * ((fingerprint[i,j] - M)**2) / VAR)
            else:
                normalization[i,j] = M0 - np.sqrt(VAR0 * ((fingerprint[i,j] - M)**2) / VAR)

    return normalization


def enhance(r, c, normal):
# image enhancement using FFT

    ImgB = normal
    k = 0.45
    m = r
    n = c
    W = 16
    inner = np.zeros((m,n))

    for i in np.arange(0,m,W):
        for j in np.arange(0,m,W):
            a = i+W;
            b = j+W;
            F = np.fft.fft2(ImgB[i:a,j:b]);
            factor = abs(F)**k;
            block = abs(np.fft.ifft2(F*factor))
            L = list()
            for array in block:
                L.extend(array)
            lmax = max(L);
            if lmax == 0:
                lmax = 1
      
            block = block/lmax;
            inner[i:a,j:b] = block
    image_fft = inner*255

    return image_fft


def adp_binary(fp_fft, m, n):
# adaptive image binarization

    W=16
    I = np.uint8(fp_fft)
    Binary = np.zeros((m,n))
    for i in np.arange(0,m,W):
        for j in np.arange(0,n,W):
            if (i+W <= m) & (j+W <= n):
                mean_thres = np.mean(I[i:i+W,j:j+W]);
                mean_thres = 0.98*mean_thres;
                Binary[i:i+W,j:j+W] = I[i:i+W,j:j+W] > mean_thres;

    return Binary

def post_processing(img_binary, m, opening_mask):
# post-processing

    # image segmentation
    
    W = 16
    for i in np.arange(1, m, 2):
        for j in np.arange(1, m, 2):
            if (i+2 <= m) & (j+2 <= m):
                if np.mean(opening_mask[i:i+W, j:j+W]) == 0:
                    img_binary[i:i+W, j:j+W] = 0

    # de-white noise by opening morphology

    kernel = np.ones((2,2), np.uint8)
    enhancedImage = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)
    enhancedImage = enhancedImage * 255

    return enhancedImage


# ----- image thinning -----

def VThin(image,array):

    h, w = image.shape[:2]
    NEXT = 1
    for i in range(h):
        for j in range(w):
            if NEXT == 0:
                NEXT = 1
            else:
                M = image[i,j-1]+image[i,j]+image[i,j+1] if 0<j<w-1 else 1
                if image[i,j] == 0  and M != 0:                  
                    a = [0]*9
                    for k in range(3):
                        for l in range(3):
                            if -1<(i-1+k)<h and -1<(j-1+l)<w and image[i-1+k,j-1+l]==255:
                                a[k*3+l] = 1
                    sum = a[0]*1+a[1]*2+a[2]*4+a[3]*8+a[5]*16+a[6]*32+a[7]*64+a[8]*128
                    image[i,j] = array[sum]*255
                    if array[sum] == 1:
                        NEXT = 0
    return image
    
def HThin(image,array):

    h, w = image.shape[:2]
    NEXT = 1
    for j in range(w):
        for i in range(h):
            if NEXT == 0:
                NEXT = 1
            else:
                M = image[i-1,j]+image[i,j]+image[i+1,j] if 0<i<h-1 else 1   
                if image[i,j] == 0 and M != 0:                  
                    a = [0]*9
                    for k in range(3):
                        for l in range(3):
                            if -1<(i-1+k)<h and -1<(j-1+l)<w and image[i-1+k,j-1+l]==255:
                                a[k*3+l] = 1
                    sum = a[0]*1+a[1]*2+a[2]*4+a[3]*8+a[5]*16+a[6]*32+a[7]*64+a[8]*128
                    image[i,j] = array[sum]*255
                    if array[sum] == 1:
                        NEXT = 0
    return image
    
def thin(image, array, num=10):
# thin the ridge to be one pixel width

    height, width = image.shape[:2]
    size = (width, height)
    img_thin = cv.CreateImage(size, 8, 1)
    img_thin = image[:]
    for i in range(num):
        VThin(img_thin, array)
        HThin(img_thin, array)
    
    return img_thin

def Two(image):
    
    height, width = image.shape[:2]
    size = (width, height)
    iTwo = cv.CreateImage(size, 8, 1)
    for i in range(height):
        for j in range(width):
            iTwo[i,j] = 255 if image[i,j] < 100 else 0

    return iTwo


def thin_processing(enhanced_image):


    array = [0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,
         0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,
         1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,
         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
         1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,1,
         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
         0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,
         0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,
         1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,
         1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,0,
         1,1,0,0,1,1,1,0,1,1,0,0,1,0,0,0]


    cv2.imwrite("enhanced.bmp", enhanced_image)
    image = cv.LoadImage("enhanced.bmp",0)
    iTwo = Two(image)
    iThin = thin(iTwo, array)
    invertThin = Two(iThin)
    #cv.WaitKey(0)

    return invertThin


# ---- isolation points removal -----

def removedot(invertThin):
# remove dots
    
    temp0 = np.array(invertThin[:])
    temp0 = np.array(temp0)
    temp1 = temp0/255
    temp2 = np.array(temp1)
    temp3 = np.array(temp2)
    
    enhanced_img = np.array(temp0)
    filter0 = np.zeros((10,10))
    W,H = temp0.shape[:2]
    filtersize = 6
    
    for i in range(W - filtersize):
        for j in range(H - filtersize):
            filter0 = temp1[i:i + filtersize,j:j + filtersize]

            flag = 0
            if sum(filter0[:,0]) == 0:
                flag +=1
            if sum(filter0[:,filtersize - 1]) == 0:
                flag +=1
            if sum(filter0[0,:]) == 0:
                flag +=1
            if sum(filter0[filtersize - 1,:]) == 0:
                flag +=1
            if flag > 3:
                temp2[i:i + filtersize, j:j + filtersize] = np.zeros((filtersize, filtersize))

    return temp2


def cross_number(enhanced_img, m, n):
# minutiae extraction using crossing number method

    r=0
    g=0
    row_start = 3
    col_start = 3
    mep = np.zeros((m,2))  # array for indices of minutiae points (end point)
    mbp = np.zeros((m,2))  # bifurcation point

    for i in range(row_start, m):
        for j in range(col_start, n):
            if enhanced_img[i,j] == 1:
                cn = (1/2)*(abs(enhanced_img[i,j+1] - enhanced_img[i-1,j+1]) + abs(enhanced_img[i-1,j+1] - enhanced_img[i-1,j]) + abs(enhanced_img[i-1,j] - enhanced_img[i-1,j-1]) + abs(enhanced_img[i-1,j-1] - enhanced_img[i,j-1])+ abs(enhanced_img[i,j-1] - enhanced_img[i+1,j-1]) + abs(enhanced_img[i+1,j-1] - enhanced_img[i+1,j])+ abs(enhanced_img[i+1,j] - enhanced_img[i+1,j+1]) + abs(enhanced_img[i+1,j+1] - enhanced_img[i,j+1]))
                if cn == 1:
                    r = r+1
                    mep[r,:] = [i,j]
                elif cn == 3:
                    g = g+1
                    mbp[g,:] = [i,j]

    return mep, mbp


def marking_init(enhanced_img, mep, mbp):
# mark initially extracted minutiae points

    img_thin = np.array(enhanced_img[:])  # convert image to array for marking points

    fig = plt.figure(figsize=(10,8),dpi=30000)

    num1 = len(mep)
    num2 = len(mbp)

    figure, imshow(img_thin, cmap = cm.Greys_r)
    title('mark extracted points')
    plt.hold(True)

    for i in range(num1):
        xy = mep[i,:]
        u = xy[0]
        v = xy[1]
        if (u != 0.0) & (v != 0.0):
            plt.plot(v, u, 'r.', markersize = 7)

    plt.hold(True)

    for i in range(num2):
        xy = mbp[i,:]
        u = xy[0]
        v = xy[1]
        if (u != 0.0) & (v != 0.0):
            plt.plot(v, u, 'c+', markersize = 7)

    plt.show()
    cv2.imwrite("initial_extraction.png", img_thin)


# ----- remove false minutiae points -----

def padwithtens(vector, pad_width, iaxis, kwargs):

    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    
    return vector


def remove_border(m, n, opening_mask, mep, mbp):
# remove border points

    newmep = np.zeros((m,2))
    newmbp = np.zeros((m,2))
    z = 25
    zpad_mask = np.lib.pad(opening_mask, z, padwithtens)
    num1 = len(mep)
    num2 = len(mbp)
    d = 48

    for i in range(num1):
        xy = mep[i,:]
        u = xy[0]
        v = xy[1]
        pt = sum(zpad_mask[(u+z-d/2):(u+z-d/2+d), (v+z-d/2):(v+z-d/2+d)])
    
        if pt == d**2:
            newmep[i,:] = mep[i,:]

    for i in range(num2):
        xy = mbp[i,:]
        u = xy[0]
        v = xy[1]
        pt = sum(zpad_mask[u+z-d/2:u+z-d/2+d, v+z-d/2:v+z-d/2+d])

        if pt == d**2:
            newmbp[i,:] = mbp[i,:]

    return newmep, newmbp


def remove_inner(newmep, newmbp):
# remove inner neighboring points

    num1 = len(newmep)
    num2 = len(newmbp)
    false_tm_index = []
    false_bi_index = []

    thres_distance = 10
    for i in range(num1):
        if newmep[i,:][0] != 0:
            for j in range(num1):
                if newmep[j,:][0] != 0:
                    d = sqrt ((newmep[i,:][0] - newmep[j,:][0])**2 + (newmep[i,:][1] - newmep[j,:][1])**2)
                    if (d <= thres_distance) & (d != 0):
                        false_tm_index.append(j)

    for i in range(num2):
        if newmbp[i,:][0] != 0:
            for j in range(num2):
                if newmbp[j,:][0] != 0:
                    d = sqrt ((newmbp[i,:][0] - newmbp[j,:][0])**2 + (newmbp[i,:][1] - newmbp[j,:][1])**2)
                    if (d <= thres_distance) & (d != 0):
                        false_bi_index.append(j)

    for i in false_tm_index:
        newmep[false_tm_index,:] = [0,0]

    for i in false_bi_index:
        newmbp[false_bi_index,:] = [0,0]

    return newmep, newmbp


# ----- write the final finger print with marked minutiae points -----

def removezero(old_list):
# remove zero pairs from array

    newList = []
    for x in old_list:
        if (x[0] != 0.0) & (x[1] !=0.0):
            newList.append(x)
                
    cArray = np.array(newList[:])            
    return cArray 


def write_final(newmep, newmbp, enhanced_img):

    eminutiae_array = removezero(newmep)
    bminutiae_array = removezero(newmbp)

    num1 = len(eminutiae_array)
    num2 = len(bminutiae_array)

    img_thin = np.array(enhanced_img[:])
    fig = plt.figure(figsize=(15,12),dpi=30000)
    figure, imshow(img_thin, cmap = cm.Greys_r)
    title('minutiae marking')
    plt.hold(True)

    for i in range(num1):
        xy = eminutiae_array[i,:]
        u = xy[0]
        v = xy[1]
        plt.plot(v, u, 'r.', markersize = 10)

    plt.hold(True)

    for i in range(num2):
        xy = bminutiae_array[i,:]
        u = xy[0]
        v = xy[1]
        plt.plot(v, u, 'c+', markersize = 15)

    plt.show()
    cv2.imwrite("final_minutiae_image.png", img_thin)


def main():
    
    # setup argument parser
    parser = argparse.ArgumentParser(description='parsing fingerprint minutiae points recognition')
    parser.add_argument('image_name', help='require a test image')
    args = parser.parse_args()

    # call components
    image_name = args.image_name
    row, col, fingerprint, fpcopy = read_image(image_name)
    finger_print, opening_mask = segment(row, col, fingerprint, fpcopy)
    
    normalized_image = normal(finger_print, row, col)
    fft_image = enhance(row, col, normalized_image)
    binarized_image = adp_binary(fft_image, row, col)
    enhanced_image = post_processing(binarized_image, row, opening_mask)
    thinned_image = thin_processing(enhanced_image)
    
    dedot_image = removedot(thinned_image)
    end_point, bifur_point = cross_number(dedot_image, row, col)
    marking_init(de_dot, end_point, bifur_point)
    new_endpoint, new_bifurpoint = remove_border(row, col, opening_mask, end_point, bifur_point)
    endPoint, bifurPoint = remove_inner(new_endpoint, new_bifurpoint)
    write_final(endPoint, bifurPoint, dedot_image)


# -----------------------------------------------------------------------------
# run script

if __name__ == '__main__':
    main()



