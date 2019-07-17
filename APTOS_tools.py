# a python file containing tools for image processing in the APTOS kaggle comp
# Zack Jensen
# 7/11/19

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2
from copy import deepcopy
from sklearn.cluster import KMeans, DBSCAN

SEED = 22

np.random.seed(SEED)


def crop_image(np_img, tol=None):
    if tol is None:
        tol = (np.max(np_img) / 255) * 10

    # crop left columns
    while np.mean(np.sum(np_img[:, 0, :], axis=1)) < tol:
        np_img = np.delete(np_img, 0, 1)  # deleting the 0th column (axis=1)

    # crop right columns
    while np.mean(np.sum(np_img[:, -1, :], axis=1)) < tol:
        np_img = np.delete(np_img, -1, 1)

    # crop top rows
    while np.mean(np.sum(np_img[0, :, :], axis=1)) < tol:
        np_img = np.delete(np_img, 0, 0)

    # crop bottom rows
    while np.mean(np.sum(np_img[-1, :, :], axis=1)) < tol:
        np_img = np.delete(np_img, -1, 0)

    return np_img


def gauss_kernel(sigma=1, dim=5):
    # gaussian filter to remove noise
    g_kernel = np.zeros((dim, dim))
    center = ((dim - 1) / 2, (dim - 1) / 2)
    gauss_dist = lambda d: np.exp(-(np.power(d, 2)) / sigma)

    for i in range(dim):
        for j in range(dim):
            dist = np.sqrt(np.power(i - center[0], 2) + np.power(j - center[0], 2))
            g_kernel[i, j] = gauss_dist(dist)

    g_kernel = g_kernel / np.sum(g_kernel)
    return g_kernel


def apply_filter(np_img, kernel):
    assert kernel.shape[0] == kernel.shape[1], "kernel must be 2d square matrix"
    img_out = cv2.filter2D(np_img, -1, kernel)
    return img_out


def edge_detection_1(np_img, c=2, window_dim=15):
    # input a window from an image, and cluster into either 1 or 2 groups
    # If 2 groups are found, return the line between groups
    m, n, n_channels = np_img.shape
    assert n_channels == 3, "image must be RGB"

    n_rows = int(np.floor(m/window_dim))
    n_cols = int(np.floor(n/window_dim))


def edge_detection_2(np_img):
    # flatten the image in both axis, and mark points before a change in 1d
    # where both 1d measurements agree, mark as an edge point
    m, n, n_channels = np_img.shape
    assert n_channels == 3, "image must be RGB"

    row_mat = np_img.reshape((-1, 3), order='C')
    col_mat = np_img.reshape((-1, 3), order='F')

    # baseline variation of the whole image
    # setting alpha=.01 for false positives

    # calculating 3d variance of the pixel data
    img_copy = deepcopy(np_img)
    rgb_avg = np.mean(img_copy, axis=0)
    rgb_dist = np.square(img_copy-rgb_avg)


def edge_detection_3(np_img, tol=10):
    # edge detection using gradients of the image
    # make kernels corresponding to df/dx, df/dy
    m, n, n_channels = np_img.shape

    dxk = np.zeros((3, 3))
    dxk[1, 0] = 1
    dxk[1, 2] = -1

    dyk = np.zeros((3, 3))
    dyk[0, 1] = 1
    dyk[2, 1] = -1

    r = np_img[..., 0]
    g = np_img[..., 1]
    b = np_img[..., 2]

    r_dx_img = apply_filter(r, dxk)
    g_dx_img = apply_filter(g, dxk)
    b_dx_img = apply_filter(b, dxk)

    r_dy_img = apply_filter(r, dyk)
    g_dy_img = apply_filter(g, dyk)
    b_dy_img = apply_filter(b, dyk)

    grad_mat = np.add(r_dx_img, np.add(g_dx_img, np.add(b_dx_img, np.add(r_dy_img, np.add(g_dy_img, b_dy_img)))))
    # grad_mat = np.add(g_dx_img, g_dy_img)

    rounding_pixel = np.vectorize(lambda x, tol_: 255 if x > tol_ else 0)
    grad_vec = grad_mat.flatten()
    rounded_vec = rounding_pixel(grad_vec, tol)
    edge_mat = rounded_vec.reshape((m, n))

    edge_mat = edge_mat.astype(np.uint8)
    return edge_mat


def generate_circle(center, radius):
    # takes a center point and a radius and returns a list of
    # (x, y) points that make the given circle

    # directions: going clockwise (dr, dl, ul, ur)
    def next_pixel(position, center_, radius_, direction):
        next_pixels = []
        x_current, y_current = position

        if direction == 'dr':
            next_pixels.append((y_current, x_current + 1))
            next_pixels.append((y_current + 1, x_current + 1))
            next_pixels.append((y_current + 1, x_current))

        if direction == 'dl':
            next_pixels.append((y_current, x_current - 1))
            next_pixels.append((y_current + 1, x_current - 1))
            next_pixels.append((y_current + 1, x_current))

        if direction == 'ul':
            next_pixels.append((y_current, x_current - 1))
            next_pixels.append((y_current - 1, x_current - 1))
            next_pixels.append((y_current - 1, x_current))

        if direction == 'ur':
            next_pixels.append((y_current, x_current + 1))
            next_pixels.append((y_current - 1, x_current + 1))
            next_pixels.append((y_current - 1, x_current))

        dist_list = []
        for pixel in next_pixels:
            p_radius = np.sqrt(np.square(pixel[0] - center_[0]) + np.square(pixel[1] - center_[1]))
            dist_list.append(np.abs(p_radius - radius_))

        min_radius_dist = min(dist_list)
        next_pixel_idx = dist_list.index(min_radius_dist)

        return next_pixels[next_pixel_idx]

    x_center, y_center = center

    # points initially defining circle: up, down, left, right
    u = (y_center + radius, x_center)
    d = (y_center - radius, x_center)
    r = (y_center, x_center + radius)
    l = (y_center, x_center - radius)

    circle = list()

    # u -> r
    circle.append(u)
    next_ = next_pixel(u, (x_center, y_center), radius, 'dr')
    while next_ != r:
        circle.append(next_)
        next_ = next_pixel(next_, (x_center, y_center), radius, 'dr')

    # r -> d
    circle.append(r)
    next_ = next_pixel(r, (x_center, y_center), radius, 'dl')
    while next_ != d:
        circle.append(next_)
        next_ = next_pixel(next_, (x_center, y_center), radius, 'dl')

    # d -> l
    circle.append(d)
    next_ = next_pixel(d, (x_center, y_center), radius, 'ul')
    while next_ != l:
        circle.append(next_)
        next_ = next_pixel(next_, (x_center, y_center), radius, 'ul')

    # l -> u
    circle.append(l)
    next_ = next_pixel(l, (x_center, y_center), radius, 'ur')
    while next_ != u:
        circle.append(next_)
        next_ = next_pixel(next_, (x_center, y_center), radius, 'ur')

    return circle


def identify_circle(np_img):
    # takes in the image of edge points and returns the radius with the radius of the eye
    # the edge image should be in the format np_img[i,j] == 0 or 255 for all i,j

    '''
    bounds on the radius: max_x/2, max_y/2 because the image is cropped
    '''

    pass


def dbscan_image_cluster(np_img_xy, eps=3, minsamples=20):
    dbscan = DBSCAN(eps=eps, min_samples=minsamples).fit(np_img_xy)
    labels = dbscan.labels_

    return labels


def k_means_cluster(np_img_xy, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters).fit(np_img_xy)
    labels = kmeans.labels_

    return labels


def segment_image(np_img_xy, labels=None):
    if not labels:
        plt.imshow(np_img_xy[..., 0:3])
        plt.show()
        return

    
