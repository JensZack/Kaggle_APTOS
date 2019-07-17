# testing the APTOS tools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from APTOS_tools import edge_detection_3, gauss_kernel, apply_filter, generate_circle, crop_image, k_means_cluster, dbscan_image_cluster


def main():

    train_filepath = 'data/train_images/'
    test_filepath = 'data/test_images/'

    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')

    train_X = df_train['id_code'].values
    train_y = df_train['diagnosis'].values

    # image list is the list of image file paths in order
    image_list = list(map((lambda fi: train_filepath+fi), os.listdir(train_filepath)))

    print('train data size: ', train_y.shape)
    print('train file name: ', os.listdir(train_filepath)[0])

    toFileLoc = lambda fi: train_filepath + fi + '.png'

    df_train['id_code'] = df_train['id_code'].apply(toFileLoc)
    df_train = df_train.rename(index=str, columns={'id_code': 'filepath'})

    test_img = Image.open(df_train['filepath'].values[1763])

    # plt.imshow(test_img)
    # plt.show()

    test_img = np.array(test_img)

    kernel = gauss_kernel(sigma=1)
    test_img = apply_filter(test_img, kernel)
    test_img = crop_image(test_img)

    ###########################################################################
    '''
    edge_img = edge_detection_3(test_img)

    f, ax = plt.subplots(1, 5, figsize=(9, 9))
    _ = ax[0].imshow(Image.fromarray(edge_img), cmap='gray')
    _ = ax[1].imshow(test_img)
    _ = ax[2].imshow(test_img[..., 0], cmap='gray')
    _ = ax[3].imshow(test_img[..., 1], cmap='gray')
    _ = ax[4].imshow(test_img[..., 2], cmap='gray')
    plt.show()
    '''
    ###########################################################################
    test_center = (int(np.floor(test_img.shape[0]/2)), int(np.floor(test_img.shape[1]/2)))
    test_radius = min(test_center)-1

    test_img_xy = np.zeros((test_img.shape[0], test_img.shape[1], 5))
    test_img_xy[..., 0:3] = test_img

    lam = 10

    for i in range(test_img.shape[0]):
        for j in range(test_img.shape[1]):
            test_img_xy[i, j, 3] = i * lam
            test_img_xy[i, j, 4] = j * lam

    test_img_xy = test_img_xy.reshape((-1, 5))
    dist_r = lambda x, y: np.sqrt(np.square(x - test_center[0]) + np.square(y - test_center[1]))
    dist_r = np.vectorize(dist_r)
    test_img_xy = test_img_xy[dist_r(test_img_xy[..., 3], test_img_xy[..., 4]) < test_radius]

    scaler = StandardScaler()
    scaler.fit(test_img_xy)
    x = scaler.transform(test_img_xy)
    print(x.shape)

    subset = np.random.choice(np.arange(test_img_xy.shape[0]), size=10 ** 3)

    labels = list()
    test_tsne = TSNE().fit_transform(x[subset])

    for i in np.linspace(1, 11, 6):
        # i = int(i)
        labels.append(dbscan_image_cluster(x[subset], eps=i/10))

    f, axs = plt.subplots(3, 2, figsize=(9, 9))

    for idx in range(len(labels)):
        axs[idx % 3, int(np.floor(idx / 3))].scatter(test_tsne[:, 0], test_tsne[:, 1], c=labels[idx])
        axs[idx % 3, int(np.floor(idx / 3))].set_title('eps: ' + str(np.linspace(1, 11, 6)[idx]))

    plt.show()




    '''
    subset = np.random.choice(np.arange(test_img_xy.shape[0]), size=10**3)
    test_img_xy_sub = test_img_xy[subset]

    test_tsne = TSNE().fit_transform(test_img_xy_sub)
    plt.scatter(test_tsne[:, 0], test_tsne[:, 1])
    plt.show()
    '''

    '''
    # generating a circle with radius 250 and center 300
    plot = np.zeros((600, 600))

    circle_idxs = generate_circle((300, 300), 250)

    for idx in circle_idxs:
        plot[idx] = 255

    plot.astype(np.uint8)
    plt.imshow(plot)
    plt.show()
    '''


if __name__ == "__main__":
    main()
