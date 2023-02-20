#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 14:12:36 2019.

@author: vbokharaie
"""

def edge_detect(img, alg_type='roberts'):
    """
    Detect edges of an image using various algorithms from skimage.filters.

    Parameters
    ----------
    img : 2-D array
        Image to process.
    alg_type : str, optional
        EDge-detection algortihm type. The default is 'roberts'.

    Returns
    -------
    edge : 2-D array
        The Cross edge map..

    """
    from skimage.filters import (
    roberts,
    sobel,
    sobel_h,
    sobel_v,
    scharr,
    scharr_h,
    scharr_v,
    prewitt,
    prewitt_v,
    prewitt_h,
    farid,
    farid_v,
    farid_h,
    )

    if alg_type == 'roberts':
        edge = roberts(img)
    elif alg_type == 'sobel':
        edge = sobel(img)
    elif alg_type == 'scharr':
        edge = scharr(img)
    elif alg_type == 'prewitt':
        edge = prewitt(img)
    elif alg_type == 'farid':
        edge = farid_v(img)

    else:
        return None


    return edge


def plot_edges(image_filename, dir_save_filters):
    """
    Plot edges of an image and save the result as an image.

    Parameters
    ----------
    image_filename : TYPE
        DESCRIPTION.
    dir_save_filters : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    from skimage import io
    import matplotlib.pyplot as plt
    from pathlib import Path

    from skimage.color import rgb2gray
    from mitgaze.filters import edge_detect

    dir_save_filters = Path(dir_save_filters)
    dir_save_filters.mkdir(parents=True, exist_ok=True)
    media = Path(image_filename)

    img = io.imread(media) # READS BGR as in (X,Y,3)
    if len(img.shape)==3:
        img = img[..., ::-1]

    img_grayscale = rgb2gray(img)
    img_grayscale = rgb2gray(img)

    (fig_W, fig_H) = (19, 12)
    fig, ax = plt.subplots(3, 2, figsize = (fig_W*2, fig_H*3))
    ax[0,0].imshow(img_grayscale, cmap=plt.cm.gray)
    edge = edge_detect(img_grayscale)
    ax[0,1].imshow(edge, cmap=plt.cm.gray)
    edge = edge_detect(img_grayscale, alg_type='sobel')
    ax[1,0].imshow(edge, cmap=plt.cm.gray)
    edge = edge_detect(img_grayscale, alg_type='scharr')
    ax[1,1].imshow(edge, cmap=plt.cm.gray)
    edge = edge_detect(img_grayscale, alg_type='prewitt')
    ax[2,0].imshow(edge, cmap=plt.cm.gray)
    edge = edge_detect(img_grayscale, alg_type='farid')
    ax[2,1].imshow(edge, cmap=plt.cm.gray)

    file_save = Path(dir_save_filters, media.stem + '_edges' + '.png')
    fig.savefig(file_save, format='png')
    plt.close('all')


