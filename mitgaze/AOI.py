#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 19:42:28 2020.

@author: vbokharaie

This module includes all functions related to AOI handling. 
It is assumed that AOIs are masks in an RGB standard in which the AOI is 
specified with one of the three main or secondary colours, r, g, b, c, m, and y. 
"""


# %% find_AOI
def find_AOI(image_name, colour='red'):
    """
    Find the AOI based on colour-codes.

    If AOI is painted over a photo with one of the main 6 colours,
    return the set of all (x,y) corrdinates of the AOI pixels.

    Parameters
    ----------
    image_name : Numpy array
        Image containing AOI colour-coded in r,g,b,c,y or m.
    colour : str, optional
        Can be one of r, g, b, c, m, y letters or fulle colour names. The default is 'red'.

    Returns
    -------
    mask : set of (int, int)
        Set of all pixels in the mask.

    """
    from PIL import Image

    r=0
    g=1
    b=2

    if colour == 'red' or colour == 'r':
        (r_min, r_max, g_min, g_max, b_min, b_max) = (250, 255, 0, 50, 0, 50)
    elif colour == 'green' or colour == 'g':
        (r_min, r_max, g_min, g_max, b_min, b_max) = (0, 50, 250, 255, 0, 50)
    elif colour == 'blue' or colour == 'b':
        (r_min, r_max, g_min, g_max, b_min, b_max) = (0, 50, 0, 50, 250, 255)
    elif colour == 'cyan' or colour == 'c':
        (r_min, r_max, g_min, g_max, b_min, b_max) = (0, 50, 250, 255, 250, 255)
    elif colour == 'magenta' or colour == 'm':
        (r_min, r_max, g_min, g_max, b_min, b_max) = (250, 255, 0, 50, 250, 255)
    elif colour == 'yellow' or colour == 'y':
        (r_min, r_max, g_min, g_max, b_min, b_max) = (250, 255, 250, 255, 0, 50)

    mask = set()

    img = Image.open(image_name)
    pix = img.load()
    if type(pix[0,0]) is int:
        rgbimg = Image.new("RGB", img.size)
        rgbimg.paste(img)
        pix = rgbimg.load()
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            r, g, b = pix[x,y]
            if r >= r_min and r <= r_max and b >= b_min and b <= b_max and g >= g_min and g <= g_max:
                mask.add((x,y))

    return mask

# %% AOI_in_out_indices
def AOI_in_out_indices(x, y, mask, smooth_factor=15):
    """
    Find indices in which gaze entered and exits an AOI.

    Parameters
    ----------
    x : 1D numpy array or list of int
        x coordintaes of gaze.
    y : 1D numpy array or list of int
        y coordintaes of gaze.
    mask : set of (int, int)
        set of (x,y) indices of AOI pixels.
    smooth_factor : int, optional
        Any in-out or out-in period less than this value is ignored. The default is 15.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    import numpy as np

    if len(mask) == 0:
        return []
    xy = list(zip(x, y))

    list_ind = [idx for idx, el in enumerate(xy) if el in mask]
    if list_ind == []:
        return []
    list_ind_diff = np.diff(list_ind)
    list_ind_diff[list_ind_diff <= smooth_factor] = 1

    list_ind_diff_jump = [idx for idx, el in enumerate(list_ind_diff) if el>1]

    list_diff_enter = [list_ind[x+1] for x in list_ind_diff_jump]
    list_diff_enter.insert(0, list_ind[0])

    list_diff_exit = [list_ind[x] for x in list_ind_diff_jump]
    list_diff_exit.insert(len(list_diff_exit), list_ind[-1])

    assert len(list_diff_enter) == len(list_diff_exit), 'something wrong here!'

    list_unsmoothed = list(zip(list_diff_enter, list_diff_exit))
    list_indices_in_out = [x for x in list_unsmoothed if x[1]-x[0] > smooth_factor]

    return list_indices_in_out


