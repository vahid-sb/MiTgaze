#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 09:58:31 2020

@author: wxiao
"""

#%% functions that are used in multiple senarios in analysis
def img_to_array(img):
    """
    convert an image object to np.ndarray

    Parameters
    ----------
    img : Image
        Input PIL image object.

    Returns
    -------
    img_out : np.ndarray
        output Image array.

    """
    import numpy as np
    
    SCREEN_W, SCREEN_H = img.size
    # img.getdata() scans the image horizontally from left to right starting 
    # at the top-left corner. so we need to reshape the data.
    img_out = np.asarray(list(img.getdata())).reshape(SCREEN_H, SCREEN_W)
    
    return img_out


def grid_mean_img(img, grid_size=(24,48)):
    """
    Calculate mean pixel value within grids

    Parameters
    ----------
    img : np.ndarray
        Image stored in an array.
    grid_size : (int,int)
        (#grid along height, #grid along width). The default is (24,48).

    Raises
    ------
    ValueError
        The input image needs to be of np.ndarray.

    Returns
    -------
    grid_img : Array of float64
        Each value in the array represents the mean pixel value within a grid.

    """
    import numpy as np
    # check img TYPE
    if not isinstance(img, np.ndarray):  
        raise ValueError("The input image needs to be of numpy.ndarray object")
    
    # check if input grid size is integer
    nH, nW = grid_size # input numGrid for each axis        
    if not (isinstance(nW, int) & isinstance(nH, int)): 
        print("Grid size need to be of type <class 'int'>.")
        import sys
        sys.exit(0) 
        
    SCREEN_H, SCREEN_W = img.shape
    
    # Calculated mean of pixel values in each grid
    # Note:nH -> array rows, nW -> array columns
    if SCREEN_H % nH == 0 & SCREEN_W % nW == 0:
        grid_img = img.reshape(nH, SCREEN_H//nH, nW, SCREEN_W//nW).mean(axis=(1,3))
    else:
        print("Divisibility check failed, grid size is not compatible with pixel number ")
        import sys
        sys.exit(0)

    return grid_img


#%% light intensity
def grid_img_light_intensity(file_path, grid_size):
    """
    input a path for an image
    convert to gray scale image
    use grid_mean (img, grid_size) to calculate mean pixel value within each grid

    Parameters
    ----------
    file_path : PosixPath
        Path to the source image.
    grid_size : (int,int)
        (#grid along height, #grid along width). The default is (24,48).

    Returns
    -------
    grid_img : Array of float64
        Each value in the array represents the mean pixel value within a grid.

    """
    from PIL import Image, ImageOps
    
    # read the image 
    img = Image.open(file_path,'r')
    
    # convert to gray scale image and calculate mean pixel value within grid
    img_gray = ImageOps.grayscale(img)
    img_gray = img_to_array(img_gray)
    grid_img = grid_mean_img(img_gray, grid_size)
    
    return grid_img


def light_intensity_analysis(file_path, dir_save_main, grid_size):
    """
    input a path for an image
    convert to gray scale image
    use grid_mean (img, grid_size) to calculate mean pixel value within each grid
    store the result in .npy and .mat format

    Parameters
    ----------
    file_path : PosixPath
        file path of the input image.
    dir_save_main : PosixPath
        directory for saving the result.
    grid_size : (int,int)
        (#grid along height, #grid along width). The default is (24,48).
     
    Returns
    -------
    None.

    """
    from pathlib import Path
    import numpy as np
    from scipy.io import savemat 
    
    # get filename from path
    media = file_path.stem 
    
    grid_img = grid_img_light_intensity(file_path, grid_size)
    
    # save together
    dir_save_all = Path(dir_save_main)
    dir_save_all.mkdir(parents = True, exist_ok = True)
    
    file_name = media + '_light_intensity'
    file_name_npy = Path(dir_save_all, file_name + '.npy')
    np.save(file_name_npy,grid_img)
    file_name_mat = Path(dir_save_all, file_name + '.mat')
    savemat(file_name_mat, {'grid_img': grid_img})


def light_intensity_to_df(file_source_folder, grid_size):
    """
    fetch light_intensity information and store in a data frame

    Parameters
    ----------
    file_source_folder : String
        Path of media source.
    grid_size : (int,int)
        (#grid along height, #grid along width). The default is (24,48).
         
    Returns
    -------
    df_light_intensity : Pandas DataFrame
        columns = ['file_path','media','img_light_intensity'].

    """
    from pathlib import Path
    import pandas as pd
    
    # get files from source
    all_files = list(Path(file_source_folder).glob('*.jpg'))
    
    row_list=[]
    
    for file_path in all_files:
        
        media = file_path.stem
        result_dict = {'file_path': file_path, 'media': media}
        
        grid_img = grid_img_light_intensity(file_path, grid_size)
        result_dict['img_light_intensity'] = grid_img
    
        row_list.append(result_dict)

    df_light_intensity = pd.DataFrame(row_list, columns = ['file_path','media',
                                                           'img_light_intensity'])
    
    return df_light_intensity


#%% color
def get_rgb_hsv(img, channel):
    """
    get R, G, B, H, S, V information from image.

    Parameters
    ----------
    img : Image
        The input color image.

    Returns
    -------
    output: PIL Image
        Single Image object depending on input channel.
        or a list of Image objects from all channels ['r', 'g', 'b', 'h', 's', 'v']

    """
    if channel not in ('r', 'g', 'b', 'h', 's', 'v', 'all'):
        print("Input channel needs to be one of ['r', 'g', 'b', 'h', 's', 'v', 'all']")
        import sys
        sys.exit(0)
        
    r,g,b = img.split()
    img_hsv = img.convert("HSV")
    h,s,v = img_hsv.split()
    
    if channel == 'r':
        output = r
    elif channel == 'g':
        output = g
    elif channel == 'b':
        output = b
    elif channel == 'h':
        output = h
    elif channel == 's':
        output = s
    elif channel == 'v':
        output = v
    elif channel == 'all':
        output = [r, g, b, h, s, v]
        
    return output


def grid_img_color(file_path, grid_size, channel):
    """
    input a path for an image
    use get_rgb_hsv(img) to get R,G,B,H,S,V information about the image
    use grid_mean (img, grid_size) to calculate mean pixel value within each grid

    Parameters
    ----------
    file_path : PosixPath
        Path to the source image.
    channel : string
        Input color needs to be one of ['r', 'g', 'b', 'h', 's', 'v']
    grid_size : (int,int)
        (#grid along height, #grid along width). The default is (24,48).

    Returns
    -------
    grid_img : Array of float64, or list of Array 
        Each value in the array represents the mean pixel value within a grid.
        
    """
    from PIL import Image
    
    if channel not in ('r', 'g', 'b', 'h', 's', 'v', 'all'):
        print("Input channel needs to be one of ['r', 'g', 'b', 'h', 's', 'v']")
        import sys
        sys.exit(0)
    
    # read the image 
    img = Image.open(file_path,'r')
    img_channel = get_rgb_hsv(img, channel)
    
    grid_img = []
    if channel == 'all':
        for img_obj in img_channel:
            img_obj = img_to_array(img_obj)
            grid_img.append(grid_mean_img(img_obj))
    else:
        img_channel = img_to_array(img_channel)
        grid_img = grid_mean_img(img_channel, grid_size)
    
    return grid_img


def color_analysis(file_path, dir_save_main, grid_size):
    """
    input a path for an image
    use get_rgb_hsv(img) to get R,G,B,H,S,V information about the image
    use grid_mean (img, grid_size) to calculate mean pixel value within each grid
    store the result in .npy and .mat format

    Parameters
    ----------
    file_path : PosixPath
        file path of the input image.
    dir_save_main : PosixPath
        directory for saving the result.
    grid_size : (int,int)
        (#grid along height, #grid along width). The default is (24,48).

    Returns
    -------
    None.

    """
    from pathlib import Path
    import numpy as np
    from PIL import Image
    from scipy.io import savemat     
    
    media = file_path.stem
    img = Image.open(file_path,'r')
    
    if img.mode == "RGB":
        grid_img_all = grid_img_color(file_path, grid_size, channel = 'all') 
        channel_name = ['R','G','B','H','S','V']
        for idx, grid_img in enumerate(grid_img_all):
            
            # save separately to subfolders according to channel name
            dir_save_subfolder = Path(dir_save_main, channel_name[idx]) 
            dir_save_subfolder.mkdir(parents = True, exist_ok = True)
            
            file_name = media + '_color_' + channel_name[idx]
            file_name_npy = Path(dir_save_subfolder, file_name + '.npy')
            np.save(file_name_npy,grid_img)
            file_name_mat = Path(dir_save_subfolder, file_name + '.mat')
            savemat(file_name_mat, {'grid_img': grid_img})
            
            # save together
            dir_save_all = Path(dir_save_main, 'color_all')
            dir_save_all.mkdir(parents = True, exist_ok = True)
            
            file_name_npy = Path(dir_save_all, file_name + '.npy')
            np.save(file_name_npy,grid_img)
            file_name_mat = Path(dir_save_all, file_name + '.mat')
            savemat(file_name_mat, {'grid_img': grid_img})

 
def color_to_df(file_source_folder, grid_size):
    """
    fetch color channel information and store in a data frame

    Parameters
    ----------
    file_source_folder : String
        Path of media source.
    grid_size : (int,int)
        (#grid along height, #grid along width). The default is (24,48).
         
    Returns
    -------
    df_color : Pandas DataFrame
        columns = ['file_path', 'media', 'img_mode', 'img_r', 'img_g', 
                   'img_b','img_h', 'img_s', 'img_v'].

    """
    from pathlib import Path
    import pandas as pd
    from PIL import Image
    
    # get files from source
    all_files = list(Path(file_source_folder).glob('*.jpg'))
    
    row_list=[]
    
    for file_path in all_files:
        
        media = file_path.stem
        result_dict = {'file_path': file_path, 'media': media}
        
        img = Image.open(file_path,'r')
        # check if the image is color or gray
        if img.mode not in ("L","RGB"):
            raise ValueError("Unsuported image mode")
            
        result_dict['img_mode'] = img.mode
        
        # extract pixel value from single channel
        if img.mode == "RGB":
            grid_img_all = grid_img_color(file_path, grid_size, channel = 'all') 
            channel_name = ['r','g','b','h','s','v']
            for idx, grid_img in enumerate(grid_img_all):
                new_key = 'img_' + channel_name[idx]
                result_dict[new_key] = grid_img
            
        row_list.append(result_dict)
    
    df_color= pd.DataFrame(row_list, columns = ['file_path', 'media', 'img_mode', 
                                                'img_r', 'img_g', 'img_b','img_h', 
                                                'img_s', 'img_v'])
    
    return df_color


#%% contrast
def apply_edge_detection_algorithm(img, algorithm):
    '''
    Apply edge detection algorithm to the input image (gray-scaled).

    Parameters
    ----------
    img : Array of float64
        array of a gray scale image.
    algorithm : String
        one of ['roberts', 'sobel', 'scharr', 'prewitt', 'farid'].

    Raises
    ------
    ValueError
        The input image needs to be of numpy.ndarray object

    Returns
    -------
    output : Array of float64
        output of the applied edge detection algorithm.

    '''
    from skimage.filters import roberts, sobel, scharr, prewitt, farid
    import numpy as np
    
    # check img TYPE
    if not isinstance(img, np.ndarray):  
        raise ValueError("The input image needs to be of numpy.ndarray object")
    
    # check algorithm
    if algorithm not in ('roberts', 'sobel', 'scharr', 'prewitt', 'farid'):
        print("Input algorithm needs to be one of ['roberts', 'sobel', 'scharr', 'prewitt', 'farid']")
        import sys
        sys.exit(0)
        
    # apply different edge detection algorithm
    if algorithm == 'roberts':    
        output = roberts(img)
    elif algorithm == 'sobel': 
        output = sobel(img)
    elif algorithm == 'scharr':
        output = scharr(img)
    elif algorithm == 'prewitt':
        output = prewitt(img)
    elif algorithm == 'farid':
        output = farid(img)
    
    return output
    

def grid_img_contrast(file_path, grid_size, algorithm):
    '''
    input a path for an image
    convert to gray scale image
    use apply_edge_detection_algorithm to detect edges
    use grid_mean (img, grid_size) to calculate mean pixel value within each grid

    Parameters
    ----------
    file_path : PosixPath
        Path to the source image.
    grid_size : (int,int)
        (#grid along height, #grid along width). The default is (24,48).
    algorithm : String
        one of ['roberts', 'sobel', 'scharr', 'prewitt', 'farid'].

    Returns
    -------
    grid_img : Array of float64,
        Each value in the array represents the mean pixel value within a grid.

    '''
    from PIL import Image, ImageOps
    import numpy as np
    
    # read the image 
    img = Image.open(file_path,'r')
    
    SCREEN_W, SCREEN_H = img.size
    
    img_gray = ImageOps.grayscale(img)
    # get data() scans the image horizontally from left to right starting at 
    # the top-left corner. so we need to reshape the data.
    img_gray = np.asarray(list(img_gray.getdata())).reshape(SCREEN_H, SCREEN_W)     
    
    # apply edge detection algorithm
    img_edge = apply_edge_detection_algorithm(img_gray, algorithm)
    
    grid_img = grid_mean_img(img_edge)
    
    return grid_img


def contrast_analysis(file_path, dir_save_main, grid_size):
    """
    input a path for an image
    convert to gray scale image
    apply edge detection algorithm
    use grid_mean (img, grid_size) to calculate mean pixel value within each grid
    store the result in .npy and .mat format

    Parameters
    ----------
    file_path : PosixPath
        file path of the input image.
    dir_save_main : PosixPath
        directory for saving the result.
    grid_size : (int,int)
        (#grid along height, #grid along width). The default is (24,48).

    Returns
    -------
    None.

    """
    from pathlib import Path
    import numpy as np
    from scipy.io import savemat 
    
    # read the image and get the image name from its path
    media = file_path.stem
    
    algorithm_list = ['roberts', 'sobel', 'scharr', 'prewitt', 'farid']
    
    for algorithm in algorithm_list:
        grid_img = grid_img_contrast(file_path, grid_size, algorithm)
            
        # save separately to subfolders according to channel name
        dir_save_subfolder = Path(dir_save_main, 'contrast_' + algorithm) 
        dir_save_subfolder.mkdir(parents = True, exist_ok = True)
        
        file_name = media + '_contrast_' + algorithm
        file_name_npy = Path(dir_save_subfolder, file_name + '.npy')
        np.save(file_name_npy, grid_img)
        file_name_mat = Path(dir_save_subfolder, file_name + '.mat')
        savemat(file_name_mat, {'grid_img': grid_img})
        
        # save together
        dir_save_all = Path(dir_save_main, 'contrast_all')
        dir_save_all.mkdir(parents = True, exist_ok = True)
        
        file_name_npy = Path(dir_save_all, file_name + '.npy')
        np.save(file_name_npy,grid_img)
        file_name_mat = Path(dir_save_all, file_name + '.mat')
        savemat(file_name_mat, {'grid_img': grid_img})

    
def contrast_to_df(file_source_folder, grid_size):
    """
    fetch edge information and store in a data frame

    Parameters
    ----------
    file_source_folder : String
        Path of media source.
    grid_size : (int,int)
        (#grid along height, #grid along width). The default is (24,48).
         
    Returns
    -------
    df_contrast : Pandas DataFrame
        columns = ['file_path','media','edge_roberts', 'edge_sobel', 
                   'edge_scharr', 'edge_prewitt', 'edge_farid'].

    """
    from pathlib import Path
    import pandas as pd

    # get files from source
    all_files = list(Path(file_source_folder).glob('*.jpg'))
    
    row_list=[]
    
    for file_path in all_files:
        media = file_path.stem
        result_dict = {'file_path': file_path, 'media': media}
        
        algorithm_list = ['roberts', 'sobel', 'scharr', 'prewitt', 'farid']
        for algorithm in algorithm_list:
            grid_img = grid_img_contrast(file_path, grid_size, algorithm)
            new_key = 'edge_'+ algorithm
            result_dict[new_key] = grid_img
            
        row_list.append(result_dict)

    df_contrast= pd.DataFrame(row_list, columns = ['file_path','media','edge_roberts', 
                                                   'edge_sobel', 'edge_scharr', 
                                                   'edge_prewitt', 'edge_farid'])

    return df_contrast


#%%
def mask_to_index_matrix(file_path, mask):
    """
    convert the mask set to an index matrix.

    Parameters
    ----------
    file_path : PosixPath
        Path to the source image. Used to get the image size.
    mask : set
        Set of tuples indicating position of the mask color.

    Returns
    -------
    img_AOI : numpy Array
        A matrix consists of 0 and 1. 1s in the matrix correspond to tuples in the mask set.

    """
    from PIL import Image
    import numpy as np
    
    img = Image.open(file_path,'r')
    
    img_AOI = np.zeros(img.size)  # img.size returns (width, height)
    for pixel in mask:
        img_AOI[pixel]=1

    img_AOI = img_AOI.T
    
    return img_AOI


def grid_img_AOI(file_path, grid_size, color):
    """
    input a path for an image
    use find_AOI to find set of points in AOI
    use mask_to_index_matrix to convert the set of points to an index matrix
    use grid_mean (img, grid_size) to calculate mean pixel value within each grid

    Parameters
    ----------
    file_path : PosixPath
        Path to the source image.
    color : string
        Input color needs to be one of ['r', 'g', 'b', 'c', 'm', 'y']
    grid_size : (int,int)
        (#grid along height, #grid along width). The default is (24,48).

    Returns
    -------
    grid_img : numpy Array
        mean pixel value within each grid of the index matrix (aka. img_AOI)

    """
    from mitgaze.AOI import find_AOI
    
    if color not in ('r', 'g', 'b', 'c', 'm', 'y'):
        print("Input color needs to be one of ['r', 'g', 'b', 'c', 'm', 'y']")
        import sys
        sys.exit(0)
        
    grid_img = []
    AOI_mask = find_AOI(file_path, color)
    if len(AOI_mask) > 0:
        img_AOI = mask_to_index_matrix(file_path, AOI_mask)
        grid_img = grid_mean_img(img_AOI, grid_size)
    
    return grid_img
        
    
def AOI_analysis(file_path, dir_save_main, grid_size):
    """
    input a path for an image
    use find_AOI to find set of points in AOI
    use mask_to_index_matrix to convert the set of points to an index matrix
    use grid_mean (img, grid_size) to calculate mean pixel value within each grid
    store the result in .npy and .mat format

    Parameters
    ----------
    file_path : PosixPath
        file path of the input image.
    dir_save_main : PosixPath
        directory for saving the result.
    grid_size : (int,int)
        (#grid along height, #grid along width). The default is (24,48).
        
    Returns
    -------
    None.
    
    """
    from pathlib import Path
    import numpy as np
    from scipy.io import savemat 
    
    media = file_path.stem
    
    AOI_color = ['r', 'g', 'b', 'c', 'm', 'y']

    # iterate through AOI color list, grid_img=[] if there is no corresponding AOI
    for color in AOI_color:
        grid_img = grid_img_AOI(file_path, grid_size, color)
        
        if len(grid_img)>0:
            # save separately to subfolders according to media name
            dir_save_subfolder = Path(dir_save_main, media) 
            dir_save_subfolder.mkdir(parents = True, exist_ok = True)
            
            file_name = media + '_AOI_' + color
            file_name_npy = Path(dir_save_subfolder, file_name + '.npy')
            np.save(file_name_npy,grid_img)
            file_name_mat = Path(dir_save_subfolder, file_name + '.mat')
            savemat(file_name_mat, {'grid_img': grid_img})
            
            # save together
            dir_save_all = Path(dir_save_main, 'AOI_all')
            dir_save_all.mkdir(parents = True, exist_ok = True)
            
            file_name_npy = Path(dir_save_all, file_name + '.npy')
            np.save(file_name_npy,grid_img)
            file_name_mat = Path(dir_save_all, file_name + '.mat')
            savemat(file_name_mat, {'grid_img': grid_img})


def AOI_to_df(file_source_folder, grid_size):
    """
    fetch AOI information and store in a data frame

    Parameters
    ----------
    file_source_folder : String
        Path of media source.
    grid_size : (int,int)
        (#grid along height, #grid along width). The default is (24,48).
         
    Returns
    -------
    df_AOI : Pandas DataFrame
        columns = ['file_path', 'media', 'AOI_r', 'AOI_g', 'AOI_b', 'AOI_c', 'AOI_m', 'AOI_y'].

    """
    from pathlib import Path
    import pandas as pd
    
    # get files from source
    all_files = list(Path(file_source_folder).glob('*.jpg'))
    
    row_list=[]
    
    for file_path in all_files:
        
        media = file_path.stem
        result_dict = {'file_path': file_path, 'media': media}
        
        AOI_color = ['r', 'g', 'b', 'c', 'm', 'y']
        for color in AOI_color:
            grid_img = grid_img_AOI(file_path, grid_size, color)
            new_key = 'AOI_'+ color
            result_dict[new_key] = grid_img
        
        row_list.append(result_dict)

    df_AOI = pd.DataFrame(row_list, columns = ['file_path', 'media', 'AOI_r', 
                                               'AOI_g', 'AOI_b', 'AOI_c', 
                                               'AOI_m', 'AOI_y'])
    
    return df_AOI

#%% generating .png
def npy_to_img(file_path):
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    # load .npy data
    data = np.load(file_path)
    media = file_path.stem
    
    # convert to image
    plt.imshow(data, cmap='gray')
    plt.axis('off')
    
    # create saving path
    dir_save = Path(file_path.parent, 'plots')
    dir_save.mkdir(parents = True, exist_ok = True)
    saving_path = Path(dir_save, media+'.png')
    
    # save as .png
    plt.savefig(saving_path, bbox_inches='tight', pad_inches=0)
    

# not used, because there is a more elegant way : 
# all_files = list(Path(dir_source).rglob('*.npy'))
def find_all_files_from_dir(source_dir):
    import os
    from pathlib import Path
    dir_list = [x[0] for x in os.walk(source_dir)]

    all_files=[]
    for dir in dir_list:
        # get all .npy files in a folder
        all_files = all_files + list(Path(dir).glob('*.npy'))
    
    return all_files


#%% 
def read_df_grid_data(dir_source):
    '''
    read grid data stored in pickle files

    Parameters
    ----------
    dir_source : String
        source directory of the pickle files for grid data.
        ex. '/home/wxiao/Documents/for_Weiyi/MEDIA_grid_operation_summary'
    
    Returns
    -------
    A List of dataframe object [df_light_intensity, df_color, df_contrast, df_AOI]

    '''
    from pathlib import Path
    import pickle
    
    # df_light_intensity: columns = ['file_path','media','img_light_intensity']
    path_df_light_intensity = Path(dir_source, 'media_grid_H24_W48_light_intensity.p')
    with open(path_df_light_intensity,"rb") as f:
            df_light_intensity = pickle.load(f)

    # df_color: columns = ['file_path', 'media', 'img_mode', 'img_r', 
    #                       'img_g', 'img_b','img_h', 'img_s', 'img_v']    
    path_df_color = Path(dir_source, 'media_grid_H24_W48_color.p')
    with open(path_df_color,"rb") as f:
            df_color = pickle.load(f)
 
    # df_contrast: columns = ['file_path','media','edge_roberts', 
    #                           'edge_sobel', 'edge_scharr', 'edge_prewitt', 'edge_farid']   
    path_df_contrast = Path(dir_source, 'media_grid_H24_W48_contrast.p')
    with open(path_df_contrast,"rb") as f:
            df_contrast = pickle.load(f)

    # df_AOI: columns = ['file_path', 'media', 'AOI_r', 'AOI_g', 
    #                       'AOI_b', 'AOI_c', 'AOI_m', 'AOI_y']
    path_df_AOI = Path(dir_source, 'media_grid_H24_W48_AOI.p')
    with open(path_df_AOI,"rb") as f:
            df_AOI = pickle.load(f)
            
    return [df_light_intensity, df_color, df_contrast, df_AOI] 
            

