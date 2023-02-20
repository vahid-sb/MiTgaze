#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 11:45:20 2020

@author: wxiao

read df_all that includes all information about light intensity, color, contrast, AOI
prepare input matrix(property_array) for later analysis
property_array have shape 24x48x14, 14 = 1(light intensity)+6(color)+1(contrast)+6(AOI)
save the new dataframe with columns = ['media','img_mode','property_array']
'media': name of the image. i.e. 'a19_6_11_BuckskinViewFromCenterOfEarth'
'img_mode': RGB or L, depends on whether it is colored or gray
'property_array': np.ndarray of shape 24x48x14
"""
if __name__ == "__main__":
    from pathlib import Path
    import pickle
    import numpy as np
    import pandas as pd
    
    dir_source = Path('/home/wxiao/Documents/for_Weiyi/MEDIA_grid_operation_summary')
    dir_save = Path('/home/wxiao/Documents/for_Weiyi/MEDIA_property_array_24x48x14')
    dir_save.mkdir(parents = True, exist_ok = True)
    
    # df_all: columns = ['file_path', 'media', 'img_light_intensity', 'img_mode', 'img_r',
    #                    'img_g', 'img_b', 'img_h', 'img_s', 'img_v', 'edge_roberts',
    #                    'edge_sobel', 'edge_scharr', 'edge_prewitt', 'edge_farid', 'AOI_r',
    #                    'AOI_g', 'AOI_b', 'AOI_c', 'AOI_m', 'AOI_y']
    path_df_all = Path(dir_source, 'media_grid_H24_W48_all.p')
    
    
    with open(path_df_all,"rb") as f:
            df_all = pickle.load(f)
    
    # 'edge_roberts' could be replaced by 'edge_sobel', 'edge_scharr', 'edge_prewitt', 'edge_farid'
    media_property = ['img_light_intensity','img_r',
                    'img_g', 'img_b', 'img_h', 'img_s', 'img_v', 'edge_roberts',
                    'AOI_r', 'AOI_g', 'AOI_b', 'AOI_c', 'AOI_m', 'AOI_y']

    # if arr_in is nan or empty list, replace with zero matrix    
    def array_checker(arr_in):
        if not isinstance(arr_in, np.ndarray):
            arr_in = np.zeros((24,48))
        return arr_in
        
    row_list=[]
    for df_idx, row in df_all.iterrows():
        property_array = np.zeros((24,48,len(media_property)))
        for i in range(len(media_property)):
            property_array[:,:,i] = array_checker(row[media_property[i]])
        row_list.append({'media': row['media'], 'img_mode': row['img_mode'], 'property_array': property_array})
        
        # save property_array as .npy
        file_name = row['media']+'_property_array_24x48x14'
        saving_path_npy= Path(dir_save, file_name + '.npy')
        np.save(saving_path_npy,property_array)
        
    # save to dataframe
    df_media_property = pd.DataFrame(row_list, columns = ['media','img_mode','property_array'])
    # save to pickle
    saving_path_pickle = Path(dir_source, 'media_grid_H24_W48_media_property.p')
    df_media_property.to_pickle(saving_path_pickle, protocol=4)
    