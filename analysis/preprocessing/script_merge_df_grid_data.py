#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 14:09:45 2020

@author: wxiao

There are 4 dataframes that records the ligtht intensity, color, contrast, AOI of each media
This script aim to:
    merge all these 4 dataframes
    save the new dataframe df_all
"""
if __name__ == "__main__":
    
    from mitgaze.extract_media_features import read_df_grid_data
    import numpy as np
    import pandas as pd
    from pathlib import Path
    
    dir_source = '/home/wxiao/Documents/for_Weiyi/MEDIA_grid_operation_summary'
    [df_light_intensity, df_color, df_contrast, df_AOI] = read_df_grid_data(dir_source)
    
    #%% actually light intensity is not the same as the V in HSV
    # arr1 = df_light_intensity.loc[df_light_intensity['media']=='c08_5_2_SilverSunlitDunes']['img_light_intensity'][0]
    # arr2 = df_color.loc[df_color['media']=='c08_5_2_SilverSunlitDunes']['img_v'][0]
    # print(np.sum(arr1!=arr2))
    
    #%% df_all concatenate all dataframe about the grid data
    df_light_intensity = df_light_intensity.drop(columns=['file_path'])
    df_color = df_color.drop(columns=['file_path'])
    df_contrast = df_contrast.drop(columns=['file_path'])
    df_AOI = df_AOI.drop(columns=['file_path'])
    
    df_all = pd.merge(df_light_intensity, df_color, how = 'outer', on = ['media'])
    df_all = pd.merge(df_all, df_contrast, how = 'outer', on = ['media'])
    df_all = pd.merge(df_all, df_AOI, how = 'outer', on = ['media'])
    
    # path for saving results
    saving_path = Path(dir_source, 'media_grid_H24_W48_all.p')
    # save dataframe to pickle file
    df_all.to_pickle(saving_path, protocol=4)
    


