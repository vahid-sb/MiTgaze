#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 10:52:28 2020

@author: wxiao
"""

if __name__ == "__main__":
    
    from pathlib import Path
    from mitgaze.extract_media_features import light_intensity_to_df, color_to_df, contrast_to_df, AOI_to_df

   
    # directory to save the results
    dir_save = Path('/home/wxiao/Documents/for_Weiyi/MEDIA_grid_operation_summary')
    
    #%% light intensity
    # media source
    dir_source = Path('/home/wxiao/Documents/for_Weiyi/MEDIA/rescaled/screen_1920x1080 (copy)')
    # path for saving results
    saving_path = Path(dir_save, 'media_grid_H24_W48_light_intensity.p')
    
    df_light_intensity = light_intensity_to_df(dir_source, grid_size=(24,48))
    
    # save dataframe to pickle file
    df_light_intensity.to_pickle(saving_path, protocol=4)
    
    #%% color
    # media source
    dir_source = Path('/home/wxiao/Documents/for_Weiyi/MEDIA/rescaled/screen_1920x1080 (copy)')
    # path for saving results
    saving_path = Path(dir_save, 'media_grid_H24_W48_color.p')
    
    df_color = color_to_df(dir_source, grid_size=(24,48))
    
    # save dataframe to pickle file
    df_color.to_pickle(saving_path, protocol=4)
    
    #%% contrast
    # media source
    dir_source = Path('/home/wxiao/Documents/for_Weiyi/MEDIA/rescaled/screen_1920x1080 (copy)')
    # path for saving results
    saving_path = Path(dir_save, 'media_grid_H24_W48_contrast.p')
    
    df_contrast = contrast_to_df(dir_source, grid_size=(24,48))
    
    # save dataframe to pickle file
    df_contrast.to_pickle(saving_path, protocol=4)
    
    #%% AOI
    # media source
    dir_source = Path('/home/wxiao/Documents/for_Weiyi/MEDIA/AOI_scaled/screen_1920x1080')
    # path for saving results
    saving_path = Path(dir_save, 'media_grid_H24_W48_AOI.p')
    
    df_AOI = AOI_to_df(dir_source, grid_size=(24,48))
    
    # save dataframe to pickle file
    df_AOI.to_pickle(saving_path, protocol=4)

    