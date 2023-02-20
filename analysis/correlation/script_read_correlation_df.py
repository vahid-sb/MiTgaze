#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 08:36:53 2020

@author: wxiao
"""

if __name__ == "__main__":
    from pathlib import Path
    import pickle
    
    
    #%% file io
    dir_source = Path('/home/wxiao/Documents/for_Weiyi/analysis_results/correlation_results')


    #%% light intensity
    # columns of df_light_intensity ['par', 'media', 'r_corr_light_intensity', 'p_corr_light_intensity']
    
    path_df_light_intensity = Path(dir_source, 'correlation_light_intensity.p')
    with open(path_df_light_intensity,"rb") as f:
            df_light_intensity = pickle.load(f)
            
            
    #%% color
    # columns of df_color 
    # ['par', 'media','img_mode',\
    # 'r_corr_r', 'p_corr_r','r_corr_g', 'p_corr_g','r_corr_b','p_corr_b',\
    # 'r_corr_h', 'p_corr_h','r_corr_s', 'p_corr_s','r_corr_v','p_corr_v']
    
    path_df_color = Path(dir_source, 'correlation_color.p')
    with open(path_df_color,"rb") as f:
            df_color = pickle.load(f)
            
            
    #%% contrast
    # columns of df_contrast 
    # ['par', 'media',\
    # 'r_corr_edge_roberts', 'p_corr_edge_roberts',\
    # 'r_corr_edge_sobel', 'p_corr_edge_sobel',\
    # 'r_corr_edge_scharr', 'p_corr_edge_scharr',\
    # 'r_corr_edge_prewitt', 'p_corr_edge_prewitt',\
    # 'r_corr_edge_farid', 'p_corr_edge_farid']
    
    path_df_contrast = Path(dir_source, 'correlation_contrast.p')
    with open(path_df_contrast,"rb") as f:
            df_contrast = pickle.load(f)
    
    
    #%% AOI
    # # columns of df_AOI
    # ['par', 'media',\
    # 'r_corr_AOI_r', 'p_corr_AOI_r',\
    # 'r_corr_AOI_g', 'p_corr_AOI_g',\
    # 'r_corr_AOI_b','p_corr_AOI_b',\
    # 'r_corr_AOI_c', 'p_corr_AOI_c',\
    # 'r_corr_AOI_m', 'p_corr_AOI_m',\
    # 'r_corr_AOI_y','p_corr_AOI_y']
    
    path_df_AOI = Path(dir_source, 'correlation_AOI.p')
    with open(path_df_AOI,"rb") as f:
            df_AOI = pickle.load(f)