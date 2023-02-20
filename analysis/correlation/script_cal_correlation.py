#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 04:23:53 2020

@author: wxiao
"""
if __name__ == "__main__":
    from pathlib import Path
    import pickle
    from scipy.stats import pearsonr
    import pandas as pd
    
    
    #%% file io
    dir_source = Path('/home/wxiao/Documents/for_Weiyi/MEDIA_grid_operation_summary/')
    dir_save = Path('/home/wxiao/Documents/for_Weiyi/analysis_results/correlation_results')
    dir_save.mkdir(parents = True, exist_ok = True)
    
    
    #%% read heatmap data
    path_df_heatmap_summary = Path(dir_source, 'HEATMAP_DATA_SUMMARY.p')
    # columns of df_heatmap_summary ['file_path','par','media','heatmap_data']
    with open(path_df_heatmap_summary,"rb") as f:
            df_heatmap_summary = pickle.load(f)
        
    
    #%% light intensity
    path_df_light_intensity = Path(dir_source, 'media_grid_H24_W48_light_intensity.p')
    # columns of df_light_intensity ['file_path','media','img_light_intensity']
    with open(path_df_light_intensity,"rb") as f:
            df_light_intensity = pickle.load(f)
            
            
    row_list=[]
    # iterate through participants list
    for par in df_heatmap_summary.par.unique():
        
        # extract all rows associated the participant, each row represent a different media
        df_heatmap_per_par = df_heatmap_summary.loc[df_heatmap_summary['par']==par]
        # iterate through the media list, get the media name and heatmap_data
        for index, row in df_heatmap_per_par.iterrows():
            media = row['media']
            heatmap_data = row['heatmap_data']
            row_dict = {'par': par, 'media': media}
            
            # use media name to find the corresponding img_light_intensity
            grid_img = df_light_intensity.loc[df_light_intensity['media']==media]['img_light_intensity'].iloc[0]
           
            # calculate correlations
            r_corr_light_intensity, p_corr_light_intensity = pearsonr(grid_img.flatten(), heatmap_data.flatten())
            
            row_list.append({'par': par, 'media': media, 'r_corr_light_intensity': r_corr_light_intensity, 'p_corr_light_intensity': p_corr_light_intensity })
    
    # save data into data frame and save data frame as pickle file
    df_correlation_light_intensity= pd.DataFrame(row_list, columns = ['par', 'media', 'r_corr_light_intensity', 'p_corr_light_intensity'])
    
    pkl_file = Path(dir_save, 'correlation_light_intensity.p')
    df_correlation_light_intensity.to_pickle(pkl_file, protocol=4)
    
    
    #%% color
    path_df_color = Path(dir_source, 'media_grid_H24_W48_color.p')
    # columns of df_color ['file_path', 'media', 'img_r', 'img_g', 'img_b','img_h', 'img_s', 'img_v']
    with open(path_df_color,"rb") as f:
            df_color = pickle.load(f)
    
            
    row_list=[]
    # iterate through participants list
    for par in df_heatmap_summary.par.unique():
        
        # extract all rows associated the participant, each row represent a different media
        df_heatmap_per_par = df_heatmap_summary.loc[df_heatmap_summary['par']==par]
        # iterate through the media list, get the media name and heatmap_data
        for index, row in df_heatmap_per_par.iterrows():
            media = row['media']
            heatmap_data = row['heatmap_data']
            row_dict = {'par': par, 'media': media}
            
            # determine if the media was gray or colored
            img_mode = df_color.loc[df_color['media']==media]['img_mode'].iloc[0]
            row_dict['img_mode'] = img_mode
            
            channel_name = ['r','g','b','h','s','v']
            # use media name to find the corresponding r,g,b,h,s,v    
            if img_mode =='RGB':  # colored
                for channel in channel_name:
                    col_name = 'img_' + channel
                    grid_img = df_color.loc[df_color['media']==media][col_name].iloc[0]
                    
                    #calculate correlation
                    r_corr, p_corr = pearsonr(grid_img.flatten(), heatmap_data.flatten())
                    new_key_r_corr = 'r_corr_' + channel
                    new_key_p_corr = 'p_corr_' + channel
                    row_dict[new_key_r_corr] = r_corr
                    row_dict[new_key_p_corr] = p_corr
                    
            row_list.append(row_dict)
    
    # save data into data frame and save data frame as pickle file
    df_correlation_color= pd.DataFrame(row_list, columns = ['par', 'media','img_mode',\
                                                            'r_corr_r', 'p_corr_r','r_corr_g', 'p_corr_g','r_corr_b','p_corr_b',\
                                                            'r_corr_h', 'p_corr_h','r_corr_s', 'p_corr_s','r_corr_v','p_corr_v'])
                                                                      
    pkl_file = Path(dir_save, 'correlation_color.p')
    df_correlation_color.to_pickle(pkl_file, protocol=4)    
            
    
    #%% contrast
    path_df_contrast = Path(dir_source, 'media_grid_H24_W48_contrast.p')
    # columns of df_contrast ['file_path','media','edge_roberts', 'edge_sobel', 'edge_scharr', 'edge_prewitt', 'edge_farid']
    with open(path_df_contrast,"rb") as f:
            df_contrast = pickle.load(f)
    
    
    row_list=[]
    # iterate through participants list
    for par in df_heatmap_summary.par.unique():
        
        # extract all rows associated the participant, each row represent a different media
        df_heatmap_per_par = df_heatmap_summary.loc[df_heatmap_summary['par']==par]
        # iterate through the media list, get the media name and heatmap_data
        for index, row in df_heatmap_per_par.iterrows():
            media = row['media']
            heatmap_data = row['heatmap_data']
            row_dict = {'par': par, 'media': media}
            
            algorithm_name = ['edge_roberts', 'edge_sobel', 'edge_scharr', 'edge_prewitt', 'edge_farid']
            
            # use media name to find the corresponding edge files
            for col_name in algorithm_name:
                grid_img = df_contrast.loc[df_contrast['media']==media][col_name].iloc[0]
                
                #calculate correlation
                r_corr, p_corr = pearsonr(grid_img.flatten(), heatmap_data.flatten())
                new_key_r_corr = 'r_corr_' + col_name
                new_key_p_corr = 'p_corr_' + col_name
                row_dict[new_key_r_corr] = r_corr
                row_dict[new_key_p_corr] = p_corr
        
            row_list.append(row_dict)
            
    # save data into data frame and save data frame as pickle file
    df_correlation_contrast = pd.DataFrame(row_list, columns = ['par', 'media',\
                                                                'r_corr_edge_roberts', 'p_corr_edge_roberts',\
                                                                'r_corr_edge_sobel', 'p_corr_edge_sobel',\
                                                                'r_corr_edge_scharr', 'p_corr_edge_scharr',\
                                                                'r_corr_edge_prewitt', 'p_corr_edge_prewitt',\
                                                                'r_corr_edge_farid', 'p_corr_edge_farid'])
        
    pkl_file = Path(dir_save, 'correlation_contrast.p')
    df_correlation_contrast.to_pickle(pkl_file, protocol=4)
    
    
    #%% AOI
    path_df_AOI= Path(dir_source, 'media_grid_H24_W48_AOI.p')
    # columns of df_color ['file_path', 'media', 'AOI_r', 'AOI_g', 'AOI_b','AOI_c', 'AOI_m', 'AOI_y']
    with open(path_df_AOI,"rb") as f:
            df_AOI = pickle.load(f)
    
    row_list=[]
    # iterate through participants list
    for par in df_heatmap_summary.par.unique():
    
        # extract all rows associated the participant, each row represent a different media
        df_heatmap_per_par = df_heatmap_summary.loc[df_heatmap_summary['par']==par]
        # iterate through the media list, get the media name and heatmap_data
        for index, row in df_heatmap_per_par.iterrows():
            media = row['media']
            heatmap_data = row['heatmap_data']
            row_dict = {'par': par, 'media': media}
            
            AOI_color = ['r', 'g', 'b', 'c', 'm', 'y']
            for color in AOI_color:
                col_name = 'AOI_' + color
                grid_img = df_AOI.loc[df_AOI['media']==media][col_name].iloc[0]
                if len(grid_img)>0:
                    # calculate correlation
                    r_corr, p_corr = pearsonr(grid_img.flatten(), heatmap_data.flatten())
                    new_key_r_corr = 'r_corr_' + col_name
                    new_key_p_corr = 'p_corr_' + col_name
                    row_dict[new_key_r_corr] = r_corr
                    row_dict[new_key_p_corr] = p_corr
                
            row_list.append(row_dict)
    
    # save data into data frame and save data frame as pickle file
    df_correlation_AOI= pd.DataFrame(row_list, columns = ['par', 'media',\
                                                            'r_corr_AOI_r', 'p_corr_AOI_r',\
                                                            'r_corr_AOI_g', 'p_corr_AOI_g',\
                                                            'r_corr_AOI_b','p_corr_AOI_b',\
                                                            'r_corr_AOI_c', 'p_corr_AOI_c',\
                                                            'r_corr_AOI_m', 'p_corr_AOI_m',\
                                                            'r_corr_AOI_y','p_corr_AOI_y'])
                                                                      
    pkl_file = Path(dir_save, 'correlation_AOI.p')
    df_correlation_AOI.to_pickle(pkl_file, protocol=4)    
    
