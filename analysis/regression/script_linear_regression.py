#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 14:07:23 2020

@author: wxiao
"""
        
if __name__ == "__main__":
    
    import pickle
    from pathlib import Path
    from mitgaze.regression_classification import linear_regression_analysis
    
    #%% data preparation
    # get df_media_property which contains property_array of each media
    # get df_heatmap which contains all heatmaps from all participants
    
    # source directory for grid operations
    dir_source = Path('/home/wxiao/Documents/for_Weiyi/MEDIA_grid_operation_summary')
    
    # df_media_property: columns = ['media','img_mode','property_array']
    path_df_media_property = Path(dir_source, 'media_grid_H24_W48_media_property.p')
    with open(path_df_media_property,"rb") as f:
            df_media_property = pickle.load(f)
            
    # df_heatmap: columns = ['file_path', 'par', 'media', 'heatmap_data']
    path_df_heatmap = Path(dir_source, 'HEATMAP_DATA_SUMMARY.p')
    with open(path_df_heatmap,"rb") as f:
            df_heatmap = pickle.load(f)

#%%
    # media directory.
    dir_media = Path('/home/wxiao/Documents/for_Weiyi/MEDIA/rescaled/screen_1920x1080 (copy)')
    
    mode = 2
    if mode == 1:
        # directory to store results.
        dir_save = Path('/home/wxiao/Documents/for_Weiyi/analysis_results/linear_regression_results_#feature24x48x14')
        dir_save.mkdir(parents = True, exist_ok = True)
    elif mode == 2:
        # directory to store results.
        dir_save = Path('/home/wxiao/Documents/for_Weiyi/analysis_results/linear_regression_results_#feature14')
        dir_save.mkdir(parents = True, exist_ok = True)
    
    # get list of all participants
    par_list = df_heatmap.par.unique()
    
    if_parallel = True
    if if_parallel:
        from joblib import Parallel, delayed
        import multiprocessing
        N_job = multiprocessing.cpu_count()
        print('parallel processing, with ', N_job, ' cores')
        print('********************************************')
        Parallel(n_jobs=N_job)\
                (delayed(linear_regression_analysis)
                          (mode, 
                           df_heatmap,
                           df_media_property,
                           par,
                           dir_media,
                           dir_save)
                           for par in par_list)
    
    else:  # if parallel-processing fails, run a for loop
        print('non-parallel processing')
        print('********************************************')
        for par in par_list:
            linear_regression_analysis(mode, df_heatmap, df_media_property, par, dir_media, dir_save)