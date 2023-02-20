#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 04:59:16 2020

@author: wxiao
"""

if __name__ == "__main__":
    
    from pathlib import Path
    from mitgaze.extract_media_features import AOI_analysis
    
    # get files from source
    dir_source = Path('/home/wxiao/Documents/for_Weiyi/MEDIA/AOI_scaled/screen_1920x1080')
    all_files = list(dir_source.glob('*.jpg'))
    
    # path for saving results
    grid_H, grid_W = (24,48)
    dir_name = '/home/wxiao/Documents/for_Weiyi/MEDIA_grid_operation_H' + str(grid_H) + '_W' + str(grid_W)
    dir_save_main = Path(dir_name, 'AOI')
    dir_save_main.mkdir(parents=True, exist_ok=True)
    
    if_parallel = True
    if if_parallel:
        from joblib import Parallel, delayed
        import multiprocessing
        N_job = multiprocessing.cpu_count()
        print('parallel processing, with ', N_job, ' cores')
        print('********************************************')
        Parallel(n_jobs=N_job)\
                (delayed(AOI_analysis)
                          (filename,
                          dir_save_main,
                          grid_size=(grid_H, grid_W))
                          for filename in all_files)
    
    else:  # if parallel-processing fails, run a for loop
        print('non-parallel processing')
        print('********************************************')
        for file_path in all_files:
            AOI_analysis(file_path, dir_save_main, grid_size=(grid_H, grid_W))

