#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 09:09:05 2020

@author: wxiao
"""

if __name__ == "__main__":
    
    from pathlib import Path
    from mitgaze.extract_media_features import npy_to_img
    
    # get files from source
    dir_source = Path('/home/wxiao/Documents/for_Weiyi/MEDIA_grid_operation_H24_W48')
    all_files = list(dir_source.rglob('*.npy'))

    
    if_parallel = True
    if if_parallel:
        from joblib import Parallel, delayed
        import multiprocessing
        N_job = multiprocessing.cpu_count()
        print('parallel processing, with ', N_job, ' cores')
        print('********************************************')
        Parallel(n_jobs=N_job)\
                (delayed(npy_to_img)
                          (file_path,)
                          for file_path in all_files)
    
    else:  # if parallel-processing fails, run a for loop
        print('non-parallel processing')
        print('********************************************')
        for file_path in all_files:
            npy_to_img(file_path)


