#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 14:50:22 2020

@author: wxiao
"""

if __name__ == "__main__":

    import pickle
    from pathlib import Path
    from mitgaze.kmeans_clustering import kmeans_clustering_block
    import numpy as np
    
#%%
    # read dataframe 
    pkl_file = Path('/home/wxiao/Documents/for_Weiyi/MEDIA_grid_operation_summary/HEATMAP_DATA_SUMMARY.p')
    with open(pkl_file,"rb") as f:
            df_heatmap_summary = pickle.load(f)

    # media directory
    dir_media = Path('/home/wxiao/Documents/for_Weiyi/MEDIA/rescaled/screen_1920x1080 (copy)')
    # directory to save clustering results
    dir_save = Path('/home/wxiao/Documents/for_Weiyi/analysis_results/kmeans_clustering_results')
    dir_save.mkdir(parents = True, exist_ok = True)

    # find the list of all media
    media_list = df_heatmap_summary.media.unique()
    
    normalization= True
    cluster_list= np.arange(2,13)
    clustering_info = {'normalization': normalization, 'cluster_list': cluster_list}
    
    if_parallel = True
    if if_parallel:
        from joblib import Parallel, delayed
        import multiprocessing
        N_job = multiprocessing.cpu_count()
        print('parallel processing, with ', N_job, ' cores')
        print('********************************************')
        Parallel(n_jobs=N_job)\
                (delayed(kmeans_clustering_block)
                          (dir_media, 
                           dir_save, 
                           df_heatmap_summary, 
                           media, 
                           clustering_info)
                          for media in media_list)
    
    else:  # if parallel-processing fails, run a for loop
        print('non-parallel processing')
        print('********************************************')
        for media in media_list:
            kmeans_clustering_block(dir_media, dir_save, df_heatmap_summary, media, clustering_info)