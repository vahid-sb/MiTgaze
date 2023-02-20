#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 15:14:55 2020.

@author: vbokharaie
"""

if __name__ == "__main__":

    # %% load main tsv file, splot and save to csv parts
    from pathlib import Path
    from mitgaze.filters import plot_edges

    ### edges
    for H in [1080, 1200]:
        if H==1080:
            dir_save = Path('/data/DATA_OUTPUTS/ET/DATA/Saliency/MEDIA_filtered/edges_1920x1080/')
            dir_source = Path('/data/DATA_OUTPUTS/ET/DATA/Saliency/MEDIA/rescaled/screen_1920x1080/')

        else:
            dir_save = Path('/data/DATA_OUTPUTS/ET/DATA/Saliency/MEDIA_filtered/edges_1920x1200/')
            dir_source = Path('/data/DATA_OUTPUTS/ET/DATA/Saliency/MEDIA/rescaled/screen_1920x1200/')
        list_media_files = list(dir_source.rglob("*"))  #list

        try:  # see if parallel processing works
            from joblib import Parallel, delayed
            N_job = 8
            print('parallel processing')
            Parallel(n_jobs=N_job)\
                (delayed(plot_edges)(file, dir_save)\
                 for file in list_media_files)
        except:  # if parallel-processing fails, run a for loop
            print('non-parallel processing')
            for file in list_media_files:
                plot_edges(file, dir_save)

