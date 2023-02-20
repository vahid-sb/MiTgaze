#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 20:37:21 2020.

@author: vbokharaie

This script is used to perform various analysis on the recording of each media
over all participants. Also plotting.
"""


if __name__ == "__main__":
    from pathlib import Path
    from mitgaze.bplots import plot_basics_parsed_media

    if_parallel = True
    grid_x = 40  # heatmap gridsize, x axis
    grid_y = 20  # heatmap gridsize, y axis
    try:
        from mitgaze.data_info import get_dir_source
        dir_save_parsed_media = get_dir_source('sal_TSV_parsed_media')
        dir_save_main = get_dir_source('sal_plot_media_basics')
        dir_source_image_1200 = get_dir_source('sal_media_scaled_1200')
        dir_source_image_1080 = get_dir_source('sal_media_scaled_1080')
        N_job = get_dir_source('N_cores')
    except ModuleNotFoundError:
        # specify your own folders for raw files and where to save parsed files.
        script_dir = Path(__file__).resolve().parent
        dir_save_parsed_media = Path(script_dir, 'parsed_media')
        dir_source_image_1200 = Path(script_dir, 'media_scaled_1200')
        dir_source_image_1080 = Path(script_dir, 'media_scaled_1080')
        dir_save_main = Path(script_dir, 'sal_plot_media_basics')
        N_job = 8

    all_files = list(dir_save_parsed_media.glob('*.tsv'))
    all_files = [x for x in all_files if '1920x' not in x.as_posix()]

    if if_parallel:
        from joblib import Parallel, delayed

        print('parallel processing, with ', N_job, ' cores')
        print('********************************************')
        Parallel(n_jobs=N_job)(delayed(plot_basics_parsed_media)
                               (filename,
                               dir_save_main,
                               dir_source_image_1200,
                               dir_source_image_1080,
                               gridsize_x=grid_x,  # heatmap gridsize, x axis
                               gridsize_y=grid_y,  # heatmap gridsize, y axis
                               if_gaze_plots=False)
                               for filename in all_files)

    else:  # if parallel-processing fails, run a for loop
        print('non-parallel processing')
        print('********************************************')
        for filename in all_files:
            plot_basics_parsed_media(filename,
                                     dir_save_main,
                                     dir_source_image_1200,
                                     dir_source_image_1080,
                                     gridsize_x=grid_x,  # heatmap gridsize, x axis
                                     gridsize_y=grid_y,  # heatmap gridsize, y axis
                                     if_gaze_plots=False)
