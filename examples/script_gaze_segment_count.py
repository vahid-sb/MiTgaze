#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 10:56:20 2020.

@author: vbokharaie

This script count number of gaze points in each segment inside the [0, 0, W_SCREEN, H_SCREEN] frame.
A heatmap of raw x-y coordinates.
"""


if __name__ == "__main__":
    from pathlib import Path
    from mitgaze.util import gaze_count_file

    if_parallel = False
    grid_x = 48 # heatmap gridsize, x axis
    grid_y = 24 # heatmap gridsize, y axis
    try:
        from mitgaze.data_info import get_dir_source
        dir_save_parsed_media = get_dir_source('sal_TSV_parsed_media')
        dir_save_main = get_dir_source('sal_plot_media_basics')
        N_job = get_dir_source('N_cores')
    except ModuleNotFoundError:
        # specify your own folders for raw files and where to save parsed files.
        script_dir = Path(__file__).resolve().parent
        dir_save_parsed_media = Path(script_dir, 'parsed_media')
        N_job = 2

    dir_save = Path(dir_save_main, 'GAZE_COUNT_GRID_' + str(grid_x) + '_' + str(grid_y))
    all_files = list(dir_save_parsed_media.glob('*.tsv'))
    all_files = [x for x in all_files if not '1920x' in x.as_posix()]

    if if_parallel:
        from joblib import Parallel, delayed

        print('parallel processing, with ', N_job, ' cores')
        print('********************************************')
        Parallel(n_jobs=N_job)\
                (delayed(gaze_count_file)
                         (filename, dir_save, grid_x, grid_y)
                         for filename in all_files)

    else:  # if parallel-processing fails, run a for loop
        print('non-parallel processing')
        print('********************************************')
        for filename in all_files:
            gaze_count_file(filename, dir_save, grid_x, grid_y)
