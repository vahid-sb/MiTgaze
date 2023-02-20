#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 14:03:59 2020.

@author: vbokharaie

This script read the raw Tobii files, removes rows with no media, parse df into
segments for each particiapnt, then save each to a TSV file.

Uses MiTgaze
"""

def func_par(filename, dir_save_parsed_par):
    """
    Wrapper for parse_df_par

    Parameters
    ----------
    filename : str or pathlib.Path
        name of TSV/CSV file.
    dir_save_parsed_par : str or pathlib.Path
        where to save the prased files.

    Returns
    -------
    None.

    """
    from pathlib import Path

    from mitgaze.file_io import parse_df_par

    filename = Path(filename)
    dir_save_parsed_par = Path(dir_save_parsed_par)
    print('Parse ', filename, ' based on participant')
    list_temp = parse_df_par(filename, dir_save_parsed_par, col_2_del=['Unnamed: 74'])

# %% main
if __name__ == "__main__":
    from pathlib import Path
    from joblib import Parallel, delayed

    from mitgaze.file_io import parse_df_media
    try:
        from mitgaze.data_info import get_dir_source
        dir_source_raw = get_dir_source('sal_TSV_raw')
        dir_save_parsed_par = get_dir_source('sal_TSV_parsed_par')
        dir_save_parsed_media = get_dir_source('sal_TSV_parsed_media')
        N_job = get_dir_source('N_cores')
    except ModuleNotFoundError:
        # specify your own folders for raw files and where to save parsed files.
        script_dir = Path(__file__).resolve().parent
        dir_source_raw = Path(script_dir, 'raw_files')   # CHANGE!
        dir_save_parsed_par = Path(script_dir, 'parsed_par')
        dir_save_parsed_media = Path(script_dir, 'parsed_media')
        N_job = 2
    # parse data file based on 'Participant Name'
    if_parse_par = True
    if if_parse_par:
        all_files = list(dir_source_raw.glob('*.tsv'))
        Parallel(n_jobs=N_job)\
                    (delayed(func_par)(filename, dir_save_parsed_par)\
                     for filename in all_files)


    # parse data file baed on 'Presented Media Name'
    if_parse_media = True
    if if_parse_media:
        print('Parse files which are already parsed to participant based on media names')
        parse_df_media(dir_save_parsed_par, dir_save_parsed_media)