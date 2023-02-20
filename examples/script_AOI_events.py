#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 01:52:46 2020.

@author: vbokharaie
"""

if __name__ == "__main__":
    """
    Reads the dataframe including gaze data for a media for all participants.
    then finds the in-out times to AOI (relative to TTL_OUT time) for each particpants.
    saves the results in a data_frame and then df.to_pickle
    AOIs are assumed to be specified with solid RGBCYM colours, i.e. an image identical
    to the original image, with AOI being painted over in primary/secondary colours.

    """

    from pathlib import Path
    import pandas as pd
    import numpy as np
    import matplotlib.image as mpimg

    from mitgaze.AOI import find_AOI, AOI_in_out_indices
    try:
        from mitgaze.data_info import get_dir_source
        dir_save_parsed_media = get_dir_source('sal_TSV_parsed_media')
        dir_AOI_1200 = get_dir_source('sal_media_AOI_scaled_1200')
        dir_AOI_1080 = get_dir_source('sal_media_AOI_scaled_1080')
        dir_save_AOI_time = get_dir_source('sal_AOI_events')
    except ModuleNotFoundError:
        # specify your own folders for raw files and where to save parsed files.
        script_dir = Path(__file__).resolve().parent
        dir_save_parsed_media = Path(script_dir, 'parsed_media')
        dir_AOI_1200 = Path(script_dir, 'media_AOI_scaled_1200')
        dir_AOI_1080 = Path(script_dir, 'media_AOI_scaled_1080')
        dir_save_AOI_time = Path(script_dir, 'AOI_times')
    dir_save_AOI_time.mkdir(exist_ok=True, parents=True)

    all_files = list(dir_save_parsed_media.glob('*.tsv'))
    all_files = [x for x in all_files if not '1920x' in x.as_posix()]

    all_files.sort()
    columns_df = ['media', 'participant',
                  'time_R', 'time_G', 'time_B',
                  'time_C', 'time_Y', 'time_M',
                  'ind_R_rel_TTL', 'ind_G_rel_TTL', 'ind_B_rel_TTL',
                  'ind_C_rel_TTL', 'ind_M_rel_TTL', 'ind_Y_rel_TTL']
    df_media_sub_AOI_times = pd.DataFrame(columns=columns_df)
    cc_df = 0
    for idx, filename in enumerate(all_files):
        print('----------------------------------------------')
        print(str(idx+1), '---', filename.stem)

        df_media = pd.read_csv(filename, sep='\t', low_memory=False)

        par_list = df_media['Participant name'].unique()
        par_list.sort()
        for participant in par_list:

            df_media_sub = df_media.loc[df_media['Participant name']==participant]

            image_AOI_name = filename.stem[6:] + '.jpg'
            H = df_media_sub['Recording resolution height'].unique()[0]
            if H ==1200:
                dir_source_AOI =dir_AOI_1200
            elif H == 1080:
                dir_source_AOI = dir_AOI_1080

            file_AOI = Path(dir_source_AOI, image_AOI_name)
            try:
                img=mpimg.imread(file_AOI)
            except FileNotFoundError:
                print('Could not find AOI_image. Skipped ...')
                continue
            my_mask_r = find_AOI(file_AOI, 'r')
            my_mask_g = find_AOI(file_AOI, 'g')
            my_mask_b = find_AOI(file_AOI, 'b')
            my_mask_c = find_AOI(file_AOI, 'c')
            my_mask_m = find_AOI(file_AOI, 'm')
            my_mask_y = find_AOI(file_AOI, 'y')

            x = np.array(df_media_sub['Gaze point X'])
            y = np.array(df_media_sub['Gaze point Y'])
            ind_r = AOI_in_out_indices(x, y, my_mask_r)
            ind_g = AOI_in_out_indices(x, y, my_mask_g)
            ind_b = AOI_in_out_indices(x, y, my_mask_b)
            ind_c = AOI_in_out_indices(x, y, my_mask_c)
            ind_m = AOI_in_out_indices(x, y, my_mask_m)
            ind_y = AOI_in_out_indices(x, y, my_mask_y)

            Rec_time = list(df_media_sub['Recording timestamp'])
            # time_ET = list(df_media_sub['Eyetracker timestamp'])
            ind_TTL_out = df_media_sub.index[df_media_sub['Event'] == 'TTL out'][0]
            ind_0 = df_media_sub.index[0]
            ind_TTL_out_rel = ind_TTL_out - ind_0
            # ind_first_non_nan = [idx for idx , el in enumerate(x) if not np.isnan(el)][0]
            s_time = Rec_time[ind_TTL_out_rel]

            time_r = [((Rec_time[x] - s_time)/1e3, (Rec_time[y] - s_time)/1e3) for (x, y) in ind_r]
            time_g = [((Rec_time[x] - s_time)/1e3, (Rec_time[y] - s_time)/1e3) for (x, y) in ind_g]
            time_b = [((Rec_time[x] - s_time)/1e3, (Rec_time[y] - s_time)/1e3) for (x, y) in ind_b]
            time_c = [((Rec_time[x] - s_time)/1e3, (Rec_time[y] - s_time)/1e3) for (x, y) in ind_c]
            time_m = [((Rec_time[x] - s_time)/1e3, (Rec_time[y] - s_time)/1e3) for (x, y) in ind_m]
            time_y = [((Rec_time[x] - s_time)/1e3, (Rec_time[y] - s_time)/1e3) for (x, y) in ind_y]

            ind_r_rel2_TTL_out = [(x-ind_TTL_out_rel, y-ind_TTL_out_rel) for (x,y) in ind_r]
            ind_g_rel2_TTL_out = [(x-ind_TTL_out_rel, y-ind_TTL_out_rel) for (x,y) in ind_g]
            ind_b_rel2_TTL_out = [(x-ind_TTL_out_rel, y-ind_TTL_out_rel) for (x,y) in ind_b]
            ind_c_rel2_TTL_out = [(x-ind_TTL_out_rel, y-ind_TTL_out_rel) for (x,y) in ind_c]
            ind_m_rel2_TTL_out = [(x-ind_TTL_out_rel, y-ind_TTL_out_rel) for (x,y) in ind_m]
            ind_y_rel2_TTL_out = [(x-ind_TTL_out_rel, y-ind_TTL_out_rel) for (x,y) in ind_y]

            df_media_sub_AOI_times.at[cc_df, 'media'] = filename.stem
            df_media_sub_AOI_times.at[cc_df, 'participant'] = participant
            df_media_sub_AOI_times.at[cc_df, 'time_R'] = time_r
            df_media_sub_AOI_times.at[cc_df, 'time_G'] = time_g
            df_media_sub_AOI_times.at[cc_df, 'time_B'] = time_b
            df_media_sub_AOI_times.at[cc_df, 'time_C'] = time_c
            df_media_sub_AOI_times.at[cc_df, 'time_Y'] = time_y
            df_media_sub_AOI_times.at[cc_df, 'time_M'] = time_m
            df_media_sub_AOI_times.at[cc_df, 'ind_R_rel_TTL'] = ind_r_rel2_TTL_out
            df_media_sub_AOI_times.at[cc_df, 'ind_G_rel_TTL'] = ind_g_rel2_TTL_out
            df_media_sub_AOI_times.at[cc_df, 'ind_B_rel_TTL'] = ind_b_rel2_TTL_out
            df_media_sub_AOI_times.at[cc_df, 'ind_C_rel_TTL'] = ind_c_rel2_TTL_out
            df_media_sub_AOI_times.at[cc_df, 'ind_M_rel_TTL'] = ind_m_rel2_TTL_out
            df_media_sub_AOI_times.at[cc_df, 'ind_Y_rel_TTL'] = ind_y_rel2_TTL_out
            cc_df = cc_df + 1

    filename = 'df_media_sub_AOI_times'
    file_p = Path(dir_save_AOI_time, filename + '.p')
    file_csv = Path(dir_save_AOI_time, filename + '.csv')
    df_media_sub_AOI_times.to_pickle(file_p, protocol=4)
    df_media_sub_AOI_times.to_csv(file_csv)