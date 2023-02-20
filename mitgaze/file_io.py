#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 14:44:04 2019.

@author: vbokharaie
"""
# pylint: disable=line-too-long
# pylint: disable=wrong-import-position


# %% save_mat
def save_mat(filename, var):
    folder = filename.parent
    folder.mkdir(parents=True, exist_ok=True)
    my_dict = {'var_name': var}
    from scipy.io import savemat
    savemat(filename, my_dict)


# %% load_mat
def load_mat(filename, var_name=None):
    from scipy.io import loadmat
    from pathlib import Path
    filename = Path(filename).as_posix()
    data_struct = loadmat(filename)
    if not var_name:
        var_name = [x for x in data_struct.keys() if not '__' in x][0]
    my_data = data_struct[var_name]
    return my_data

#%%
def read_data(filename, recorder='tobii'):
    """
    Read Eye-Tracking data from disk.

    Parameters
    ----------
    filename : pathlib Path
        Name of the file.
    recorder : str, optional
        Name of the recorder manufacturer. The default is 'tobii'.

    Returns
    -------
    dataset : gaze_ds Class
        A Class incorporating all Eye-Tracking data.

    """
    from mitgaze.gaze_ds import gaze_ds

    import pandas as pd
    from pathlib import Path
    p_filename = Path(filename)
    if not p_filename.exists():
        print('Where is the file??? :/')
        import sys
        sys.exit(0)
    if p_filename.suffix == '.csv':
        df = pd.read_csv(filename, sep=',', low_memory=False)
    elif p_filename.suffix == '.tsv':
        df = pd.read_csv(filename, sep='\t', low_memory=False)
    else:
        print('Input file should be a .tsv or .csv file.')
        import sys
        sys.exit(0)
    dataset = gaze_ds(df)
    if recorder == 'tobii':
        dataset.read_tobii(if_verbose=True)
    return dataset

#%%
def read_df(df, recorder='tobii'):
    """
    Convert pd.Dataframe which includes recording data to a gaze_ds object.

    Parameters
    ----------
    df : pandas Dataframe
        dataframe inclduing all reocrding data.
    recorder : str, optional
        Type of the Eye-tracker. The default is 'tobii'.

    Returns
    -------
    dataset : gaze_ds object
        An object inclduing all relevant data.

    """
    from mitgaze.gaze_ds import gaze_ds

    dataset = gaze_ds(df)
    if recorder == 'tobii':
        dataset.read_tobii(if_verbose=True)
    else:
        print('unknown recorder!')
        dataset = None
    return dataset

#------------------------------------------------------------------------
#%%
def df_columns(df, if_len = False, loc=20):
    cols = list(df.columns)
    cols.sort()
    if if_len:
        for x in cols:
            try:
                print('location ',loc, ': ', x , '-->', len(df.loc[loc, x]))
            except TypeError:
                print('location ',loc, ': ', x)
    else:
        for x in cols:
            print('\''+x+'\',')


def parse_df_par(filename, dir_save, col_2_del=None):
    import glob
    import pandas as pd
    from pathlib import Path
    #
    dir_save = Path(dir_save)
    dir_save.mkdir(parents=True, exist_ok=True)
    #
    my_df = pd.read_csv(filename, sep='\t', low_memory=False)
    if len(list(my_df.columns)) == 1:
        print('Not a proper TSV file. Trying CSV.')
        print(filename)
        my_df = pd.read_csv(filename, low_memory=False)
    if len(list(my_df.columns)) == 1:
        import sys
        print('data file not a CSV or TSV with required columns')
        print(filename)
        sys.exit(0)

    df_cols = list(my_df.columns)
    if not col_2_del is None:
        for col in col_2_del:
            if col in df_cols:
                del my_df[col]

    ind_media_nan = my_df['Presented Media name'].isnull()
    my_df_no_nan = my_df[~ind_media_nan]

    # project_names = my_df_no_nan.loc[:,'Project name'].unique().tolist()
    participant_names = my_df_no_nan.loc[:,'Participant name'].unique().tolist()
    # timeline_names = my_df_no_nan.loc[:,'Timeline name'].unique().tolist()
    list_files_saved = []
    for pa_name in participant_names:
        # my_cond = ((my_df['Project name']==pr_name) &
        #            (my_df['Participant name']==pa_name) &
        #            (my_df['Timeline name']==tl_name))
        my_cond = my_df_no_nan['Participant name']==pa_name
        df_2_save = my_df_no_nan[my_cond]
        file2save = 'Par_' + pa_name + '.tsv'
        file2save = Path(dir_save, file2save)
        df_2_save.to_csv(file2save, sep='\t', index=False)
        list_files_saved.append(file2save)
        # save summary df for each df_par
        df_summary = extract_media_info(df_2_save)
        file_sum = Path(file2save.parent, 'summary', 'summary_' + file2save.stem + '.tsv')
        file_sum.parent.mkdir(exist_ok=True, parents=True)
        df_summary.to_csv(file_sum, sep='\t', index=False)
    return list_files_saved

def extract_media_info(df_in):
#%%
    import pandas as pd
    import numpy as np

    columns = [
        'Average calibration accuracy (degrees)',
        'Average calibration accuracy (mm)',
        'Average calibration accuracy (pixels)',
        'Average calibration precision (degrees)',
        'Average calibration precision (mm)',
        'Average calibration precision (pixels)',
        'Event',
        'Event value',
        'Export date',
        'Original Media height',
        'Original Media width',
        'Participant name',
        'Presented Media height',
        'Presented Media name',
        'Presented Media position X (RCSpx)',
        'Presented Media position Y (RCSpx)',
        'Presented Media width',
        'Presented Stimulus name',
        'Project name',
        'Recording Fixation filter name',
        'Recording date',
        'Recording duration',
        'Recording monitor latency',
        'Recording name',
        'Recording resolution height',
        'Recording resolution width',
        'Recording software version',
        'Recording start time',
        'Timeline name',
        ]
    df_out = pd.DataFrame(columns=columns, index=[0])
    for col in columns:
        my_list = list(df_in[col].unique())
        if len(my_list) == 1:
            df_out.at[0, col] = my_list[0]
        else:
            df_out.at[0, col] = my_list
    df_out.loc[0, 'Eyetracker timestamp'] = np.max(df_in['Eyetracker timestamp'])
    return df_out
#%%
def parse_df_media(dir_source, dir_save):
    import pandas as pd
    from pathlib import Path
    #
    dir_save = Path(dir_save)
    dir_save.mkdir(parents=True, exist_ok=True)
    #
    dict_media_df = {}
    all_files = list(dir_source.glob('*.tsv'))
    for filename in all_files:
        print(filename)
        my_df = pd.read_csv(filename, sep='\t', low_memory=False, error_bad_lines=False)
        list_media = list(my_df['Presented Media name'].unique())
        for media in list_media:
            df_media = my_df[my_df['Presented Media name'] == media]
            dict_keys = list(dict_media_df.keys())
            if media in dict_keys:
                df_current = dict_media_df[media]
                df_current = df_current.append(df_media, ignore_index=True)
                dict_media_df[media] = df_current
            else:
                dict_media_df[media] = df_media
            # save summary
            df_summary = extract_media_info(df_media)
            par_name = list(df_media['Participant name'].unique())[0]
            media_stem = Path(media).stem
            file_sum = Path(dir_save, 'summary_par_media', par_name + '_' + media_stem + '.tsv')
            file_sum.parent.mkdir(exist_ok=True, parents=True)
            df_summary.to_csv(file_sum, sep='\t', index=False)

    list_files_saved = []
    for media in dict_media_df.keys():
        df_2_save = dict_media_df[media]
        media_stem = Path(media).stem
        filesave = Path(dir_save, 'Media_' + media_stem + '.tsv')
        df_2_save.to_csv(filesave, sep='\t', index=False)
        list_files_saved.append(filesave)
        # save summary df for each df_par
        df_summary = extract_media_info(df_2_save)
        file_sum = Path(filesave.parent, 'summary', 'summary_' + filesave.stem + '.tsv')
        file_sum.parent.mkdir(exist_ok=True, parents=True)
        df_summary.to_csv(file_sum, sep='\t', index=False)

    return list_files_saved


def df_rec_remove_redundancies(df_rec):
    ''' Checks 'Recording name', if there are more than one,
            only chooses the one with the most rows.
    This might happen if there have been two recordings for one timeline in one project
    which usually happens when calibration takes too long and a new recording starts.
    This can be easily avoided if files are saved properly.

    :param df_rec: a dataframe created from a csv/tsv file.
                    It is assumed it has recording data for one subject, one timeline
                    one project only. If more than one subject, it chooses only one
    :type df_rec: Pandas dataframe

    :raises Error: stops execution if more than one subject
    :raises Error: stops execution if more than one timeline

    :return: df_rec, with only one subject name
    :rtype: Pandas dataframe
    '''
    Recording_name_list = df_rec.loc[:,'Recording name'].unique().tolist()
    Participant_name_list = df_rec.loc[:,'Participant name'].unique().tolist()
    Timeline_name_list = df_rec.loc[:,'Timeline name'].unique().tolist()

    if len(Participant_name_list)>1:
        print('More than one particiapnt in this dataframe. Might casue issues!')
    if len(Timeline_name_list)>1:
        print('More than one Timeline in this dataframe. Might casue issues!')

    MAX_len = 0
    if len(Recording_name_list)==1:
        print('dataframe is returned unchnaged')
        return df_rec
    else:
        for recording in Recording_name_list:
            df_temp = df_rec.loc[df_rec['Recording name']==recording]
            list_media = df_temp.loc[:,'Presented Media name'].unique()
            len_media_list = len(list_media)
            if len_media_list > MAX_len:
                MAX_len = len_media_list
                chosen_recording = recording
        df_rec = df_rec.loc[df_rec['Recording name']==chosen_recording]
        print('There were', len(Recording_name_list), 'recordings.')
        len_media = len(list(df_rec['Presented Media name'].unique()))
        print(chosen_recording, ' was chosen, which indluds: ')
        print('--> ', len_media , 'media')
        print('--> ', len(df_rec), 'rows')
        return df_rec


def df_rec_curate_media_names(df_rec, remove_media_that_includs=['1920x']):
#    import numpy as np

#    list_media_original = list(df_rec.loc[:,'Presented Media name'].unique())


    df_rec_new = df_rec.dropna(subset=['Presented Media name'])

#    list_media = [x for x in list_media_original if str(x) != 'nan']

    for str_2_remove in remove_media_that_includs:
        df_rec_new = \
            df_rec_new[~df_rec_new['Presented Media name'].str.contains(str_2_remove)]
#        list_media = [x for x in list_media if (not str_2_remove in x)]

    list_media = list(df_rec_new.loc[:,'Presented Media name'].unique())

    for idx, media in enumerate(list_media):
        imagename_jpg = media.replace(" ", "")
        imagename_jpg = imagename_jpg.replace("-", "_")
        imagename_jpg = imagename_jpg.replace("&", "_")
        imagename_jpg = imagename_jpg.replace("#", "_")
        imagename_jpg = imagename_jpg.replace("#", "_")
        imagename_jpg = imagename_jpg.replace("#", "_")
        imagename_jpg = imagename_jpg.replace(",", "_")
        if not imagename_jpg==media:
            print(media, ' --->', imagename_jpg)
            list_media[idx] = imagename_jpg
            df_rec_new.loc[df_rec_new['Presented Media name']==media, 'Presented Media name'] =\
                    imagename_jpg
    return df_rec_new

def check_if_media_files_available(list_media, dir_source):
    from pathlib import Path
    print('*******************')
    print('List of media includes', len(list_media), ' files ...')
    print('Missing files in ', dir_source)
    if_changes = False
    list_full_file = []
    for filename in list_media:
        filename_full = Path(dir_source, filename)
        if not filename_full.is_file():
            print(filename)
            if_changes = True
        list_full_file.append(filename_full)
    if not if_changes:
        print('----> There was none. :)')
    return list_full_file

def codify_subject_names(df_rec, participants_codes):
    secret_names = list(df_rec.loc[:,'Participant name'].unique())
    for old_name in secret_names:
        try:
            new_name = participants_codes[old_name]
            df_rec.loc[df_rec['Participant name']==old_name, 'Participant name'] = new_name
        except:
            pass
    return df_rec

def print_df_rec_summary(df_rec):
    Recording_name_list = df_rec.loc[:,'Recording name'].unique().tolist()
    Participant_name_list = df_rec.loc[:,'Participant name'].unique().tolist()
    Timeline_name_list = df_rec.loc[:,'Timeline name'].unique().tolist()

    print('*************************************************')
    print('dataframe includes',len(df_rec), 'rows')
    print('Recording:',Recording_name_list)
    print('Participant:',Participant_name_list)
    print('Timeline:',Timeline_name_list)
    print('*************************************************')

