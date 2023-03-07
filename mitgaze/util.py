#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:58:19 2020.

@author: vbokharaie
"""

# pylint: disable=line-too-long
# pylint: disable=wrong-import-position
# from mitgaze import _Lib

# __methods__ = []  # self is a DataStore
# register_method = _Lib.register_method(__methods__)

# %% gaze_count_file
def gaze_count_file(filename, dir_save, Nx, Ny):
    """
    Run segment_gaze_points for each media-participant x-y data in the file.

    Parameters
    ----------
    filename : str or pathlib.Path
        TSV/CSV file including x-y gaze data.
    dir_save : str or pathlib.Path
        where to save.
    Nx : int
        number of segments along screen width.
    Ny : int
        number of segments along screen height.

    Returns
    -------
    None.

    """
    import pandas as pd
    import numpy as np
    from pathlib import Path

    from mitgaze.util import remove_nan
    from mitgaze.util import linsegments
    from mitgaze.util import segment_gaze_points
    from mitgaze.file_io import save_mat

    filename = Path(filename)
    dir_save = Path(dir_save)
    dir_save.mkdir(parents=True, exist_ok=True)

    print('Loading file ', filename)
    df_all = pd.read_csv(filename, low_memory=False, sep='\t')
    print('DONE!')
    # print('Results would be saved in ', dir_save)

    list_media = list(df_all['Presented Media name'].unique())
    list_par = list(df_all['Participant name'].unique())

    for media in list_media:
        print('Media: ', media)
        df_media = df_all[df_all['Presented Media name'] == media]
        for par in list_par:
            # print('Par: ', par)
            df_media_par = df_media[df_media['Participant name'] == par]
            SCREEN_H = df_media_par['Recording resolution height'].unique()[0]
            SCREEN_W = df_media_par['Recording resolution width'].unique()[0]
            x = np.array(df_media_par['Gaze point X'])
            y = np.array(df_media_par['Gaze point Y'])
            x, y = remove_nan(x, y)
            x_seg = linsegments(SCREEN_W, Nx)
            y_seg = linsegments(SCREEN_H, Ny)
            gaze_seg_count = segment_gaze_points(x, y, x_seg, y_seg)

            # save
            # print('saving ...')
            filename_save_base = par + '_' + media + '_GAZE_SEGMENTS'
            filename_np = Path(dir_save, filename_save_base)
            np.save(filename_np, gaze_seg_count)

            filename_mat = Path(dir_save, filename_save_base + '.mat')
            save_mat(filename_mat, gaze_seg_count)

# %% linsegments
def linsegments(max_val, N, min_val=0):
    """
    Wrapper for numpy.linspace.

    Parameters
    ----------
    max_val : float
        maximum point in range.
    N : int
        number of segments.
    min_val : float, optional
        minimum value in the range. The default is 0.

    Returns
    -------
    x_seg : numpy.array
        an array with values marking the segment coordinates.

    """
    import numpy as np
    x_seg = np.linspace(min_val, max_val, N+1)
    return x_seg

# %% segment_gaze_points
def segment_gaze_points(x, y, x_seg, y_seg):
    """
    Count number of gaze points in each segment.

    Parameters
    ----------
    x : numpy.array
        array of x coordinates.
    y : numpy.array
        array of y coordinates.
    x_seg : numpy.array
        values marking x axis segments.
    y_seg : numpy.array
        values marking y axis segments..

    Returns
    -------
    array_seg_count : numpy.array
        Count of (x,y) pairs in each region.

    """
    import numpy as np

    xx_seg = list(zip(x_seg[:-1], x_seg[1:]))
    yy_seg = list(zip(y_seg[:-1], y_seg[1:]))
    array_seg_count = np.zeros((len(xx_seg), len(yy_seg)))
    for idx, xx in enumerate(xx_seg):
        for idy, yy in enumerate(yy_seg):
            points_in_region = [(a, b) for (a, b) in zip(x, y) if
                                ((a>=xx[0] and a < xx[1]) and (b>=yy[0] and b<yy[1]))]
            array_seg_count[idx, idy] = len(points_in_region)

    return array_seg_count

# %% remove_nan
def remove_nan(x_with_nan, y_with_nan):
    """
    Remove nan from x/y data.

    Parameters
    ----------
    x_with_nan : numpy array
        x data.
    y_with_nan : numpy array
        y data.

    Returns
    -------
    x : numpy array
        x data without nan.
    y : numpy array
        y data without nan.

    """
    import numpy as np
    x_ind = list(np.where(np.isnan(x_with_nan))[0])
    y_ind = list(np.where(np.isnan(y_with_nan))[0])
    nan_ind = list(set(x_ind+y_ind))
    x = np.delete(x_with_nan, nan_ind)
    y = np.delete(y_with_nan, nan_ind)
    return x, y

# %% df_columns
def df_columns(df, if_len = False, loc=20):
    """
    Print the columns of Pandas df, sorted alphabetically.

    Parameters
    ----------
    df : pandas Datafame
        the df.
    if_len : Bool, optional
        If print length of df. The default is False.
    loc : int, optional
        start th eprint from a certain location in the list of column names. The default is 20.

    Returns
    -------
    None.

    """
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

#################################################################################
# %% rescale_fill_save_folder
# read all the media files in a folder, finds them in the csv file
# then reads presented dimentions and screen dimensions, then saved all filled in images
# in a specified folder
def rescale_fill_save_folder(dir_source_media, tsv_file, dir_save):
    """
    Rescale images.

    Reads all images in a folder, and scale them based on Screen W/H read from tsv file,
    fill surrounding areas of image with black, and then save them.

    Parameters
    ----------
    dir_source_media : pathlib Path or str.
        folder containig media files.
    tsv_file : pathlib Path or str
        filename, output of Eye-Tracking recorder tsv or csv.
    dir_save : pathlib Path or str
        save folder.

    Returns
    -------
    None.

    """
    import numpy as np
    from pathlib import Path
    import pandas as pd

    dir_source_media = Path(dir_source_media)
    tsv_file = Path(tsv_file)
    dir_save = Path(dir_save)

    dir_save.mkdir(parents=True, exist_ok=True)


    list_media_files = list(dir_source_media.rglob("*"))

    df_all = pd.read_csv(tsv_file)

    for media_file in list_media_files:
        media = Path(media_file).name
        df_media = df_all[df_all['Presented Media name']==media]
        try:

            W_screen = np.int16(df_media['Recording resolution width'].unique()[0])
            H_screen = np.int16(df_media['Recording resolution height'].unique()[0])


            W_presented = np.int16(df_media['Presented Media width'].unique()[0])
            H_presented = np.int16(df_media['Presented Media height'].unique()[0])

            rescale_fill_media(media_file,
                               dir_save,
                               W_presented, H_presented,
                               W_screen, H_screen)
            print('Media: ',media, ' ,H = ', H_screen)
        except IndexError:
            print('media name not found in data file: ',media)



# %% rescale_fill_media
# rescale image to presented dimensions, fill with black, save
def rescale_fill_media(media_file,
                       dir_save,
                       W_presented, H_presented,
                       W_screen=1920, H_screen=1200):
    """
    Rescale each image.

    Parameters
    ----------
    media_file : pathlib Path or str
        filename for media.
    dir_save : pathlib Path or str
        save folder.
    W_presented : int
        Width media.
    H_presented : int
        Height of media.
    W_screen : int, optional
        Screen width. The default is 1920.
    H_screen : int, optional
        Screen height. The default is 1200.

    Returns
    -------
    None.

    """
    import cv2
    from pathlib import Path

    media_file = Path(media_file)
    dir_save = Path(dir_save)


    img_current = cv2.imread(media_file, -1) # READS BGR as in (X,Y,3)
    img_current_resized = cv2.resize(img_current,
                                     dsize=(W_presented, H_presented),
                                     interpolation=cv2.INTER_CUBIC)
#    if len(img_current_resized.shape)==3:
#        img_current_resized = img_current_resized[..., ::-1]
    img_current1 = img_fill(img_current_resized,
                            W_screen=W_screen,
                            H_screen=H_screen)
    media = Path(media_file).stem

    file_save = Path(dir_save, media+'_'+str(W_screen)+'x'+str(H_screen)+'.jpg')
    cv2.imwrite(file_save, img_current1)


#%% img_fill
# This is a function to fill an image with np.nan to fit the 1920x1080 screen resolutioin
# just to prevent annoyance of fitting a smaller image into the frame
def img_fill(img, W_screen = 1920, H_screen=1200, fill='zeros'):
    """
    Fill image with solid black such that it fits the screen width and height.

    Parameters
    ----------
    img : numpay array
        image.
    W_screen : int, optional
        Screen width. The default is 1920.
    H_screen : int, optional
        Screen height. The default is 1200.
    fill : text
        how to fill areas around the rescaled image. default, 'zeros', otherwise it'd be 'np.nan'

    Returns
    -------
    img_filled: numpy array
    Filled imge or None if input image dimension is not 2 or 3

    """
    import numpy as np

    if len(img.shape) == 2:
        (H_img, W_img) = np.shape(img)
        if fill == 'zeros':
            img_out = np.zeros((H_screen,W_screen,))
        else:
            img_out = np.empty((H_screen,W_screen,))
            img_out[:] = np.nan
        W_start = np.int16((W_screen-W_img)/2)
        H_start = np.int16((H_screen-H_img)/2)
        img_out[H_start:H_start+H_img, W_start:W_start+W_img,  ] = img
        return img_out
    elif len(img.shape) == 3:
        (H_img, W_img, dummy) = np.shape(img)
        if fill == 'zeros':
            img_out = np.zeros((H_screen, W_screen ,3))
        else:
            img_out = np.empty((H_screen, W_screen ,3))
            img_out[:] = np.nan
        W_start = np.int16((W_screen-W_img)/2)
        H_start = np.int16((H_screen-H_img)/2)
        img_out[H_start:H_start+H_img, W_start:W_start+W_img,:] = img
        img_out = np.uint8(img_out)
        return img_out

# %% parse_tobii_output
def parse_tobii_output(filename, dir_save, overwrite=False):
    """
    Read Tobii Eye-tracker output files.

    Then splits it to subset of project name / participant name / timeline name and saves.

    Parameters
    ----------
    filename : pathlib Path or str
        name of the tsv/csv file.
    dir_save : pathlib Path or str
        dir_save.
    overwrite : Bool, optional
        Overwrite if destinations filename already exists? The default is False.

    Returns
    -------
    list_files_saved : list of pathlib.Path
        list of saved files.

    """
    import glob
    import pandas as pd
    from pathlib import Path

    filename = Path(filename)
    dir_save = Path(dir_save)

    df_temp = pd.read_csv(filename, sep='\t')
    project_names = df_temp.loc[:,'Project name'].unique().tolist()
    participant_names = df_temp.loc[:,'Participant name'].unique().tolist()
    timeline_names = df_temp.loc[:,'Timeline name'].unique().tolist()
    list_files_saved = []
    for pr_name in project_names:
        for pa_name in participant_names:
            for tl_name in timeline_names:
                my_cond = ((df_temp['Project name']==pr_name) &
                           (df_temp['Participant name']==pa_name) &
                           (df_temp['Timeline name']==tl_name))
                df_2_save = df_temp[my_cond]
                file2save = 'PROJECT_'+pr_name+\
                            '_SUB_'+pa_name+\
                            '_TL_'+tl_name+'.csv'
                file2save = Path(dir_save, file2save)
                file_present = glob.glob(file2save)
                if overwrite or (not file_present):
                    df_2_save.to_csv(file2save)
                else:
                    print('Output file already exists, did NOT overwrite:\n',file2save)
                list_files_saved.append(file2save)
    return list_files_saved

# %% divide_img_2_segments
def divide_img_2_segments(img, nw=4, nh=4):
    """
    Segment each image, gemoterically, uniform steps.

    Parameters
    ----------
    img : numpy array.
        The image to be segmented.
    nw : int, optional
        number of segments in width.. The default is 4.
    nh : int, optional
        number of segments in height. The default is 4.

    Returns
    -------
    list_w : list of int
        x corrdinates of segments.
    list_h : list of int
        y corrdinates of segments.

    """
    import numpy as np

    H = img.shape[0]
    W = img.shape[1]
    list_h = [0]
    for cc in np.arange(nh)+1:
        list_h.append(int(H*cc/nh))

    list_w = [0]
    for cc in np.arange(nw)+1:
        list_w.append(int(W*cc/nw))

    return list_w, list_h

# %% divide_images_in_folder_2_segments
def divide_images_in_folder_2_segments(dir_source, dir_save, nw=4, nh=4):
    """
    Read all images in a folder and segment them and then save.

    Parameters
    ----------
    dir_source : pathlib Path or str
        dir_source.
    dir_save : pathlib Path or str
        dir_save.
    nw : int, optional
        number of segments in width. The default is 4.
    nh : int, optional
        number of segments in height. The default is 4.

    Returns
    -------
    None.

    """
    import cv2
    import pandas as pd
    from pathlib import Path
    import numpy as np

    import glob


    list_files1 = [f for f in glob.glob(dir_source + "**/*.jpg", recursive=True)]
    list_files2 = [f for f in glob.glob(dir_source + "**/*.png", recursive=True)]

    list_files = list_files1 + list_files2

    if not list_files:
        print('no files found!')
        return
    df = pd.DataFrame(index = np.arange(len(list_files)),
                      columns = ['media_name', 'list_h', 'list_w'])

    for idx, media_file in enumerate(list_files):
        media_name = Path(media_file).name
        img = cv2.imread(media_file, -1)
        list_w, list_h = divide_img_2_segments(img, nw, nh)

        df.at[idx, 'media_name'] = media_name
        df.at[idx, 'list_w'] = list_w
        df.at[idx, 'list_h'] = list_h
    filesave = 'media_divided_in_'+str(nw)+'x'+str(nh)+'_segments.csv'
    file_save = Path(dir_save, filesave)
    df.to_csv(file_save)
    print('outputs saved in:')
    print(file_save)

# %% describe
#%% describe
def describe(df, media_names=None):
    """
    Summary of an Eye-Tracking recording, per media.

    Parameters
    ----------
    df : Pandas Dataframe
        includes the Eye-tracking data.
    media_names : name os Presented Media to consider, optional
        If not used, all Presented Media in the df would be considered. The default is None.

    Returns
    -------
    df_out : Pandas Dataframe
        Summary of df. Few cells are dict encapsulated as list to make code more straightforward.

    """
    from collections import Counter

    from mitgaze.util import remove_nan

    import numpy as np
    import pandas as pd

    if not media_names:
        media_names = list(df['Presented Media name'].unique())
    # %%
    columns1a = [
        'Presented Media name',
        'Participant name',
        'Project name',
        'Timeline name',
        'Export date',
        ]
    columns1b = [
        # 'Presented Stimulus name',
        'Recording resolution height',
        'Recording resolution width',
        'Presented Media height',
        'Presented Media width',
        'Original Media width',
        'Original Media height',
        'Presented Media position X (RCSpx)',
        'Presented Media position Y (RCSpx)',
        'Recording date',
        'Recording Fixation filter name',
        'Recording duration',
        'Recording monitor latency',
        'Recording name',
        'Recording resolution height',
        'Recording resolution width',
        'Recording software version',
        'Recording start time',
        'Average calibration accuracy (degrees)',
        'Average calibration accuracy (mm)',
        'Average calibration accuracy (pixels)',
        'Average calibration precision (degrees)',
        'Average calibration precision (mm)',
        'Average calibration precision (pixels)',
                ]

    columns2a = [
        'Event',
        'Event value',
        'Eye movement type',
        ]

    columns2b = [
        'Eye movement type index',
        'Validity left',
        'Validity right',
        ]
    df[columns2b] = df[columns2b].astype(pd.Int64Dtype())

    columns3 = [
        'len_xy',
        'len_xy_not_nan',
        'len_xy_nan',
        'x_mean',
        'y_mean',
        'x_sd',
        'y_sd',
        'Eyetracker timestamp',
        'Recording timestamp',
        ]

    columns4 = [
        'Fixation point',
        'Fixation point (MCSnorm)',
        ]

    columns5 = [
        'Fs_media',
        'Recording timestamp start (s)',
        'Recording timestamp duration (s)',
        'Eyetracker timestamp start (s)',
        'Eyetracker timestamp duration (s)',
            ]
    column6 = [
        'pupil_left_len',
        'pupil_left_len_nan',
        'pupil_left_max',
        'pupil_left_min',
        'pupil_left_mean',
        'pupil_left_sd',
        'pupil_right_len',
        'pupil_right_len_nan',
        'pupil_right_max',
        'pupil_right_min',
        'pupil_right_mean',
        'pupil_right_sd',
        ]

    columns = columns1a + columns5 + columns1b + columns2a + columns2b + \
                columns3 + columns4 + column6

    df_out = pd.DataFrame(index = np.arange(len(media_names)), columns=columns)
    df_out = df_out.astype(object)
    for idx, media in enumerate(media_names):
        df_media = df.loc[df['Presented Media name'] == media]

        W_screen = np.int16(df_media['Recording resolution width'].unique()[0])
        H_screen = np.int16(df_media['Recording resolution height'].unique()[0])

        x_with_nan = np.array(df_media['Gaze point X'])
        y_with_nan = np.array(df_media['Gaze point Y'])
        x_with_nan = [np.nan if x<0 else x for x in x_with_nan]
        x_with_nan = [np.nan if x>W_screen else x for x in x_with_nan]
        y_with_nan = [np.nan if x<0 else x for x in y_with_nan]
        y_with_nan = [np.nan if x>H_screen else x for x in y_with_nan]
        x, y = remove_nan(x_with_nan, y_with_nan)


        df_out.at[idx, 'len_xy'] = len(x_with_nan)
        df_out.at[idx, 'len_xy_not_nan'] = len(x)
        df_out.at[idx, 'len_xy_nan'] = len(x_with_nan) - len(x)
        df_out.at[idx, 'x_mean'] = np.mean(x)
        df_out.at[idx, 'y_mean'] = np.mean(y)
        df_out.at[idx, 'x_sd'] = np.std(x)
        df_out.at[idx, 'y_sd'] = np.std(y)

        for col in columns1a+columns1b:
            val = df_media[col].unique().tolist()
            if len(val)==1: val = val[0]
            df_out.at[idx, col] = val

        for col in columns2a+columns2b:
            df_out.at[idx, col] = [dict(Counter(df_media[col]))]


        ind_s = df_media.index[0]
        ind_e = df_media.index[-1]
        t_s = df_media.loc[ind_s, 'Recording timestamp']
        t_e = df_media.loc[ind_e, 'Recording timestamp']

        df_out.at[idx, 'Recording timestamp start (s)'] = t_s/1000
        df_out.at[idx, 'Recording timestamp duration (s)'] = (t_e-t_s)/1000
        df_out.at[idx, 'Fs_media'] = 1000/((t_e-t_s)/(ind_e-ind_s))

        Fix_X = df_media['Fixation point X'].tolist()
        Fix_X_MCS = df_media['Fixation point X (MCSnorm)'].tolist()

        Fix_Y = df_media['Fixation point Y'].tolist()
        Fix_Y_MCS = df_media['Fixation point Y (MCSnorm)'].tolist()

        Fix = []
        Fix_MCS = []
        for cc in np.arange(len(Fix_X)):
            if not np.isnan(Fix_X[cc]) and not np.isnan(Fix_Y[cc]):
                Fix.append((Fix_X[cc], Fix_Y[cc]))
                Fix_MCS.append((Fix_X_MCS[cc], Fix_Y_MCS[cc]))

        df_out.at[idx, 'Fixation point']= [dict(Counter(Fix))]
        df_out.at[idx, 'Fixation point (MCSnorm)']= [dict(Counter(Fix_MCS))]


        ### pupil
        pupil_left =  df_media['Pupil diameter left']
        pupil_right = df_media['Pupil diameter right']

        df_out.at[idx, 'pupil_left_max'] = pupil_left.max()
        df_out.at[idx, 'pupil_left_min'] = pupil_left.min()
        df_out.at[idx, 'pupil_left_mean'] = pupil_left.mean()
        df_out.at[idx, 'pupil_left_sd'] = pupil_left.std()
        df_out.at[idx, 'pupil_left_len'] = len(pupil_left)
        df_out.at[idx, 'pupil_left_len_nan'] = pupil_left.isna().sum()

        df_out.at[idx, 'pupil_right_max'] = pupil_right.max()
        df_out.at[idx, 'pupil_right_min'] = pupil_right.min()
        df_out.at[idx, 'pupil_right_mean'] = pupil_right.mean()
        df_out.at[idx, 'pupil_right_sd'] = pupil_right.std()
        df_out.at[idx, 'pupil_right_len'] = len(pupil_right)
        df_out.at[idx, 'pupil_right_len_nan'] = pupil_right.isna().sum()

    return df_out

###############################################################################
### New utility files 2021
#%% used to make a list of (x, y) values for discontinusous line plots
def make_plot_list_discontinuous(list_x, list_y, list_values, color='g'):
    list_out = []
    for x, y, v in zip(list_x, list_y, list_values):
        list_out.append((x, y))
        list_out.append((v, v))
        list_out.append(color)
    return list_out


#%% used to make a list of (x, y) values for continusous lien plots
def make_plot_list_continuous(list_x, list_y, list_values):
    list_x_axis = []
    list_y_axis = []
    for x, y, v in zip(list_x, list_y, list_values):
        list_x_axis.append(x)
        list_x_axis.append(y)
        list_y_axis.append(v)
        list_y_axis.append(v)
    return list_x_axis, list_y_axis


# %% plot fixation scores points for each media, per participant
def plot_df_media(media, df_media, dir_save, col_score, col_score_weighted, y_max=1.0):
    from mitgaze.util import make_plot_list_discontinuous, make_plot_list_continuous
    import matplotlib.pyplot as plt
    from pathlib import Path
    import numpy as np
    list_par = list(df_media['Participant name'].unique())
    list_par.sort()
    fig1, ax1 = plt.subplots(nrows=len(list_par), figsize=(25, 2*len(list_par)))
    fig1.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
    fig1.suptitle('Discontinous Line plot for scores of ' + col_score)
    fig2, ax2 = plt.subplots(nrows=len(list_par), figsize=(25, 2*len(list_par)))
    fig2.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
    fig2.suptitle('Continous Line plot for scores of' + col_score)
    fig3, ax3 = plt.subplots(nrows=len(list_par), figsize=(25, 2*len(list_par)))
    fig3.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
    fig3.suptitle('Bar plot for WEIGHTED scores' + col_score)
    for ind, par in enumerate(list_par):
        df_media_par = df_media[df_media['Participant name']==par]
        df_media_par = df_media_par.sort_values('Start recording Time corrected')
        # plot_fixations(df_media_par)
        list_start_times = list(df_media_par['Start recording Time corrected']/1000)
        list_end_times = list(df_media_par['End recording Time corrected']/1000)
        list_scores = list(df_media_par[col_score])
        if len(list_scores) == 0:
            raise Exception('column does not exist')
        list_scores_weighted = list(df_media_par[col_score_weighted])
        #
        list_plot = make_plot_list_discontinuous(list_start_times, list_end_times, list_scores)
        ax1[ind].plot(*list_plot)
        ax1[ind].set_ylim([0,y_max])
        ax1[ind].set_ylabel(par)
        #
        list_x, list_y = make_plot_list_continuous(list_start_times, list_end_times, list_scores)
        ax2[ind].plot(list_x, list_y)
        ax2[ind].set_ylim([0,y_max])
        ax2[ind].set_ylabel(par)
        #
        ax3[ind].bar(np.arange(len(list_scores_weighted)), list_scores_weighted)
        ax3[ind].set_ylabel(par)
    # save
    dir_save1 = Path(dir_save, 'score_line_discontinuous')
    dir_save1.mkdir(exist_ok=True, parents=True)
    filesave1 = Path(dir_save1, media+'.png')
    fig1.savefig(filesave1)
    #
    dir_save2 = Path(dir_save, 'line_continuous')
    dir_save2.mkdir(exist_ok=True, parents=True)
    filesave2 = Path(dir_save2, media+'.png')
    fig2.savefig(filesave2)
    #
    dir_save3 = Path(dir_save, 'bar_plot')
    dir_save3.mkdir(exist_ok=True, parents=True)
    filesave3 = Path(dir_save3, media+'.png')
    fig3.savefig(filesave3)
    #
    plt.close('all')


# %% plot lines plots for trajectory of fixation scores with and without weights
def plot_fixation_scores(df_in, col_score, col_score_weighted, dir_save_main, num_cores, y_max=1.0):
    from pathlib import Path
    from joblib import Parallel, delayed

    from mitgaze.util import plot_df_media
    dir_save = Path(dir_save_main, col_score)
    dir_save.mkdir(parents=True, exist_ok=True)
    print('saving in', dir_save)
    list_media = list(set(list(df_in['Presented Media name'])))
    Parallel(n_jobs=num_cores)(delayed(plot_df_media)(media,
                                        df_in[df_in['Presented Media name']==media],
                                        dir_save,
                                        col_score, col_score_weighted,
                                        y_max)
                                        for media in list_media)

# %% count number of each elemnt in the list
def list_count(my_list):
    my_set = set(my_list)
    my_dict = {}
    for x in my_set:
        my_dict[x] = my_list.count(x)
    return my_dict


# %% calculate scores for each saliency feature
def calc_score(dir_source_m, df_fixation, col='luminance', L0=None):
    import matplotlib.pyplot as plt
    from pathlib import Path
    import numpy as np

    df_fixation_score = df_fixation.copy()
    df_fixation_score_norm = df_fixation.copy()

    list_D = [20, 40, 80, 120, 160, 200]
    for D in list_D:
        if col == 'luminance':
            dir_source = Path(dir_source_m, 'FoV_' + str(D) + '_pixels')
        elif col == 'luminance':
            dir_source = Path(dir_source_m, 'FoV_' + str(D) + '_pixels_L0_' + str(L0))
        elif col == 'hsv':
            dir_source = Path(dir_source_m, 'FoV_' + str(D) + '_pixels_L0_' + str(L0))

        list_files_c = list(dir_source.glob('*'))
        dict_media_c = {}
        for filename in list_files_c:
            img = plt.imread(filename, format='jpeg')
            dict_media_c[filename.name] = img

        for ind in df_fixation_score.index:
            med_name = df_fixation_score.loc[ind, 'media']
#            par_name = df_fixation_score.loc[ind, 'par']
            dict_fix = df_fixation_score.loc[ind, 'dict_fixation']
            dict_fix_norm = df_fixation_score.loc[ind, 'dict_fixation_values_normalised']

            try:
                img = dict_media_c[med_name]
            except KeyError:
                print(med_name)
                continue
            # score
            score_fix = 0.0
            for point in dict_fix.keys():
                try:
                    score = dict_fix[point] * img[int(point[0]), int(point[1])] / 255.0
                except IndexError:
                    pass
                score_fix = np.nansum([score_fix, score])
            df_fixation_score.loc[ind, 'score_' + col + '_D_' + str(D)] = score_fix
            # score normalised
            score_fix_n = 0
            for point in dict_fix_norm.keys():
                try:
                    score = dict_fix_norm[point] * img[int(point[0]), int(point[1])] / 255.0
                except IndexError:
                    pass
                score_fix_n = np.nansum([score_fix_n, score])
            df_fixation_score_norm.loc[ind, 'norm_score_' + col + '_D_' + str(D)] = score_fix_n
    return df_fixation_score, df_fixation_score_norm


# %% calculate score per row of df for each saliency feature
def score_per_row(ind, x, y, med_name, len_fixation, dict_salience):
    import numpy as np
    dict_out = {}
    dict_out['ind'] = ind
    dict_out['x'] = x
    dict_out['y'] = y
    dict_out['med_name'] = med_name
    dict_out['len_fixation'] = len_fixation
    dict_out['score'] = np.nan
    dict_out['score_weighted'] = np.nan
    if x >= 1920 or x < 0:
        return dict_out
    if y >= 1200 or y < 0:
        return dict_out
    try:
        img = dict_salience[med_name]
    except KeyError:
        return dict_out
    dict_out['score'] = img[y, x] / 255.0
    dict_out['score_weighted'] = len_fixation * img[y, x] / 255.0

    return dict_out


# %% summary of columns in Tobii output TSV/CSV file related to gaze behaviour
def df_saliency_summary(df_in):
    dict_unique = {}
    dict_unique_len = {}
    col_2_use = [x for x in df_in.columns if not('Gaze' in x or 'timestamp' in x)]
    df = df_in[col_2_use]
    for col in df.columns:
        list_uni = list(df[col].unique())
        len_list_uni = len(list_uni)
        dict_unique[col] = list_uni
        dict_unique_len[col] = len_list_uni

    return dict_unique, dict_unique_len


# %% a summary list of fixatin points and the amount of time particpant has epnt at each
def func_par_fixation_data(df_par_media):
    import numpy as np
    from mitgaze.util import list_count
    H_screen = int(list(df_par_media['Recording resolution height'])[0])
    media_name = list(df_par_media['Presented Media name'].unique())[0]
    par_name = list(df_par_media['Participant name'].unique())[0]
    #
    df_par_media = df_par_media[df_par_media['Gaze point X'].notna()]
    df_par_media = df_par_media[df_par_media['Gaze point Y'].notna()]

    df_par_media = df_par_media[df_par_media['Fixation point X'].notna()]
    df_par_media = df_par_media[df_par_media['Fixation point Y'].notna()]
    list_X = list(df_par_media['Fixation point X'])
    list_Y = list(df_par_media['Fixation point Y'])
    if H_screen == 1080:
        list_Y = [int(x*1200/1080) for x in list_Y]
        par_name = par_name + '_1080'

    list_fixation_XY = list(zip(list_X, list_Y))
    dict_fixation_XY = list_count(list_fixation_XY)
    sum_val = np.nansum(list(dict_fixation_XY.values()))
    dict_fixation_XY_norm = {}
    for key in dict_fixation_XY.keys():
        dict_fixation_XY_norm[key] = dict_fixation_XY[key]/sum_val

    return   dict_fixation_XY, dict_fixation_XY_norm, media_name, par_name


####################################################################################
### for contrast and luminance calculations

# %% a cricular mask with a wieght monotinically decreasing from the centre
def create_circular_mask(h, w=None, center=None, radius=None):
    """Calculate a weighted mask as per the folloiwng paper.
    https://www.sciencedirect.com/science/article/pii/S0042698905005559

    Code partically adopted from this response in stackoverflow:
        https://stackoverflow.com/a/44874588/1870969
    """

    import numpy as np

    if w is None:
        w = h
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    # mask True for inside the circle, False for outside
    mask = dist_from_center <= radius
    # Frazor weight as per the paper. Wieght value indside the circle, np.nan outside the circle
    frazor_weight = 0.5 * (np.cos(np.pi/radius*dist_from_center) + 1)
    frazor_weight[dist_from_center>radius] = np.nan

    return mask, frazor_weight


# %% weighted contrast with the weight given by create_circular_mask
def weighted_contrast(img_subset, mask_weights, L0):
    """Calculate RMS contrast.

    as per eq (3) in Frazor et al, 2005
    https://www.sciencedirect.com/science/article/pii/S0042698905005559

    Note on L0 from ibid:
        L0 is a “dark light” parameter, chosen to be 7 td (1 cd/m2 assuming a 3 mm pupil),
        based on human (photopic) intensity discrimination data (e.g., Hood & Finkelstein, 1986).
        This dark light parameter takes into account the reduction in visual sensitivity at low luminance,
        which is due (presumably) to spontaneous neural activity and other sources of internal noise.
        As it turned out, using this modified measure had very little effect on the results of
        the data analyses and had no impact on the global trends.
    """

    import numpy as np

    if not img_subset.shape == mask_weights.shape:
        raise Exception("Shape mismatch in weighted_contrast function.")
    mask_nan = np.ones(img_subset.shape)
    mask_nan[np.isnan(img_subset)]=0

    W_SUM = np.nansum(mask_nan * mask_weights)
    if W_SUM == 0:
        return 0
    L_SUM = np.nansum(img_subset * mask_weights)

    signa_term = np.nansum(np.square(mask_weights * img_subset))

    contrast = np.sqrt(signa_term / ((L_SUM + L0) ** 2) * W_SUM)

    return contrast


# %% pad an image with np.nan, so saliency features with a circular mask can be caluclated easier
def pad_with_nan(img, R):
    """Add a band of width R around an image filled with np.nan."""
    import numpy as np

    H = img.shape[0]
    W = img.shape[1]
    img_out = np.empty((H + 2* R, W + 2* R))
    img_out[:] = np.nan

    img_out[R:R+H, R:R+W] = img

    return img_out


# %% calculate contrast for multiple media, function can be sued in a parallel loop
def func_par_contrast(filename, R, mask_weights, dir_save, L0=1):
    from PIL import Image
    import numpy as np
    from pathlib import Path
    import matplotlib.pyplot as plt
#            img_Image=Image.open(filename).convert('L')
    img=np.load(filename)
    H = img.shape[0]
    W = img.shape[1]
    img_pad = pad_with_nan(img, R)
    img_contrast = np.zeros(img.shape)
    for x in np.arange(H)+R:
        for y in np.arange(W)+R:
            img_subset = img_pad[x-R:x+R, y-R:y+R]
            img_contrast[x - R, y - R] = weighted_contrast(img_subset, mask_weights, L0)
    img_contrast[np.isnan(img_contrast)] = 0
    img_contrast_sq = np.square(img_contrast)
    img_contrast_255 = np.uint8(img_contrast_sq * 255 / np.nanmax(img_contrast_sq.flatten()))
    plt.imshow(img_contrast_255, cmap='gray')
    filesave = filename.stem
    filesave = Path(dir_save, filesave + '.jpg')
    Image_save = Image.fromarray(img_contrast_255.astype(np.uint8))
    Image_save.save(filesave)
    print(filesave.name)


# %% weighted luminance for an image with the weight given by create_circular_mask
def weighted_luminance(img_subset, mask_weights):
    """Calculate Luminance.

    as per eq (3) in Frazor et al, 2005
    https://www.sciencedirect.com/science/article/pii/S0042698905005559
    """
    import numpy as np

    if not img_subset.shape == mask_weights.shape:
        raise Exception("Shape mismatch in weighted_contrast function.")

    W_SUM = np.nansum(mask_weights)
    L_SUM = np.nansum(img_subset * mask_weights)
    weighted_L = L_SUM / W_SUM

    return weighted_L


# %% run weighted_luminance for multiple media, can be used in parallel loop
def func_par_luminance(filename, R, mask_weights, dir_save):
    from PIL import Image
    import numpy as np
    from pathlib import Path
    img_Image= np.load(filename)
    img=np.array(img_Image)
    H = img.shape[0]
    W = img.shape[1]
    img_pad = pad_with_nan(img, R)
    img_L = np.zeros(img.shape)
    W_SUM = np.nansum(mask_weights)
    for x in np.arange(H)+R:
        for y in np.arange(W)+R:
            img_subset = img_pad[x-R:x+R, y-R:y+R]
            L_SUM = np.nansum(img_subset * mask_weights)
            img_L[x - R, y - R] = L_SUM / W_SUM
    filesave = filename.stem
    filesave = Path(dir_save, filesave + '.jpg')
    Image_save = Image.fromarray(img_L.astype(np.uint8))
    Image_save.save(filesave)
