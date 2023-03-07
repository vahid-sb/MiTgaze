#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 11:03:09 2019.

@author: vbokharaie

Includes function to plot media and gaze data. 
"""


# %% plot_basics_parsed_media
def plot_basics_parsed_media(filename,
                             dir_save_main,
                             dir_source_image_1200,
                             dir_source_image_1080=None,
                             gridsize_x=40,
                             gridsize_y=20,
                             if_adaptive_gridsize=False,
                             if_gaze_plots=True,):
    """
    Get an Eye-Tracking Recording tsv/csv file as input, plot gaze_basic, gaze_order and heatmap.

    Dataframe should include only one media name. If not, only the first one is considered.
    If you have files with more than one media name in them, parse them using mitgaze.file_io.parse_df_media

    Function saves the results in two seperate folders, one under the Media Name folder, other arranged
    based on participant names.

    Parameters
    ----------
    filename : str or pathlib Path
        a tsv or csv file inclduing the Eye-Tarcking Recording parsed per media..
    dir_save_main : str or pathlib Path
        Where to save.
    dir_source_image_1200 : str or pathlib Pathstr or pathlib Path
        Where are media files (jpg or png). If recordings are done in different screens, use both parameter.
    dir_source_image_1080 : str or pathlib Pathstr or pathlib Path, (Optional)
        If media recordings are done in two screens with two resouloutions, use this extra parameter.
    gridsize_x : int, optional
        gridsize for heatmap in x axis. default is 40.
    gridsize_y : int, optional
        gridsize for heatmap in y axis. default is 20.

    Returns
    -------
    None.

    """
    import pandas as pd
    from pathlib import Path

    filename = Path(filename)
    dir_save_main = Path(dir_save_main)
    dir_source_image_1200 = Path(dir_source_image_1200)
    if not dir_source_image_1080 is None:
        dir_source_image_1080 = Path(dir_source_image_1080)

    print('Loading ', filename)
    df_media = pd.read_csv(filename, low_memory=False, sep='\t')
    if len(list(df_media['Presented Media name'].unique())) != 1:
        import sys
        print('Something wrong. Dataframe should include only one Presented Media name.')
        print('If you have a recording of many Media, first parse the df based on media names.')
        sys.exit(0)
    else:
        media = list(df_media['Presented Media name'].unique())[0]
        df_media = df_media[df_media['Presented Media name'] == media]
    # dir_save = Path(dir_save_main, media_name)
    # dir_save.mkdir(parents=True, exist_ok=True)
    participant_names = list(df_media['Participant name'].unique())
    for par in participant_names:
        df_media_par = df_media[df_media['Participant name'] == par]
        SCREEN_H = df_media_par['Recording resolution height'].unique()[0]
        if SCREEN_H == 1080 and (not dir_source_image_1080 is None):
            dir_source_image = dir_source_image_1080
        else:
            dir_source_image = dir_source_image_1200

        if if_gaze_plots:
            # gaze plot basic
            plot_gaze_df(df_media_par, dir_source_image, dir_save_main, plot_type='GAZE_BASIC',
                             if_save_media=False)
            # gaze order
            plot_gaze_df(df_media_par, dir_source_image, dir_save_main, plot_type='GAZE_ORDER',
                             if_save_media=False)
        # plot kde and save data files
        xx, yy, z = plot_heatmap_df(df_media_par, dir_source_image,
                                    dir_save_main, if_save_fig=True,
                                    gridsize_x=gridsize_x, gridsize_y=gridsize_y,)



# %% plot_heatmap_df
def plot_heatmap_df(df_media, dir_source_image,
                     dir_save_main, if_save_fig=True,
                     gridsize_x=40, gridsize_y=20,
                     if_adaptive_gridsize=False,):
    """
    Plot heatmaps for all media and gaze info in a pd.Dataframe.

    Parameters
    ----------
    df_media : pandas Dataframe
        Dataframe of Eye-tracking reocrding.
    dir_source_image : str or pathlib Path
        where the media fiels are.
    dir_save_main : str or pathlib Path
        where to save.
    if_save_fig : Bool, optional
        if save the figures to disk. The default is True.
    gridsize_x : int, optional
        gridsize for heatmap in x axis. default is 40.
    gridsize_y : int, optional
        gridsize for heatmap in y axis. default is 20.
    if_adaptive_gridsize : bool
        if True, overwrite gridsizes and set them adaptively based on recording accuracies in x, y axes.

    Returns
    -------
    xx: numpy array
        x-grid coordinates for heatmaps. shape (gridsize_y, gridsize_x)
    yy: numpy array
        y-grid coordinates for heatmaps. shape (gridsize_y, gridsize_x)
    z: 2D numpy array
        KDE values for each cell. shape (gridsize_y, gridsize_x)

    """
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from mitgaze.util import remove_nan

    dir_source_image = Path(dir_source_image)

    media = list(df_media.loc[:,'Presented Media name'].unique())[0]
    W_screen = np.int16(df_media['Recording resolution width'].unique()[0])
    H_screen = np.int16(df_media['Recording resolution height'].unique()[0])
    t_start = np.min(df_media.loc[:,'Recording timestamp'])
    t_end = np.max(df_media.loc[:,'Recording timestamp'])

    calib_accuracy = np.int16(df_media['Average calibration accuracy (pixels)'].unique()[0])
    calib_precision = np.int16(df_media['Average calibration precision (pixels)'].unique()[0])

    if if_adaptive_gridsize:
        gridsize_x = int(W_screen/calib_accuracy,)
        gridsize_y = int(H_screen/calib_accuracy)

        dir_save_main = Path(dir_save_main, 'HEATMAP_grid_adaptive')
    else:
        subfolder = 'HEATMAP_grid_' + str(gridsize_x) + 'x' + str(gridsize_y)
        dir_save_main = Path(dir_save_main, subfolder)

    x = np.array(df_media['Gaze point X'])
    y = np.array(df_media['Gaze point Y'])
    imagename = Path(dir_source_image, media)
    img_current = plt.imread(imagename) # READS BGR as in (X,Y,3)

    par_name = df_media.loc[:,'Participant name'].unique().tolist()[0]
    timeline_name = df_media.loc[:,'Timeline name'].unique().tolist()[0]
    title_info_str = '\n Participant '+par_name+',   '+timeline_name + \
                    ',   Total recording time: '+str((t_end-t_start)/1000)+' seconds'+\
                    '\n Calibration Accuracy: '+str(calib_accuracy)+\
                    ',   Calibration Precision: '+str(calib_precision)

    (fig_W, fig_H) = (W_screen/100, H_screen/100)

    ### just the image
    fig, ax = plt.subplots(figsize = (fig_W, fig_H))
    x, y = remove_nan(x, y)
    try:
        ax, xx, yy, z = plot_heatmap(x, y, img_current,
                                     gridsize_x=gridsize_x, gridsize_y=gridsize_y, )
    except TypeError:
        print('************************************')
        print('Heatmap failed for ', imagename, ' participant ', par_name)
        print('******************************************')
        return None, None, None
    ax.set_title(title_info_str)
    if if_save_fig:
        media_name = Path(media).stem

        dir_save_per_media = Path(dir_save_main, 'HEATMAP_per_media', media_name)
        dir_save_per_media.mkdir(exist_ok=True, parents=True)

        dir_save_per_par = Path(dir_save_main, 'HEATMAP_per_par', par_name)
        dir_save_per_par.mkdir(exist_ok=True, parents=True)

        filename_save_base = par_name + '_' + media_name

        filename_save_per_media = filename_save_base + '_HEATMAP' + '.png'
        filename_save_per_media_pm = Path(dir_save_per_media, filename_save_per_media)
        filename_save_per_media_pp = Path(dir_save_per_par, filename_save_per_media)
        fig.savefig(filename_save_per_media_pm, format='png')
        fig.savefig(filename_save_per_media_pp, format='png')

    plt.close('all')

    ### save kde data
    media = list(df_media['Presented Media name'].unique())[0]
    par = list(df_media['Participant name'].unique())[0]

    df_kde = pd.DataFrame(index=[0], columns=['xx', 'yy', 'z'])
    df_kde.at[0, 'xx'] = xx
    df_kde.at[0, 'yy'] = yy
    df_kde.at[0, 'z'] = z
    dir_save_data = Path(dir_save_main, 'HEATMAP_DATA')
    dir_save_data.mkdir(parents=True, exist_ok=True)

    csv_file = par+ '_' + media_name + '_KDE.csv'
    csv_file = Path(dir_save_data, csv_file)
    df_kde.to_csv(csv_file, index=False)

    pkl_file = par+ '_' + media_name + '_KDE.p'
    pkl_file = Path(dir_save_data, pkl_file)
    df_kde.to_pickle(pkl_file, protocol=4)
    return xx, yy, z


# %% heatmap for gaze data for each image
def plot_heatmap(x, y, img, ax=None,
                 gridsize_x=40, gridsize_y=20,
                 title_str='', cmap='viridis'):
    """
    Plot heatmaps based on gaze data.

    Parameters
    ----------
    x : numpy.array
        array of x values.
    y : numpy.array
        array of y values.
    img :png or jpg image.
        the image used as stimulus while recording x-y values..
    ax : matplotlib axes, optional
        axis of the figure to be used for plotting. The default is None.
    gridsize_x : int, optional
        gridsize for heatmap in x axis. default is 40.
    gridsize_y : int, optional
        gridsize for heatmap in y axis. default is 20.
    title_str : str, optional
        Optional text to be used as title. The default is ''.


    Returns
    -------
    ax : matplotlib axes,
        Axes to plot on, otherwise uses current axes..
    xx: numpy array
        x-grid coordinates for heatmaps.
    yy: numpy array
        y-grid coordinates for heatmaps.
    z: 2D numpy array
        KDE values for each cell.

    """
    import matplotlib.pyplot as plt

    from mitgaze.sns_distributions import kdeplot

    if ax is None:
        ax = plt.gca()

    if len(img.shape)==3:
        ax.imshow(img)
    else:
        ax.imshow(img , cmap = 'gray')

    ax, xx, yy, z = kdeplot(x,y, shade=True, ax=ax,
                shade_lowest = False, vertical = False,
                cmap = cmap, alpha = 0.3,
                cbar = False,
                gridsize_x = gridsize_x,
                gridsize_y = gridsize_y,)
    ax.set_xlim((0,img.shape[1]))
    ax.set_ylim((img.shape[0],0))
    ax.set_title(title_str)
    return ax, xx, yy, z

# %% plot_gaze_df
def plot_gaze_df(df_media, dir_source_image, dir_save_main, plot_type='GAZE_ORDER',
                 if_save_media=False, cmap='viridis'):
    """
    Get an Eye-Tracking Recording Dataframe as input, plot gaze_basic, gaze_order.

    Saves them in two seperate folders, one under the Media Name folder, other arranged
    based on participant names.


    Parameters
    ----------
    df_media : pandas Dataframe
        Dataframe of Eye-tracking data.
    dir_source_image : str or pathlib Path
        where the media files are.
    dir_save_main : str or pathlib Path
        where to save.
    plot_type : str, optional
        what kind of gaze plot. Either color-code order or monochrome plot. The default is 'GAZE_ORDER'.
    if_save_media : Bool, optional
        if save the plots. The default is False.

    Returns
    -------
    None.

    """
    from pathlib import Path
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np


    dir_source_image = Path(dir_source_image)
    dir_save_main = Path(dir_save_main)

    # general oarams from df
    W_screen = np.int16(df_media['Recording resolution width'].unique()[0])
    H_screen = np.int16(df_media['Recording resolution height'].unique()[0])
    t_start = np.min(df_media.loc[:,'Recording timestamp'])
    t_end = np.max(df_media.loc[:,'Recording timestamp'])

    calib_accuracy = np.int16(df_media['Average calibration accuracy (pixels)'].unique()[0])
    calib_precision = np.int16(df_media['Average calibration precision (pixels)'].unique()[0])

    # gaze data
    x = np.array(df_media['Gaze point X'])
    y = np.array(df_media['Gaze point Y'])
    media = list(df_media.loc[:,'Presented Media name'].unique())[0]
    imagename = Path(dir_source_image, media)
    img_current = plt.imread(imagename) # READS BGR as in (X,Y,3)

    par_name = df_media.loc[:,'Participant name'].unique().tolist()[0]
    timeline_name = df_media.loc[:,'Timeline name'].unique().tolist()[0]

    # dir_save = Path(dir_save_main, par_name, timeline_name, output_type)

    title_info_str = '\n Participant '+par_name+',   '+timeline_name + \
                    ',   Total recording time: '+str((t_end-t_start)/1000)+' seconds'+\
                    '\n Calibration Accuracy: '+str(calib_accuracy)+\
                    ',   Calibration Precision: '+str(calib_precision)
    (fig_W, fig_H) = (W_screen/100, H_screen/100)
     ### plot just the image
    fig, ax = plt.subplots(figsize = (fig_W, fig_H))
    ax.imshow(img_current, cmap = 'gray')
    ax.set_title(title_info_str)

    media = list(df_media.loc[:,'Presented Media name'].unique())[0]
    media_name = Path(media).stem

    dir_save_per_media = Path(dir_save_main, plot_type + '_per_media', media_name)
    dir_save_per_media.mkdir(exist_ok=True, parents=True)
    dir_save_per_par = Path(dir_save_main, plot_type + '_per_par', par_name)
    dir_save_per_par.mkdir(exist_ok=True, parents=True)

    filename_save_base = par_name + '_' + media_name
    if if_save_media:
         # save only the media
         filename_save_nodata = filename_save_base + '.png'
         filename_save_nodata_pm = Path(dir_save_per_media, filename_save_nodata)
         filename_save_nodata_pp = Path(dir_save_per_par, filename_save_nodata)
         fig.savefig(filename_save_nodata_pm, format='png')
         fig.savefig(filename_save_nodata_pp, format='png')

    if plot_type == 'GAZE_BASIC':
        ### plot gaze plot
        N = np.shape(x)[0]
        ax.scatter(x[1:N], y[1:N], color= 'c', marker  = 'o')
        ax.set_xlim((0,img_current.shape[1]))
        ax.set_ylim((img_current.shape[0],0))
        ax.set_title(title_info_str)
    elif plot_type == 'GAZE_ORDER':
        ### gaze order
        # my_color = cm.seismic_r(np.linspace(0, 1, x.shape[0]))
        # my_color = cm.rainbow_r(np.linspace(0, 1, x.shape[0]))
        my_color = cm.RdYlBu(np.linspace(0, 1, x.shape[0]))
        N = np.shape(x)[0]
        my_alpha = np.linspace(.7,.3, x.shape[0])
        my_color[:,3] = my_alpha
        ax.scatter(x[1:N], y[1:N], color= my_color[1:N], marker  = 'o')
        ax.set_xlim((0,img_current.shape[1]))
        ax.set_ylim((img_current.shape[0],0))

        #    filename_save = str(idx2+1).zfill(2)+'_'+filename_save+'.png'
        title_info_str = title_info_str + '\n Using ' + cmap + ' ColorMap'


    filename_save_per_media = filename_save_base + '_' + plot_type + '.png'
    filename_save_per_media_pm = Path(dir_save_per_media, filename_save_per_media)
    filename_save_per_media_pp = Path(dir_save_per_par, filename_save_per_media)
    fig.savefig(filename_save_per_media_pm, format='png')
    fig.savefig(filename_save_per_media_pp, format='png')


    plt.close('all')


# %% plot_gaze
def plot_gaze(x, y, img, order=True, ax=None,
              title_str=''):
    """
    Plot the gaze data. Ordered or non-ordered.

    Parameters
    ----------
    x : numpy.array
        array of x values.
    y : numpy.array
        array of y values.
    img :png or jpg image.
        the image used as stimulus while recording x-y values..
    order : Bool, optional
        If order of gaze values should be highlighted using a certain colormap. The default is True.
    ax : matploblib figure axis, optional
        axis of the figure to be used for plotting. The default is None.
    title_str : str, optional
        Optional text to be used as title. The default is ''.

    Returns
    -------
    ax : matploblib figure axis
        axis of the current figure.

    """
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax = plt.gca()

    if len(img.shape)==3:
        ax.imshow(img)
        gaze_color = 'p'
    else:
        gaze_color = 'p'
        ax.imshow(img , cmap = 'gray')

    ### gaze order
    if order:
        my_color = cm.viridis(np.linspace(0, 1, x.shape[0]))
        title_info_str = title_str + '\n Colour order in gaze plot: Yellow -> Green'
        N = np.shape(x)[0]
        my_alpha = np.linspace(.7,.3, x.shape[0])
        my_color[:,3] = my_alpha
        ax.scatter(x[1:N], y[1:N], color= my_color[1:N], marker  = 'o')
        ax.set_xlim((0,img.shape[1]))
        ax.set_ylim((img.shape[0],0))
        ax.set_title(title_info_str)

    ### gaze trail
    else:
        ax.plot(x, y, gaze_color, alpha = 0.4)
        ax.set_xlim((0,img.shape[1]))
        ax.set_ylim((img.shape[0],0))
        ax.set_title(title_str)

    return ax



