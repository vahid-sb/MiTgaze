#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 11:42:28 2020

@author: wxiao
"""
# save and load
def save_obj(saving_path, obj):
    import pickle
    with open(saving_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(obj_path):
    import pickle
    with open(obj_path, 'rb') as f:
        return pickle.load(f)
    
    
#%% functions used in kmeans clustering
def save_kmeans_clustering_results(dir_media, dir_save, clustering_info):
    '''
    save clustering results as .png
    save clustering data in .npy

    Parameters
    ----------
    dir_media : PosixPath
        media directory.
    dir_save : PosixPath
        directory to save results.
    num_samples : int
        DESCRIPTION.
    media : String
        media name.
    est : estimator
        esitimator for clustering

    Returns
    -------
    None.

    '''
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    
    # load clustering_info
    media = clustering_info['media']
    est = clustering_info['est']
    par_list = clustering_info['df_per_media']['par'].to_list()
    
    # extract information from the fitted estimator for ploting
    clu_data = get_data_from_est(est)
    
    # check if it is an RGB image or gray scale image
    img_path = Path(dir_media, media+'.jpg')
    [img, img_mode] = plot_helper_img(img_path)
    
    n_clusters = clu_data['centroids'].shape[0]
    
    # plot
    fig, axs = plt.subplots(n_clusters,2, figsize=(12, 3 * n_clusters))
    for i in range(n_clusters):
        if img_mode == 'RGB':
            axs[i,0].imshow(img)  
        else:
            axs[i,0].imshow(img, cmap='gray')
        
        axs[i,1].imshow(clu_data['centroids'][i,:,:], cmap='gray')
        # set title
        axs[i,0].set_title(media)
        axs[i,1].set_title(plot_helper_title(i, clu_data, par_list, option='general'))
        # turn off axis
        axs[i,0].axis('off')
        axs[i,1].axis('off')
    
    # main directory to save clustering results
    dir_save = Path(dir_save,str(n_clusters)+'_clusters')
    dir_save.mkdir(parents = True, exist_ok = True)
    file_name = media +'_#heatmap' + str(len(clu_data['labels'])) +'_#cluster'+str(n_clusters)
    
    # save plots
    dir_save_plots = Path(dir_save,'plots')
    dir_save_plots.mkdir(parents = True, exist_ok = True)
    saving_path_png = Path(dir_save_plots, file_name +'.png')
    plt.savefig(saving_path_png)
    plt.close('all')
    
    # save data in pickle file
    dir_save_data = Path(dir_save,'data')
    dir_save_data.mkdir(parents = True, exist_ok = True)
    saving_path_npy= Path(dir_save_data, file_name + '.pkl')
    save_obj(saving_path_npy, clustering_info)   
    
            
def kmeans_clustering_analysis(dir_media, dir_save, clustering_info):
    # load clustering_info
    cluster_list = clustering_info['cluster_list']
    data = clustering_info['data']
    
    # clustering analysis    
    for n_clusters in cluster_list:
        if n_clusters > data.shape[0]:  # more num_clusters than num_samples
            continue
        est = get_clustering_est(data, n_clusters) # get fitted estimator
        clustering_info['est'] = est
        
        # plot and save data
        save_kmeans_clustering_results(dir_media, dir_save, clustering_info)
        
        
def kmeans_clustering_block(dir_media, dir_save, df_heatmap_summary, media, clustering_info):
    from pathlib import Path
    import numpy as np
    from sklearn.preprocessing import normalize
    
    # load configuration data
    normalization = clustering_info['normalization']
    
    # create subfolder depending on normalization
    if normalization:
        dir_save = Path(dir_save,'normalized')
        dir_save.mkdir(parents = True, exist_ok = True)
    else:
        dir_save = Path(dir_save,'unnormalized')
        dir_save.mkdir(parents = True, exist_ok = True)
        
    # prepare data for clustering
    df_per_media = df_heatmap_summary.loc[df_heatmap_summary['media']==media]  # select corresponding rows
    data = df_per_media['heatmap_data'].to_list()
    data = np.asarray(data)
    data = data.reshape(data.shape[0],-1)
    if normalization:
        data = normalize(data, norm='l2')
        
    clustering_info.update({'media': media, 'df_per_media': df_per_media, 'data': data})
        
    # kmeans clustering
    kmeans_clustering_analysis(dir_media, dir_save, clustering_info)
        
#%% color-gray pairs, calibration pairs, flipping pairs
# build a case/switch statement
def color_pairs(dir_save):
    from pathlib import Path
    dir_save = Path(dir_save,'color')
    dir_save.mkdir(parents = True, exist_ok = True)
    
    # 2 color/gray pairs
    pair_list = [['a08_6_5a_MohoPeruDoorWithWindows', 'b08_6_5b_MohoPeruDoorWithWindows'],
             ['a07_6_2a_Keyhole_LAC_Color', 'b17_6_2b_TheKeyhole_LAC_bw']]
    name_list = ['MohoPeruDoorWithWindows', 'Keyhole_LAC']
    return [pair_list, name_list, dir_save]


def calib_pairs(dir_save):
    from pathlib import Path
    dir_save = Path(dir_save,'calibration')
    dir_save.mkdir(parents = True, exist_ok = True)
    
    # 4 calibrtion pairs
    pair_list = [['a02_2_7_TheBeggarWoman', 'b02_2_7_TheBeggarWoman'], 
                  ['a02_2_7_TheBeggarWoman', 'c02_2_7_TheBeggarWoman'],
                  ['b02_2_7_TheBeggarWoman', 'c02_2_7_TheBeggarWoman'],
                  ['a18_3_19_FernInGrottoCloudForest', 'b16_3_19_FernInGrottoCloudForest']]
    name_list = ['TheBeggarWoman_ab','TheBeggarWoman_ac','TheBeggarWoman_bc','FernInGrottoCloudForest']
    
    return [pair_list, name_list, dir_save]


def flip_pairs(dir_save):
    from pathlib import Path
    # create subfolder to save results
    dir_save = Path(dir_save,'flip')
    dir_save.mkdir(parents = True, exist_ok = True)        
    
    # 9 pairs of Flip
    pair_list = [['b18_X7_WhiteCliffsDrama', 'R180_X7_WhiteCliffsDrama'],
                 ['c14_9_5_Slickrock_ReflectingPool', 'R180_Slickrock_ReflectingPool'],
                 ['a15_4_2_ArniMarbleQuarry', 'b05_180_4_2_ArniMarbleQuarry'],
                 ['a12_X2_LDHWallVortex_cropped', 'b20_180_X2_LDHWallVortex_cropped'],
                 ['a20_9_1_TheQueenOfMaligne', 'b19_180_9_1_TheQueenOfMaligne'],
                 ['a21_E_8_11_HomeCloudPanorama', 'c20_180_E_8_11_HomeCloudPanorama'],
                 ['a05_3_6_CircularChimney_AntelopeCyn', 'b10_180_3_6_CircularChimney_AntelopeCyn'],
                 ['b06_X5_Stripes_Ribbons_WaterholesCyn', 'b21_180_X5_Stripes_Ribbons_WaterholesCyn'],
                 ['a19_6_11_BuckskinViewFromCenterOfEarth', 'b13_180_6_11_BuckskinViewFromCenterOfEarth']] 
    name_list = ['WhiteCliffsDrama','Slickrock_ReflectingPool','ArniMarbleQuarry','LDHWallVortex_cropped', 
                 'TheQueenOfMaligne', 'HomeCloudPanorama','CircularChimney_AntelopeCyn','Stripes_Ribbons_WaterholesCyn',
                 'BuckskinViewFromCenterOfEarth']
    
    return [pair_list, name_list, dir_save]


def clustering_preparation(dir_save, key):
    '''
    a case/switch statement with 3 cases: 'color', 'calib', 'flip'
    depending on the case, get the assocated:
        pair_list: list of media pairs
        name_list: list of name of the media pairs
        dir_save: path of the folder for saving results
        
    Parameters
    ----------
    dir_save : PosixPath
        path to the folder which saves the result.
    key : String
        available key values: 'color', 'calib', 'flip'

    Returns
    -------
    output
        DESCRIPTION.

    '''
    # options={'color':color_pairs(dir_save),
    #          'calib':calib_pairs(dir_save),
    #          'flip':flip_pairs(dir_save)}
    # [pair_list, name_list, dir_save] = options[key]
    if key in ['color', 'calib', 'flip']:

        if key=='color':
            [pair_list, name_list, dir_save] = color_pairs(dir_save)
        elif key=='calib':
            [pair_list, name_list, dir_save] = calib_pairs(dir_save)
        elif key=='flip':
            [pair_list, name_list, dir_save] = flip_pairs(dir_save)
            
        output = {'pair_list': pair_list, 'name_list': name_list, 'dir_save': dir_save}
        output['n_pairs'] = len(name_list)
    else:
        print("clustering_preparation(dir_save, key), try key='color', 'calib' or 'flip' ")
        import sys
        sys.exit(0)
        
    return output
    

#%% functions

def get_clustering_data(df, normalization, operation):
    
    import numpy as np
    from sklearn.preprocessing import normalize
    
    data0 = df['media0'].to_list()
    data0 = np.asarray(data0)
    data0 = data0.reshape(data0.shape[0],-1)
    
    data1 = df['media1'].to_list()
    data1 = np.asarray(data1)
    data1 = data1.reshape(data1.shape[0],-1)

    if normalization:
        data0 = normalize(data0, norm='l2')
        data1 = normalize(data1, norm='l2')
    
    if operation in ['diff', 'abs_diff']:
        if operation == 'diff':
            output = data0-data1
        elif operation == 'abs_diff':
            output = abs(data0-data1)
    else:
        print("get_clustering_data(df, operation, normalization): try operation='diff' or 'abs_diff' ")
        import sys
        sys.exit(0)
        
    return output
    

def get_clustering_est(data, n_clusters):
    '''
    call KMeans to specify number of clusters
    fit the data to get the estimator

    Parameters
    ----------
    data : numpy.ndarray
        prepared for kmeans clustering: {array-like, sparse matrix} of shape (n_samples, n_features)
    n_clusters : int
        number of clusters.

    Returns
    -------
    est : estimator
        fitted estimator.

    '''
    from sklearn.cluster import KMeans
    # k-means clustering
    est = KMeans(n_clusters)
    est.fit(data)
    return est


def get_data_from_est(est):
    '''
    extract information from the fitted estimator for ploting

    Parameters
    ----------
    est : estimator
        fitted estimator of data.

    Returns
    -------
    est_data : dictionary
        includes cluster centroids, cluster labels, and number of samples in each cluster

    '''
    import numpy as np
    centroids = est.cluster_centers_
    n_clusters = centroids.shape[0]
    centroids = centroids.reshape(n_clusters,24,48)
    labels = est.labels_
    values, counts = np.unique(labels, return_counts=True)
    est_data = {'est': est, 'centroids': centroids, 'labels': labels, 'label_values': values, 'label_counts': counts}
    return est_data


def plot_helper_img(img_path):
    '''
    a helper function for ploting

    Parameters
    ----------
    img_path : PosixPath
        path of a image.

    Returns
    -------
    list
        img: image data for ploting
        img_mode: mode of the image, 'L' for gray scale image or 'RGB' for color image.

    '''
    from PIL import Image
    import matplotlib.pyplot as plt
    # check if it is an RGB image or gray scale image
    img_file = Image.open(img_path,'r')
    img_mode = img_file.mode
    # prepare data for plotting
    img = plt.imread(img_path)
    return [img, img_mode]


def plot_helper_title(clu_idx, plot_info, par_list, option):
    from operator import itemgetter
    import numpy as np
    basic_title = 'cluster '+ str(plot_info['label_values'][clu_idx]) + ' (#media/#total = ' + str(plot_info['label_counts'][clu_idx]) + '/' + str(len(plot_info['labels'])) +')'
    if option == 'general':
        title = basic_title
    elif option == 'diff':
        title= 'difference \n'+ basic_title
    elif option == 'abs_diff':
        title= 'abs(difference) \n'+ basic_title
    else:
        print("plot_helper_title(clu_idx, plot_info, par_list, option): option could be 'general', 'diff', 'abs_diff' ")
        import sys
        sys.exit(0)
        
    if plot_info['label_counts'][clu_idx]<5:
        idx = np.where(plot_info['labels']==clu_idx)[0]
        pars = itemgetter(*idx)(par_list)
        title = title +'\n' + str(pars)
        
    return title

    
def save_pair_diff_clustering_results(dir_media, dir_save, clustering_info):
    '''
    plot the clustering result
    save the plot as .png
    save data used for plotting as .npy file
    
    Parameters
    ----------
    dir_media : PosixPath
        path to the media directory.
    dir_save : PosixPath
        path to the folder which saves the result.
    name : String
        name of the media pairs.
    data_pair : list
        contains 2 string object, which are media names as stored in the media directory.
    est_diff : dictionary
        stores information about the fitted estimator for plotting.
    est_abs_diff : dictionary
        stores information about the fitted estimator for plotting.

    Returns
    -------
    None.

    '''
    import numpy as np
    from pathlib import Path
    import matplotlib.pyplot as plt

    
    # load clustering_info, current key_list=[]
    data_pair = clustering_info['data_pair']
    name = clustering_info['name']
    est_diff = clustering_info['est_diff']
    est_abs_diff = clustering_info['est_abs_diff']
    par_list = clustering_info['df_data_pair']['par'].to_list()
    
    # extract information from the fitted estimator for ploting
    clu_diff = get_data_from_est(est_diff)
    clu_abs_diff = get_data_from_est(est_abs_diff)
    
    # plot
    img_path_0 = Path(dir_media, data_pair[0] +'.jpg')
    img_path_1 = Path(dir_media, data_pair[1] + '.jpg')
    [img_0, img_mode_0] = plot_helper_img(img_path_0)
    [img_1, img_mode_1] = plot_helper_img(img_path_1)
    
    n_clusters = clu_diff['centroids'].shape[0]
    fig, axs = plt.subplots(n_clusters,4, figsize=(20, 3 * n_clusters))
    for i in range(n_clusters):
        if img_mode_0 == 'RGB':
            axs[i,0].imshow(img_0)  
        else:
            axs[i,0].imshow(img_0, cmap='gray')
        
        if img_mode_1 == 'RGB':
            axs[i,1].imshow(img_1)  
        else:
            axs[i,1].imshow(img_1, cmap='gray')
        
        axs[i,2].imshow(clu_diff['centroids'][i,:,:], cmap='gray')
        axs[i,3].imshow(clu_abs_diff['centroids'][i,:,:], cmap='gray')
        
        # set title
        axs[i,0].set_title(data_pair[0])
        axs[i,1].set_title(data_pair[1])
        axs[i,2].set_title(plot_helper_title(i, clu_diff, par_list, option='diff'))
        axs[i,3].set_title(plot_helper_title(i, clu_abs_diff, par_list, option='abs_diff'))
        # turn off axis
        axs[i,0].axis('off')
        axs[i,1].axis('off')
        axs[i,2].axis('off')
        axs[i,3].axis('off')
        
    # main directory to save clustering results
    dir_save = Path(dir_save,str(n_clusters)+'_clusters')
    dir_save.mkdir(parents = True, exist_ok = True)
    file_name = name+'_#heatmap' + str(len(clu_diff['labels'])) +'_#cluster'+str(n_clusters)
    
    # save plots
    dir_save_plots = Path(dir_save,'plots')
    dir_save_plots.mkdir(parents = True, exist_ok = True)
    saving_path_png = Path(dir_save_plots, file_name +'.png')
    plt.savefig(saving_path_png)
    plt.close('all')
    
    # save data as .pkl 
    dir_save_data = Path(dir_save,'data')
    dir_save_data.mkdir(parents = True, exist_ok = True)
    saving_path_npy= Path(dir_save_data, file_name + '.pkl')
    save_obj(saving_path_npy, clustering_info)
            
            
def data_pair_to_df(df_heatmap_summary, data_pair):
    '''
    for each participant, get the heatmaps for the media pairs

    
    Parameters
    ----------
    df_heatmap_summary : pandas. dataframe
        dataframe that stores the heatmap data for all participants and media
    data_pair : list
        contains 2 string object, which are media names as stored in the media directory.
    dir_save : PosixPath
        path of folder to save the result.

    Returns
    -------
    df_data_pair : pandas.dataframe
        columns = ['par', 'media0', 'media1']

    '''
    import pandas as pd
    
    # check if media is viewed by the same list of participants
    par_list_0 = df_heatmap_summary.loc[df_heatmap_summary['media']==data_pair[0]]['par']
    par_list_1 = df_heatmap_summary.loc[df_heatmap_summary['media']==data_pair[1]]['par']
    par_list = list((set(par_list_0) & set(par_list_1)))  # find intersection
    
    row_list=[]
    # for each participant, get the heatmap for the media pairs, calculate difference and absolute difference between heatmap data 
    for par in par_list:
        media0 = df_heatmap_summary.loc[(df_heatmap_summary['par']==par) & (df_heatmap_summary['media']==data_pair[0])]['heatmap_data'].iloc[0]
        media1 = df_heatmap_summary.loc[(df_heatmap_summary['par']==par) & (df_heatmap_summary['media']==data_pair[1])]['heatmap_data'].iloc[0]
      
        row_list.append({'par': par, 'media0': media0, 'media1': media1})
    
    # original in comparison to normalized
    df_data_pair = pd.DataFrame(row_list, columns = ['par', 'media0', 'media1'])
    
    return df_data_pair


def pair_diff_clustering_analysis(dir_media, dir_save, clustering_info):
    # load clustering_info, current key_list=['cluster_list','df_data_pair','data_pair','name']
    cluster_list = clustering_info['cluster_list']
    df_data_pair = clustering_info['df_data_pair']
    normalization = clustering_info['normalization']
    
    # clustering analysis    
    for n_clusters in cluster_list:
        if n_clusters > len(df_data_pair): # more num_clusters than num_samples
            continue
        data_diff = get_clustering_data(df_data_pair, normalization, operation='diff')
        est_diff = get_clustering_est(data_diff, n_clusters) # get fitted estimator
        data_abs_diff = get_clustering_data(df_data_pair, normalization, operation='abs_diff')
        est_abs_diff = get_clustering_est(data_abs_diff, n_clusters) # get fitted estimator
        # plot and save data
        clustering_info.update({'data_diff': data_diff, 'est_diff': est_diff, 'data_abs_diff': data_abs_diff, 'est_abs_diff': est_abs_diff})
        save_pair_diff_clustering_results(dir_media, dir_save,clustering_info)
        
        
def pair_diff_clustering_block(df_heatmap_summary, dir_media, dir_save, clustering_info, case):
    from pathlib import Path
    
    # load configuration data
    normalization = clustering_info['normalization']

    # get the media pairs, names of the pairs, create subfolder to store the results
    source_dict = clustering_preparation(dir_save, case)
    
    # create subfolder depending on normalization
    dir_save = source_dict['dir_save']
    if normalization:
        dir_save = Path(dir_save,'normalized')
        dir_save.mkdir(parents = True, exist_ok = True)
    else:
        dir_save = Path(dir_save,'unnormalized')
        dir_save.mkdir(parents = True, exist_ok = True)
        
    # iterate over the pairs in the name_list
    for idx in range(source_dict['n_pairs']):
        data_pair = source_dict['pair_list'][idx] 
        name = source_dict['name_list'][idx] 
        clustering_info.update({'data_pair': data_pair, 'name': name})
        
        # prepare the data, columns = ['par', 'media0', 'media1']
        df_data_pair = data_pair_to_df(df_heatmap_summary, data_pair)
        clustering_info['df_data_pair'] = df_data_pair
        
        # kmeans clustering
        pair_diff_clustering_analysis(dir_media, dir_save, clustering_info)