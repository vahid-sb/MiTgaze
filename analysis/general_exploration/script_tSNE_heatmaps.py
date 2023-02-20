#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 01:54:15 2020

@author: wxiao
"""
if __name__ == "__main__":
    
    import pickle
    from pathlib import Path
    from mitgaze.kmeans_clustering import kmeans_clustering_block
    import numpy as np
    from sklearn.manifold import TSNE as sklearn_TSNE
    from collections import defaultdict
    from bokeh.palettes import Plasma256
    from sklearn.preprocessing import normalize
    
    
    # read dataframe 
    pkl_file = Path('/home/wxiao/Documents/for_Weiyi/MEDIA_grid_operation_summary/HEATMAP_DATA_SUMMARY.p')
    with open(pkl_file,"rb") as f:
        df_heatmap_summary = pickle.load(f)
            
    # drop some rows
    idx_drop = df_heatmap_summary.loc[df_heatmap_summary['par'].isin(['P306', 'P501'])].index
    df_heatmap_summary = df_heatmap_summary.drop(idx_drop)
    
    # get heatmap matrix, participant list and media list
    data_matrix = df_heatmap_summary.heatmap_data.to_list()
    data_matrix = np.asarray(data_matrix)  # shape: 2003,24,48
    data_matrix = data_matrix.reshape(data_matrix.shape[0],-1)
    data_matrix = normalize(data_matrix, norm='l2')
    
    individuals = df_heatmap_summary.par.to_list()
    media_list = df_heatmap_summary.media.to_list()
    
    # collect media index for indexing heatmap data later
    media_index = defaultdict(list)
    for idx, media in enumerate(media_list):
        media_index[media].append(idx)
    
    #%% setting colors
    media_index_keys = list(media_index.keys())
    media_index_keys.sort()
    
    def set_color_dict(mediaList):
        num_media = len(mediaList)
        color_idx = np.linspace(0,255,num_media)
        color_idx = [int(i) for i in color_idx]
        color_dict={}
        
        for idx, media in enumerate(mediaList):
            color_dict[media]=Plasma256[color_idx[idx]]
        
        return color_dict
    
    color_dict = set_color_dict(media_index_keys)
    
    # media_a = []
    # media_b=[]
    # media_c=[]
        
    # for media in media_index_keys:
    #     if media[0]=='a':
    #         media_a.append(media)
    #     elif media[0]=='b':
    #         media_b.append(media)
    #     elif media[0]=='c':
    #         media_c.append(media)
            
    # color_dict_a = set_color_dict(media_a)
    # color_dict_b = set_color_dict(media_b)
    # color_dict_c = set_color_dict(media_c)
    
    #%%
    # Use the tSNE projection of the genotype data
    dim1=0
    dim2=1
    show_label=False
    img_io={}
    img_io['fig_title'] = 't-SNE projection of heatmap data'
    img_io['html_name'] = 'interactive_tsne.html'
    img_io['html_title'] = 't-SNE projection'
    img_io['dir_save'] = '/home/wxiao/Documents/for_Weiyi/analysis_results/tSNE_heatmaps'
    
    
    # Project the genotype data matrix to two dimensions via t-SNE. This may take several minutes to run.
    proj_tsne = sklearn_TSNE(n_components = 2).fit_transform(data_matrix)   #sklearn
    
    #%% plot
    '''
    Note: change to function later
    '''
    # Generate interactive HTML files
    dset = proj_tsne
    
    from pathlib import Path
    from bokeh.plotting import figure, show, save, output_file
    from bokeh.models import ColumnDataSource, LabelSet
    
    p = figure(plot_width=1350, plot_height=800)
    p.title.text = img_io['fig_title']
    
        
    source = ColumnDataSource(data=dict(x1=list(dset[:,dim1]),
                                    x2=list(dset[:,dim2]),
                                    names = individuals))
    
    labels = LabelSet(x='x1', y='x2', text='names', level='glyph',
                      x_offset=2, y_offset=2, source=source, render_mode='canvas')
    
    p.add_layout(labels)
    
    for media in media_index_keys:
        proj_within_media = dset[media_index[media]]
        p.circle(proj_within_media[:,dim1], proj_within_media[:,dim2],
                 legend_label=media, color = color_dict[media])
        
    p.legend.location ="bottom_left" #  # "top_left"
    p.legend.click_policy="hide"
    p.add_layout(p.legend[0], 'right')
    
    dir_save = img_io['dir_save']
    saving_path = Path(dir_save, img_io['html_name'] )
    output_file(saving_path, title=img_io['html_title'])
    
    # save(p)
    show(p)
    
