#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 11:51:55 2020

@author: wxiao
"""

if __name__ == "__main__":
    
    from pathlib import Path
    import numpy as np
    from mitgaze.kmeans_clustering import load_obj, get_data_from_est
    import pandas as pd
    from scipy.stats import pearsonr
    
    #%% set source and saving directories
    dir_source = Path('/home/wxiao/Documents/for_Weiyi/analysis_results/kmeans_clustering_results(all)')
    dir_save = Path('/home/wxiao/Documents/for_Weiyi/analysis_results/cluster_correlation_results')
    
    normalization=False
    if normalization:
        dir_source = Path(dir_source, 'normalized')
        dir_save = Path(dir_save, 'normalized')
        dir_save.mkdir(parents = True, exist_ok = True)
    else:
        dir_source = Path(dir_source, 'unnormalized')
        dir_save = Path(dir_save, 'unnormalized')
        dir_save.mkdir(parents = True, exist_ok = True)
    
    n_clusters=4
    # write to .txt files
    f_summary = Path(dir_save, 'correlation_#'+str(n_clusters)+'_clusters_mean.txt')
    with open(f_summary, 'w+') as f:
       f.write('dir_source: '+ str(dir_source)+'\n\n')
    
    #%% load correlation dataframe
    # columns: ['media', 'img_light_intensity', 'img_mode', 'img_r', 'img_g', 'img_b',
    #           'img_h', 'img_s', 'img_v', 'edge_roberts', 'edge_sobel', 'edge_scharr',
    #           'edge_prewitt', 'edge_farid', 'AOI_r', 'AOI_g', 'AOI_b', 'AOI_c','AOI_m', 'AOI_y']   
    f_corr = Path('/home/wxiao/Documents/for_Weiyi/analysis_results/correlation_results/correlation_all.p')
    df_corr = load_obj(f_corr) 
    properties=['img_light_intensity', 'img_r', 'img_g', 'img_b', 'img_h', 'img_s', 'img_v', 'edge_roberts', 'edge_sobel', 'edge_scharr',
                'edge_prewitt', 'edge_farid', 'AOI_r', 'AOI_g', 'AOI_b', 'AOI_c','AOI_m', 'AOI_y']
            
    #%% load clustering results
    dir_data = Path(dir_source,str(n_clusters)+'_clusters/data')
    all_files = list(dir_data.glob('*.pkl'))
    all_files.sort()
    
    
    for pkl_file in all_files:
    
        # find media name, cluster centroids
        clustering_info = load_obj(pkl_file)
        media=clustering_info['media']
        img_mode = df_corr.loc[df_corr['media']==media]['img_mode'].iloc[0]
        par_list = clustering_info['df_per_media']['par'].to_list()
        labels = clustering_info['est'].labels_
        
        with open(f_summary, 'a') as f:
            f.write('------------------------------------------------------------------------------\n')
        for i in range(n_clusters):
            pars = [par_list[idx] for idx, label in enumerate(labels) if label==i ]
            with open(f_summary, 'a') as f:
                f.write('# '+ pkl_file.stem+', '+ img_mode +' (mean correlation):'+'\n')
                f.write('cluster_'+str(i)+' (#pars: '+str(len(pars))+'): '+ str(pars) +'\n')
                
            # find correlations
            df_media = df_corr.loc[(df_corr['media']==media) & (df_corr['par'].isin(pars))]
            df_mean = df_media.mean(axis=0, skipna=True) # this is of type: Series
            columns = df_mean.index.to_list()
           
            # np.nanargmax does not return all occurence of maximum values, only the first one is returned
            r_max = np.nanmax(df_mean)
            idx_max = np.argwhere(df_mean.to_list()==r_max)
            
            with open(f_summary, 'a') as f: 
                f.write('The highest correlation value is from: '+ str([properties[int(idx)] for idx in idx_max])+ '\n')
                        
                for col in columns:
                    f.write(col+': '+str(df_mean[col])+'\n')
                
                f.write('\n')
                    
            
        