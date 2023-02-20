#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 11:28:16 2020

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
    f_summary = Path(dir_save, 'correlation_#'+str(n_clusters)+'_clusters_centroids.txt')
    with open(f_summary, 'w+') as f:
       f.write('dir_source: '+ str(dir_source)+'\n\n')
    
    #%% load image property dataframe
    # columns: ['media', 'img_light_intensity', 'img_mode', 'img_r', 'img_g', 'img_b',
    #           'img_h', 'img_s', 'img_v', 'edge_roberts', 'edge_sobel', 'edge_scharr',
    #           'edge_prewitt', 'edge_farid', 'AOI_r', 'AOI_g', 'AOI_b', 'AOI_c','AOI_m', 'AOI_y']   
    f_property = Path('/home/wxiao/Documents/for_Weiyi/MEDIA_grid_operation_summary/media_grid_H24_W48_all.p')
    df_property = load_obj(f_property) 
    properties=['img_light_intensity', 'img_r', 'img_g', 'img_b', 'img_h', 'img_s', 'img_v', 'edge_roberts', 'edge_sobel', 'edge_scharr',
                'edge_prewitt', 'edge_farid', 'AOI_r', 'AOI_g', 'AOI_b', 'AOI_c','AOI_m', 'AOI_y']
            
    #%% load clustering results
    dir_data = Path(dir_source,str(n_clusters)+'_clusters/data')
    all_files = list(dir_data.glob('*.pkl'))
    all_files.sort()
    
    def get_pearson_correlation(a,b):
        error=0
        err_msg=''

        if type(b)==list:
            if len(b)==0:
                error=1
                err_msg='N/A'
                r_corr=np.nan
                p_corr=np.nan
        elif type(b)==float:
            if pd.isna(b):
                error=1
                err_msg='N/A'
                r_corr=np.nan
                p_corr=np.nan
        elif type(b)==np.ndarray:
            if sum(b.flatten())==0:
                error=1
                err_msg='array of zeros'
                r_corr=np.nan
                p_corr=np.nan
            else:
                r_corr, p_corr = pearsonr(a.flatten(), b.flatten())
        
        return [r_corr, p_corr, error, err_msg] 
    
    
    for pkl_file in all_files:
    
        # find media name, cluster centroids
        clustering_info = load_obj(pkl_file)
        media = clustering_info['media']
        est_data = get_data_from_est(clustering_info['est'])
        centroids = est_data['centroids']
        img_mode = df_property.loc[df_property['media']==media]['img_mode'].iloc[0]
        with open(f_summary, 'a') as f:
            f.write('------------------------------------------------------------------------------\n')
            
        for i in range(centroids.shape[0]):
            ctr = centroids[i,:,:]
            corr=[]
            with open(f_summary, 'a') as f:
                f.write('# '+media+', '+ img_mode +', centroids '+str(i)+' (correlation r-value, p-value):'+'\n')
                
            for j in range(len(properties)):
                prop = df_property.loc[df_property['media']==media][properties[j]].iloc[0]
                [r_corr, p_corr, error, err_msg] = get_pearson_correlation(ctr, prop)
                corr.append(abs(r_corr)) #abs for finding maximum
            
            # np.nanargmax does not return all occurence of maximum values, only the first one is returned
            r_max = np.nanmax(corr)
            idx_max = np.argwhere(corr==r_max)
         
            with open(f_summary, 'a') as f:
                f.write('The highest correlation value is from: '+ str([properties[int(idx)] for idx in idx_max])+ '\n')
                
            for j in range(len(properties)):
                prop = df_property.loc[df_property['media']==media][properties[j]].iloc[0]
                [r_corr, p_corr, error, err_msg] = get_pearson_correlation(ctr, prop)
                with open(f_summary, 'a') as f:
                    if error:
                        f.write(properties[j]+': '+ str(r_corr)+', '+str(p_corr)+', error: '+err_msg+ '\n')
                    else:
                        f.write(properties[j]+': '+ str(r_corr)+', '+str(p_corr)+'\n')
                        
            with open(f_summary, 'a') as f:
                f.write('\n')
