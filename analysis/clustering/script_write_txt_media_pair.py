#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 00:58:42 2020

@author: wxiao
get the stored information from KMeans clustering analysis for media pairs
for each media pair, write the participants under each cluster (diff, abs_diff)
results stored in summary.txt
"""

if __name__ == "__main__":
    
    from pathlib import Path
    import numpy as np
    from mitgaze.kmeans_clustering import load_obj
    import pandas as pd
    
    
    # kmeans pairs
    subfolders = ['color', 'calibration', 'flip']
    idx=2
    dir_source = Path('/home/wxiao/Documents/for_Weiyi/analysis_results/kmeans_clustering_results_media_pairs', subfolders[idx])
    dir_save = Path('/home/wxiao/Documents/for_Weiyi/analysis_results/kmeans_clustering_results_media_pairs', subfolders[idx])
    
    normalization=True
    if normalization:
        dir_source = Path(dir_source, 'normalized')
        dir_save = Path(dir_save, 'normalized')
    else:
        dir_source = Path(dir_source, 'unnormalized')
        dir_save = Path(dir_save, 'unnormalized')
    
    # write to .txt files
    f_summary = Path(dir_save, 'summary.txt')
    with open(f_summary, 'w+') as f:
       f.write('dir_source: '+ str(dir_source)+'\n\n')
       
    cluster_list = list(np.arange(2,13))
    
    def listToString(aList):
        output = ", ".join(aList)
        return output
            
    for n_clusters in cluster_list:
        dir_data = Path(dir_source,str(n_clusters)+'_clusters/data')
        all_files = list(dir_data.glob('*.pkl'))
               
        for pkl_file in all_files:
            # extract info from .pkl file
            clustering_info = load_obj(pkl_file)
            
            # write results to .txt 
            with open(f_summary, 'a') as f:
                f.write('## '+ str(n_clusters)+'_clusters' + '\n')
    
            # put par_list and labels into dataframe for easier indexing
            par_list = clustering_info['df_data_pair']['par'].to_list()
            labels = clustering_info['est_diff'].labels_
            df_results = {'par': par_list, 'label': labels}
            df_results = pd.DataFrame(df_results)
               
            with open(f_summary, 'a') as f:
                f.write('# ' + pkl_file.stem + ', (diff)'+'\n')
                for idx in range(n_clusters):
                    pars = df_results.loc[df_results['label'] == idx]['par'].to_list()
                    f.write('cluster_'+ str(idx)+ ': ' + listToString(pars)+'\n')
                
            labels = clustering_info['est_abs_diff'].labels_
            df_results = {'par': par_list, 'label': labels}
            df_results = pd.DataFrame(df_results)
                    
            with open(f_summary, 'a') as f:
                f.write('# ' + pkl_file.stem + ', (abs_diff)'+'\n')
                for idx in range(n_clusters):
                    pars = df_results.loc[df_results['label'] == idx]['par'].to_list()
                    f.write('cluster_'+ str(idx)+ ': ' + listToString(pars)+'\n')
                f.write('\n')
            
    


    

    
