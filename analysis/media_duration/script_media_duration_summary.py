#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 17:53:13 2021

@author: wxiao

calculate mean and std of each media viewing time
"""
if __name__ == "__main__":
        
    from pathlib import Path
    import pickle
    import pandas as pd
    import numpy as np
    
    df_path = Path('/home/wxiao/Documents/for_Weiyi/MEDIA_grid_operation_summary/MEDIA_DURATION_SUMMARY.p')
    
    with open(df_path, 'rb') as f:
        df_summary = pickle.load(f)
        
    media_list = df_summary.media.unique()
    media_list.sort()
    
    #%%
    txt_path = Path('/home/wxiao/Documents/for_Weiyi/analysis_results/general_info/media_duration.txt')
    with open(txt_path, 'w+') as f:
           f.write('source: '+ str(df_path)+'\n')
           f.write('For each media, calculate mean and standard deviation of its duration'+'\n\n')
           f.write('-----------------------------------------------------------')
           f.write('\n')
           
    for media in media_list:
        dur = df_summary.loc[df_summary['media']==media]['duration']
        dur = dur/1000 # from milliseconds to seconds
        mean = np.mean(dur)
        std = np.std(dur)
        num_par = len(dur)
        
        with open(txt_path, 'a') as f:
           f.write('# '+media + ', viewed by '+ str(num_par)+' participants'+'\n')
           f.write('mean [s]: '+str(mean) +'\n')
           f.write('std [s]: '+str(std)+'\n\n')
           f.write('-----------------------------------------------------------')
           f.write('\n')
        
    #%%
    t_dur = df_summary.duration
    t_dur = t_dur/1000 # from milliseconds to seconds
    t_mean = np.mean(t_dur)
    t_std = np.std(t_dur)
    num_par = len(t_dur)
    
    with open(txt_path, 'a') as f:
           f.write('# Mean and std over all '+str(num_par)+' samples are:'+'\n')
           f.write('mean [s]: '+str(t_mean) +'\n')
           f.write('std [s]: '+str(t_std)+'\n\n')
           f.write('-----------------------------------------------------------')
           f.write('\n')