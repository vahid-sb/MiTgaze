#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 17:40:47 2021

@author: wxiao
"""
if __name__ == "__main__":
    
    from pathlib import Path
    import pickle
    import pandas as pd
    
    dir_source = Path('/home/wxiao/Documents/for_Weiyi/MEDIA_duration')
    all_files = list(dir_source.glob('*.p'))
    all_files.sort()
    
    # load all df to a list and concatenate
    df_list=[]
    for file in all_files:
        with open(file, 'rb') as f :
            df = pickle.load(f)
            df_list.append(df)
            
    df_summary = pd.concat(df_list)
    
    # save data frame as pickle file
    dir_save = Path('/home/wxiao/Documents/for_Weiyi/MEDIA_grid_operation_summary')
    saving_path = Path(dir_save, 'MEDIA_DURATION_SUMMARY.p')
    df_summary.to_pickle(saving_path, protocol=4)
