#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 03:53:03 2020

@author: wxiao

find all .npy files in a folder
extract participants name and photo name from file name
read heatmap data from .npy files
"""
if __name__ == "__main__":
    
    from pathlib import Path
    import pandas as pd
    import numpy as np
    
    
    #%%
    # get the path for all gaze count files with.npy
    dir_source = Path('/home/wxiao/Documents/for_Weiyi/GAZE_COUNT_GRID_48_24')
    all_files = list(dir_source.glob('*.npy'))
    
    row_list=[]
    
    for file_path in all_files:
    
        fName = file_path.stem.split('.jpg')[0]   # example output: 'P201_a02_2_7_TheBeggarWoman'
        fName = fName.split('_')
        # par means participant
        par = fName[0]  
        media = '_'.join(fName[1:])
        # read heatmap_data and transpose it, so the shape was (24,48)
        heatmap_data = np.load(file_path).T
        
        row_list.append({'file_path': file_path, 'par': par, 'media': media, 'heatmap_data': heatmap_data})
    
    
    # save data into data frame 
    df_heatmap_summary= pd.DataFrame(row_list, columns = ['file_path','par','media','heatmap_data'])
    
    
    """
    Note: data for b22_HS_bridge.png was deleted from the heatmap data frame for consistency reasons.
    Currently, 'b22_HS_bridge' is been excludede from all analysis
    """
    # exclude data for b22_HS_bridge.png
    df_heatmap_summary = df_heatmap_summary.drop(df_heatmap_summary[df_heatmap_summary['media']=='b22_HS_bridge'].index)
    
    
    # save data frame as pickle file
    dir_save = Path('/home/wxiao/Documents/for_Weiyi/MEDIA_grid_operation_summary')
    saving_path = Path(dir_save, 'HEATMAP_DATA_SUMMARY.p')
    df_heatmap_summary.to_pickle(saving_path, protocol=4)
