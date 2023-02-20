#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 13:19:25 2021

@author: wxiao

given the participant number
find the duration of each media viewed by that participant

Note:
    Recording timestamp use computer clock, in milliseconds
    Eyetracker timestamp use eyetracker clock
"""

if __name__ == "__main__":
    
    from pathlib import Path
    import pandas as pd
    
    # find all .tsv files in the folder
    dir_source = Path('/home/wxiao/Documents/for_Weiyi/TSV/parsed_par')
    all_files = list(dir_source.glob('*.tsv'))
    all_files.sort()
    
    #%%
    for filename in all_files:
        # get participant name 
        par = filename.stem[4:]
        print(par)
        
        # only extract these 3 columns from the .tsv file
        col_list = ['Recording timestamp','Eyetracker timestamp','Presented Media name']
        # note: for P501 set low_memory = True, because it is too large
        df_csv = pd.read_csv(filename, sep='\t', usecols=col_list, low_memory=False, error_bad_lines=False)
       
        # get the list of media viewed by the participant
        media_list = df_csv['Presented Media name'].unique()
        media_list.sort()
        
        # media has name that start with 'a', 'b', 'c', 'R'
        media_list = [media for media in media_list if media.startswith(('a','b','c','R'))]
        
        # generate a df for each participant
        row_list=[]
        for media in media_list:
            # use recording time stamps, returns a series of time stamps which uses the computer clock 
            time_stamps = df_csv.loc[df_csv['Presented Media name']==media]['Recording timestamp']
            duration = max(time_stamps)-min(time_stamps)
            media = media.strip('.jpg')
            row_list.append({'par': par, 'media': media, 'duration': duration})
            
        df_duration= pd.DataFrame(row_list, columns = ['par','media','duration'])
        
        # save data frame as pickle file
        dir_save = Path('/home/wxiao/Documents/for_Weiyi/temp')
        saving_path = Path(dir_save, par+'.p')
        df_duration.to_pickle(saving_path, protocol=4)
        
