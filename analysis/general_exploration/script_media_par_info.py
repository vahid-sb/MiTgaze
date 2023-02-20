#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 07:06:25 2020

@author: wxiao

Generate a 'par_per_media.txt' file: 
    for each media, which participant viewed the media in experiment. 

Generate a 'media_per_par.txt' file:
    for each par, which media are viewed in experiment
    
"""
if __name__ == "__main__":

    import pickle
    from pathlib import Path
    import numpy as np
               
    # set directory to store the .txt files
    dir_save = Path('/home/wxiao/Documents/for_Weiyi/analysis_results/general_info')
    
    # read dataframe 
    pkl_file = Path('/home/wxiao/Documents/for_Weiyi/MEDIA_grid_operation_summary/HEATMAP_DATA_SUMMARY.p')
    with open(pkl_file,"rb") as f:
        df_heatmap_summary = pickle.load(f)
 
    # media info and participants info
    media_list = df_heatmap_summary.media.unique()
    media_list.sort()
    
    par_list = df_heatmap_summary.par.unique()
    par_list.sort()
    
    #%%
    par_per_media={}
    # find participant for each media
    for media in media_list:
        pars = df_heatmap_summary.loc[df_heatmap_summary['media']==media]['par'].to_list()
        pars.sort()
        par_per_media[media]=pars
    
    media_per_par={}
    #find media viewed by each participant
    for par in par_list:
        photos = df_heatmap_summary.loc[df_heatmap_summary['par']==par]['media'].to_list()
        photos.sort()
        media_per_par[par]=photos
    
    #%%
    # find common participants across media
    d=list(par_per_media.values())  
    d=d[2:] #ignore the participant info for the first two media
    common_pars = set(d[0]).intersection(*d)
    common_pars = list(common_pars)
    common_pars.sort()
    
    # there is very few media viewed by all participants, so find the participants who viewed 64 photos
    d=list(media_per_par.values())
    d=[len(l) for l in d]
    d=np.asarray(d)
    idx_64=np.where(d==64)[0]
    par_64=par_list[idx_64]
        
    # check if all 64 photos viewed by those participants are the same
    d=list(media_per_par.values())
    d=[d[i] for i in idx_64]
    common_photos = set(d[0]).intersection(*d)  # length shoud be 64
    
    #%%
    def listToString(aList):
        output = ", ".join(aList)
        return output
    
    # write par_per_media to .txt files
    f_par_per_media = Path(dir_save, 'par_per_media.txt')
    with open(f_par_per_media, 'w+') as f:
        f.write('source_file: '+ str(pkl_file)+'\n\n')
    
    with open(f_par_per_media, 'a') as f:
        for media in media_list:
            pars = par_per_media[media]
            f.write('# '+ media+' ('+str(len(pars))+' participants): '+'\n')
            f.write(listToString(pars)+'\n')
            f.write('\n')
    
    # R180_Slickrock_ReflectingPool: viewed by 4 participants
    # R180_X7_WhiteCliffsDrama: viewed by 4 participants
    with open(f_par_per_media, 'a') as f:
        f.write('########## SUMMARY ########## \n')
        f.write('Apart from R180_Slickrock_ReflectingPool and R180_X7_WhiteCliffsDrama, \n')
        f.write('The list of common participants are: ' + '(' + str(len(common_pars)) +' in total)'+'\n')
        f.write(listToString(common_pars))
        
    #%%
    # write media_per_par to .txt files
    f_media_per_par = Path(dir_save, 'media_per_par.txt')
    with open(f_media_per_par, 'w+') as f:
        f.write('source_file: '+ str(pkl_file)+'\n\n')
    
    with open(f_media_per_par, 'a') as f:
        for par in par_list:
            photos = media_per_par[par]
            f.write('# '+ par +' ('+ str(len(photos)) +'photos): '+'\n')
            f.write(listToString(photos)+'\n')
            f.write('\n')
        
    with open(f_media_per_par, 'a') as f:
        f.write('########## SUMMARY ########## \n')
        f.write('The following is a list of participants who viewed 64 photos in experiment ('+str(len(par_64))+' participants): '+'\n')
        f.write(listToString(par_64))
        
        
        
        
        
        
        
        
        
        
        