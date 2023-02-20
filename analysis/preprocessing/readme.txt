scripts in this folder call functions in mitgaze/extract_media_features.py
-----------------------------------------------------------------------------------------------------------------------------
#script_AOI.py
extract AOI with 'r','g','b','c','m','y' masks from media, results save as .npy, .mat

source media are from: dir_source = Path('/home/wxiao/Documents/for_Weiyi/MEDIA/AOI_scaled/screen_1920x1080')
save to: dir_save_main = Path('/home/wxiao/Documents/for_Weiyi/MEDIA_grid_operation_H24_W48/AOI')

-----------------------------------------------------------------------------------------------------------------------------	
#script_color.py
extract 'r','g','b','h','s','v' information from media, results save as .npy, .mat

source media are from: dir_source = Path('/home/wxiao/Documents/for_Weiyi/MEDIA/rescaled/screen_1920x1080 (copy)')  <-- 'b22_HS_bridge' is removed in this folder
save to: dir_save_main = Path('/home/wxiao/Documents/for_Weiyi/MEDIA_grid_operation_H24_W48/color')

-----------------------------------------------------------------------------------------------------------------------------
#script_contrast.py
apply edge detection algorithm on media, results save as .npy, .mat

source media are from: dir_source = Path('/home/wxiao/Documents/for_Weiyi/MEDIA/rescaled/screen_1920x1080 (copy)')  <-- 'b22_HS_bridge' is removed in this folder
save to: dir_save_main = Path('/home/wxiao/Documents/for_Weiyi/MEDIA_grid_operation_H24_W48/contrast')

-----------------------------------------------------------------------------------------------------------------------------
#script_light_intensity.py
calculate light intensity of each media, results save as .npy, .mat

source media are from: dir_source = Path('/home/wxiao/Documents/for_Weiyi/MEDIA/rescaled/screen_1920x1080 (copy)')  <-- 'b22_HS_bridge' is removed in this folder
save to: dir_save_main = Path('/home/wxiao/Documents/for_Weiyi/MEDIA_grid_operation_H24_W48/light_intensity')

-----------------------------------------------------------------------------------------------------------------------------
#script_plot_npy.py
plot and save the .npy files generated above as .png files
Note: It takes a long time to run, even with parallel processing

-----------------------------------------------------------------------------------------------------------------------------
#script_media_data_to_df.py
put information about AOI, color, contrast, light intensity into dataframe, save as pickle files:
'media_grid_H24_W48_AOI.p'
'media_grid_H24_W48_color.p'
'media_grid_H24_W48_contrast.p'
'media_grid_H24_W48_light_intensity.p'

for AOI, source media are from: dir_source = Path('/home/wxiao/Documents/for_Weiyi/MEDIA/AOI_scaled/screen_1920x1080')
for color, contrast, light intensity, media are from: dir_source = Path('/home/wxiao/Documents/for_Weiyi/MEDIA/rescaled/screen_1920x1080 (copy)')  <-- 'b22_HS_bridge' is removed in this folder
dir_save = Path('/home/wxiao/Documents/for_Weiyi/MEDIA_grid_operation_summary')

-----------------------------------------------------------------------------------------------------------------------------
#script_heatmap_data_to_df.py
put the heatmap into dataframe and save as pickle file: 'HEATMAP_DATA_SUMMARY.p'

source files are from: dir_source = Path('/home/wxiao/Documents/for_Weiyi/GAZE_COUNT_GRID_48_24')
save to: dir_save = Path('/home/wxiao/Documents/for_Weiyi/MEDIA_grid_operation_summary')

-----------------------------------------------------------------------------------------------------------------------------
#script_merge_df_grid_data.py
merge dataframe that stores light_intensity, color, contrast and AOI information, save as pickle files: 'media_grid_H24_W48_all.p'

source files are from: dir_source = '/home/wxiao/Documents/for_Weiyi/MEDIA_grid_operation_summary'
save to: dir_save=dir_source

-----------------------------------------------------------------------------------------------------------------------------
#script_df_media_property_24x48x14.py
extract 14=1(light_intensity)+6(color)+1(contrast)+6(AOI) channels of information from 'media_grid_H24_W48_all.p', results save as .npy files for each media
and save total information as pickle files: 'media_grid_H24_W48_media_property.p'

source files are from: dir_source = Path('/home/wxiao/Documents/for_Weiyi/MEDIA_grid_operation_summary')
save .npy files to: dir_save = Path('/home/wxiao/Documents/for_Weiyi/MEDIA_property_array_24x48x14')
save dataframe to: dir_save = dir_source

