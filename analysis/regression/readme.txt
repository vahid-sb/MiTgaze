scripts in this folder call functions in mitgaze/regression_classification.py
-----------------------------------------------------------------------------------------------------------------------------
# script_linear_regression.py
for each participant, find mapping from media property data to heatmap data, traning/testing: 85%/15% (choose randomly) 

est = LinearRegression()
 
pickle files for media property data (24x48x14) and heatmap data(24x48) are from: 
dir_source = Path('/home/wxiao/Documents/for_Weiyi/MEDIA_grid_operation_summary')
photo used for ploting are from:
dir_media = Path('/home/wxiao/Documents/for_Weiyi/MEDIA/rescaled/screen_1920x1080 (copy)')  <-- 'b22_HS_bridge' is removed in this folder

results for # features = 24x48x14 (mode=1) are saved in:
dir_save = Path('/home/wxiao/Documents/for_Weiyi/analysis_results/linear_regression_results_#feature24x48x14')
results for # features = 14 (mode=2) are saved in:
dir_save = Path('/home/wxiao/Documents/for_Weiyi/analysis_results/linear_regression_results_#feature14')

-----------------------------------------------------------------------------------------------------------------------------
# script_svm_regression.py
for each participant, find mapping from media property data to heatmap data, traning/testing: 85%/15% (choose randomly) 

est = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
 
pickle files for media property data (24x48x14) and heatmap data(24x48) are from: 
dir_source = Path('/home/wxiao/Documents/for_Weiyi/MEDIA_grid_operation_summary')
photo used for ploting are from:
dir_media = Path('/home/wxiao/Documents/for_Weiyi/MEDIA/rescaled/screen_1920x1080 (copy)')  <-- 'b22_HS_bridge' is removed in this folder

results for # features = 14 are saved in:
dir_save = Path('/home/wxiao/Documents/for_Weiyi/analysis_results/svm_regression_results_SVR')

    
