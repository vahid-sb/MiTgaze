
#script_kmeans_clustering.py
for each media, do kmeans clustering of the heatmap data across participants.

heatmap data are from: pkl_file = Path('/home/wxiao/Documents/for_Weiyi/MEDIA_grid_operation_summary/HEATMAP_DATA_SUMMARY.p')
media for plotting are from: dir_media = Path('/home/wxiao/Documents/for_Weiyi/MEDIA/rescaled/screen_1920x1080 (copy)')
clustering results are saved to: dir_save = Path('/home/wxiao/Documents/for_Weiyi/analysis_results/kmeans_clustering_results')
    
#script_pair_diff_clustering.py
some media belong to the 3 categories of media pairs, namely the color-gray pairs, the calibration pairs and the flipping pairs.
for each media pair, calculate the difference between the heatmap data, and do clustering.  

heatmap data are from: pkl_file = Path('/home/wxiao/Documents/for_Weiyi/MEDIA_grid_operation_summary/HEATMAP_DATA_SUMMARY.p')
media for plotting are from: dir_media = Path('/home/wxiao/Documents/for_Weiyi/MEDIA/rescaled/screen_1920x1080 (copy)')
clustering results are saved to: dir_save = Path('/home/wxiao/Documents/for_Weiyi/analysis_results/kmeans_clustering_results_media_pairs')
