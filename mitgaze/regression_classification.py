# -*- coding: utf-8 -*-

#%% functions used in regression
def get_train_test_data(df_heatmap, df_media_property, par):
    '''
    Extract heatmap information from df_heatmap for the input par (participant)
    Based on the media viewed by each participant, merge heatmap df and media property df
    Prepare the training and testing data

    Parameters
    ----------
    df_heatmap : pandas.dataframe
        df for all heatmaps from all participants.
    df_media_property : pandas.dataframe
        df that stores the property_array of shape 24x48x14 for all media.
    par : String
        participant number. e.g. 'P505'

    Returns
    -------
    list : [x_train, y_train, x_test, y_test, media_test]
        x_train: np.ndarray with shape (#training sample,24,48,14) <- (24,48,14) is property_array
        y_train: np.ndarray with shape (#training sample,24,48) <- (24,48) is heatmap_data
        x_test: np.ndarray with shape (#testing sample,24,48,14) <- (24,48,14) is property_array
        y_test: np.ndarray with shape (#testing sample,24,48) <- (24,48) is heatmap_data
        media_test: list of String, contains media used for testing, used as reference when ploting test and prediction results

    '''
    import pandas as pd
    import numpy as np
    
    df_heatmap_per_par = df_heatmap.loc[df_heatmap['par']==par]
    # based on the media viewed by each participant, merge heatmap df and media property df
    df_per_par = pd.merge(df_heatmap_per_par, df_media_property, how='inner', on=['media'])
    
    # determine #media for training and testing data
    num_media = len(df_heatmap_per_par.index)
    num_train = int(num_media*0.85)
    
    # determine the index of training and testing data
    idx_rand = np.random.permutation(num_media)
    idx_train = idx_rand[:num_train]
    idx_test = idx_rand[num_train:]
    
    # determine the list of training and testing media
    media_train = df_per_par['media'].iloc[idx_train].tolist()
    media_test = df_per_par['media'].iloc[idx_test].tolist()
    
    # prepare training data
    x_train = df_per_par.loc[df_per_par['media'].isin(media_train)]['property_array'].tolist()
    y_train = df_per_par.loc[df_per_par['media'].isin(media_train)]['heatmap_data'].tolist()   
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    
    # prepare test data
    x_test = df_per_par.loc[df_per_par['media'].isin(media_test)]['property_array'].tolist()
    y_test = df_per_par.loc[df_per_par['media'].isin(media_test)]['heatmap_data'].tolist()
    media_test = df_per_par.loc[df_per_par['media'].isin(media_test)]['media'].tolist()   # kepp the media order inline with x_test, y_test
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    
    return [x_train, y_train, x_test, y_test, media_test]



def get_2norm(y_test, y_pred):
    '''
    y_test and y_pred have shape (#sample, -1)
    calculate 2-norm for each sample
                                  
    Parameters
    ----------
    y_test : np.ndarray
        testing heatmaps
        shape (#sample, 1152) 1152=24x48.
    y_pred : np.ndarray
        predicted heatmaps
        shape (#sample, 1152) 1152=24x48.

    Returns
    -------
    norm2_list: float64
        2-norm for each (test, prediction) pairs.

    '''
    import numpy as np
    num_test = y_test.shape[0]
    norm2_list=[]
    for i in range(num_test):
        test = y_pred[i,:]-y_test[i,:]
        norm2_list.append(np.linalg.norm(test)) # calculate the 2-norm between test and prediction
    
    return norm2_list


def save_regression_results(data_log):
    '''
    save regression result plots

    Parameters
    ----------
    data_log: dict
    The following variables are extracted from data_log.
    
    dir_media : PosixPath
        media directory.
    dir_save : PosixPath
        directory to store results.
    par : String
        participant number.
    num_train: int
        number of training samples.
    num_test: int
        number of testing samples.
    y_test : np.ndarray
        test heatmaps.
    y_pred : np.ndarray
        preduicted heatmaps.
    media_test : list
        list of media names used for testing.
    norm_list : list
        list of 2-norm between test and prediction.

    Returns
    -------
    None.

    '''
    import matplotlib.pyplot as plt
    from pathlib import Path
    from PIL import Image
    import numpy as np
    
    # extract useful variables
    dir_media = data_log['dir_media']
    dir_save = data_log['dir_save']
    par = data_log['par']
    num_train = data_log['num_train'] 
    num_test = data_log['num_test'] 
    y_test = data_log['y_test'] 
    y_pred = data_log['y_pred']
    media_test = data_log['media_test']
    norm_list = data_log['norm_list']
    
    # reshape for ploting
    y_test = y_test.reshape(num_test,24,48)
    y_pred = y_pred.reshape(num_test,24,48)
    fig, axs = plt.subplots(num_test,3,figsize=(15,3*num_test))
    
    for i in range(num_test):
        # first img in each row: media
        img_path = Path(dir_media,media_test[i]+'.jpg')
        img_file = Image.open(img_path,'r')
        img_mode = img_file.mode
        img = plt.imread(img_path)
        if img_mode == 'RGB':
            axs[i,0].imshow(img)
        else:
            axs[i,0].imshow(img, cmap='gray')
        # second img in each row: heatmap (test)
        axs[i,1].imshow(y_test[i,:,:], cmap='gray')
        # third img in each row: heatmap (predict)
        axs[i,2].imshow(y_pred[i,:,:], cmap='gray') 
        
        # setting titles
        if i==0:
            axs[i,0].set_title('media:\n' + media_test[i])
            axs[i,1].set_title('heatmap (test)\n')
            axs[i,2].set_title('heatmap (pred)\n 2-norm between test and prediction: ' + '{:.4e}'.format(norm_list[i]))
        else:
            axs[i,0].set_title(media_test[i])
            axs[i,2].set_title('2-norm between test and prediction: ' + '{:.4e}'.format(norm_list[i]))
            
        # turn off axis
        axs[i,0].axis('off')
        axs[i,1].axis('off')
        axs[i,2].axis('off')
    
    # save plots
    dir_save_plots = Path(dir_save,'plots')
    dir_save_plots.mkdir(parents = True, exist_ok = True)
    # i.e. 'Linear_regression_P404_#train51_#test13.png'
    file_name = 'Linear_regression_'+par+'_#train'+str(num_train)+'_#test'+str(num_test)
    saving_path_png = Path(dir_save_plots,file_name + '.png')
    plt.savefig(saving_path_png)
    plt.close('all')
    
    # save data_log as .npy
    dir_save_data = Path(dir_save,'data')
    dir_save_data.mkdir(parents = True, exist_ok = True)
    saving_path_npy= Path(dir_save_data, file_name + '.npy')
    np.save(saving_path_npy,data_log)


def linear_regression_analysis(mode, df_heatmap, df_media_property, par, dir_media, dir_save):
    '''
    do linear regression analysis for the input par (participant)
    call get_train_test_data to get the corresponding training and testing samples 
    apply sklearn.linear_model.LinearRegression
    call get_2norm to calculate 2-norm distance between test and prediction heatmaps
    call save_regression_results to save the regression testing results
    
    Parameters
    ----------
    mode: int 1 or 2
        mode1: #feature=24x48x14 mode2: #feature=14
    df_heatmap : pandas.dataframe
        df for all heatmaps from all participants.
    df_media_property : pandas.dataframe
        df that stores the property_array of shape 24x48x14 for all media.
    par : String
        participant number. e.g. 'P505'
    dir_media : PosixPath
        media directory.
    dir_save : PosixPath
        directory to store results.

    Returns
    -------
    None.

    '''
    from sklearn.linear_model import LinearRegression

    # get training and testing data
    [x_train, y_train, x_test, y_test, media_test] = get_train_test_data(df_heatmap, df_media_property, par)
    
    # get the number of media involved in training and testing
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    
    if mode == 1:
        # reshape training data
        x_train = x_train.reshape(num_train,-1)
        y_train = y_train.reshape(num_train,-1)
        
        # reshape testing data
        x_test = x_test.reshape(num_test,-1)
        y_test = y_test.reshape(num_test,-1)
        
        # training and testing
        est = LinearRegression()
        est.fit(x_train, y_train)
        y_pred = est.predict(x_test)
        
    elif mode == 2:
        # reshape training data
        x_train = x_train.reshape(-1, 14)
        y_train = y_train.reshape(-1)
        
        # reshape testing data
        x_test = x_test.reshape(-1, 14)
        y_test = y_test.reshape(-1)
        
        # training and testing
        est = LinearRegression()
        est.fit(x_train, y_train)
        y_pred = est.predict(x_test)
        
    # calculate 2-norm
    y_test = y_test.reshape(num_test,-1)
    y_pred = y_pred.reshape(num_test,-1)
    norm_list = get_2norm(y_test, y_pred)
    
    data_log = {'mode': mode,
                'df_heatmap': df_heatmap, 'df_media_property': df_media_property,'par':par,
                'dir_media': dir_media, 'dir_save': dir_save,
                'num_test': num_test, 'num_train': num_train ,
                'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test, 'media_test': media_test,
                'est': est, 'y_pred': y_pred, 'norm_list': norm_list}
        
    # saving result plots
    save_regression_results(data_log)
    

def svm_regression_analysis(df_heatmap, df_media_property, par, est, dir_media, dir_save):
    '''
    do svm regression analysis for the input par (participant)
    call get_train_test_data to get the corresponding training and testing samples 
    apply sklearn.svm.SVR of 'rbf' kernel
    call get_2norm to calculate 2-norm distance between test and prediction heatmaps
    call save_regression_results to save the regression testing results
    
    Parameters
    ----------
    df_heatmap : pandas.dataframe
        df for all heatmaps from all participants.
    df_media_property : pandas.dataframe
        df that stores the property_array of shape 24x48x14 for all media.
    par : String
        participant number. e.g. 'P505'
    est: SVM estimator
    dir_media : PosixPath
        media directory.
    dir_save : PosixPath
        directory to store results.

    Returns
    -------
    None.

    '''
    # get training and testing data
    [x_train, y_train, x_test, y_test, media_test] = get_train_test_data(df_heatmap, df_media_property, par)
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    
    x_train = x_train.reshape(-1,14)
    y_train = y_train.reshape(-1)
    
    x_test = x_test.reshape(-1,14)
    y_test = y_test.reshape(-1)
    
    # training and testing
    est.fit(x_train, y_train)
    y_pred = est.predict(x_test)
    
    # calculate 2-norm
    y_test = y_test.reshape(num_test,-1)
    y_pred = y_pred.reshape(num_test,-1)
    norm_list = get_2norm(y_test, y_pred)

    data_log = {'df_heatmap': df_heatmap, 'df_media_property': df_media_property,'par':par,
                'dir_media': dir_media, 'dir_save': dir_save,
                'num_test': num_test, 'num_train': num_train ,
                'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test, 'media_test': media_test,
                'est': est, 'y_pred': y_pred, 'norm_list': norm_list}
        
    # saving result plots
    save_regression_results(data_log)
    