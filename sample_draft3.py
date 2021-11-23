## Sample code to estimate the horizontal velocity
## developed by Ryohtaroh T. Ishikawa
## Reference: Ishikawa et al. (2021) accepted to A&A
import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras.backend import tensorflow_backend
## python version     3.5.2
## keras version      2.3.1
## tensorflow version 1.15.0

if __name__=='__main__':
    ## GPU setting
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    tensorflow_backend.set_session(session)
    
    ## size of the data
    Nsample_test = 1
    Nx = 128
    Ny = 128
    Nxy = Nx * Ny
    sx= (Nsample_test,Nx,Ny,3,2)## 3 consective frames and 2 physical quantities (Uz & T)
    sy= (Nsample_test,Nx,Ny,1)
    ## Readout the data
    X_data = np.memmap(filename='./X_test_memmap_for_open.dat', dtype=np.float32, mode='r',shape=sx)
    y_data = np.memmap(filename='./y_test_memmap_for_open.dat', dtype=np.float32, mode='r',shape=sy)
    
    ## Readout the optimized model
    model = keras.models.load_model('save_model_for_MURaM')
    
    ## calculation with the model
    predict0 = model.predict(X_data)
    #print(np.shape(predict0))
    
    ## plot
    ## Note that the all physical quantities are generalized.
    plot_tf = False## True for plotting the results
    if plot_tf:
        fig = plt.figure(figsize=(10,8))
        ## (Input) vertical velocity
        ax1 = fig.add_subplot(2,2,1)
        img1 = ax1.imshow(-1.*X_data[0,:,:,1,0],cmap='seismic',origin='lower',vmin=-3,vmax=3)
        cbar1 = fig.colorbar(img1,ax=ax1,aspect=20,pad=0.08,shrink=0.95,orientation='vertical')
        ax1.set_title('Uz Input')
        ax1.set_ylabel('Y [pixel]')
        ## (Input) Temperature
        ax2 = fig.add_subplot(2,2,2)
        img2 = ax2.imshow(X_data[0,:,:,1,1],cmap='gist_gray',origin='lower',vmin=-3,vmax=3)
        cbar2 = fig.colorbar(img2,ax=ax2,aspect=20,pad=0.08,shrink=0.95,orientation='vertical')
        ax2.set_title('T Input')
        ## (Ground Trueth) horizontal velocity
        ax3 = fig.add_subplot(2,2,3)
        img3 = ax3.imshow(y_data[0,:,:,0],origin='lower',vmin=-3,vmax=3)
        cbar3 = fig.colorbar(img3,ax=ax3,aspect=20,pad=0.08,shrink=0.95,orientation='vertical')
        ax3.set_title('Uy Answer')
        ax3.set_xlabel('X [pixel]')
        ax3.set_ylabel('Y [pixel]')
        ## (Prediction) horizontal velocity
        ax4 = fig.add_subplot(2,2,4)
        img4 = ax4.imshow(predict0[0,:,:,0],origin='lower',vmin=-3,vmax=3)
        cbar4 = fig.colorbar(img4,ax=ax4,aspect=20,pad=0.08,shrink=0.95,orientation='vertical')
        ax4.set_title('Uy Estimation')
        ax4.set_xlabel('X [pixel]')
        plt.show()
    else:
        pass
    
    print('END')
#EOF
