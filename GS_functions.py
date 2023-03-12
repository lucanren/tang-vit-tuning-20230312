'''@Author: Shang Gao  * @Date: 2023-01-18 18:15:07  * @Last Modified by:   Shang Gao  * @Last Modified time: 2023-01-18 18:15:07 '''
import os
import sys
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
import PIL
import torch
import shutil
# from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
# import Ranger
# from audtorch.metrics.functional import pearsonr
from pytictoc import TicToc
import imshowtools 
# sys.path.append(where you put this file),from GS_functions import GF
# sys.path.append('/user_data/shanggao/tang/'),from GS_functions import GF
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
t = TicToc()
import numpy as np
import scipy
import cv2
import copy
import itertools

class GF:
    def all_pairs_to_corrmap(NCall,pair_list,numofneurons,digvalue=None):
        '''
        NCall=NC_all_pairs: len(NCall) == len(pair_list), this is the correlations for each pair_list (allcomb return by  GF.find_pair_combinations)
        e.g. NCall.shape(44000,), pair_list.shape(44000,), numofneurons 299
        Out:
        corrmap 
        '''
        NCall[np.isnan(NCall)]=0 ## nan means one of both the neurons rsp are lower than filteredrsp, (nan due to [])
        corrmap_midvar=np.zeros((numofneurons,numofneurons))
        corrmap=np.zeros((numofneurons,numofneurons))
        assert len(NCall)==len(pair_list)
        if digvalue is None:
            digvalue2=1
        else:
            digvalue2=digvalue
        for dig in range(numofneurons):
            corrmap[dig,dig]=digvalue2
        for pl in range(len(pair_list)):
            posh,posw=pair_list[pl]
            corrmap_midvar[posh,posw]=NCall[pl]
        corrmap=corrmap+corrmap_midvar+corrmap_midvar.T
        return corrmap
    def vectindex_to_location(vectidx,H):
        '''
        vectidx is vector indexing, column first, it means from topleft to topbottom and second top coloumn;
        this is coloumn first, so only need to input H;
        vectidex start from 0, return results are also using 0 indexing.
        vectidx can be scalar or 1d vector
        '''
        newW=vectidx//H
        newH=vectidx%H
        return newH,newW
    def matlab_vector_indexing(twodmatrix,onedvector,assignvalue=None):
        '''
        Input 2d matrix and 1d vector to index
        this function should perform consistent matrix manipulation in MATLAB
        if assignvalue=None, return the matrix value by using the 1dvector indexing 
        if assignvalue=value_you_defined, it will replace the index part in matrix as the assignvalue
        '''
        onedvector=np.array(onedvector)
        onedvector=onedvector.flatten()
        assert len(twodmatrix.shape)==2, 'twodmatrix should be 2d'
        H,W=twodmatrix.shape
        twodmatrix=twodmatrix.flatten(order='F') #F is in coloumn order
        if assignvalue is None:
            finalmatrix=twodmatrix[onedvector]
        else:
            twodmatrix[onedvector]=assignvalue
            finalmatrix=twodmatrix.reshape((H,W)).T
        return finalmatrix

    def find_pair_combinations(numbersets,pairnum=2):
        allcomb=tuple(itertools.combinations(numbersets, pairnum))
        return allcomb
    def tangcnn_parameters(name=None,sitename=None,print_hint=1):
        '''
        get parameters
        e.g. GF.tangcnn_parameters()['num_of_neurons']
        '''
        if print_hint:
            print('\n1.everyday_training_label\n2.num_of_training_samples\n3.num_of_neurons\n4.loss_name')
        loss_name = [
        "MSE",
        "mix_corr_MAE",
        "MAE",
        "MSEN",
        "VE",
        "VE2",
        "VE3",
        "VE4",
        "R2_1",
        "R2_2",
    ]
        everyday_training_label = {
            "M1S1": [(0, 19000), (0, 24000), (0, "_all")],
            "M1S2": [(0, 14700), (0, 30700), (0, "_all")],
            "M1S3": [(0, 9670), (0, 12670), (0, 19670), (0, 25670), (0, "_all")],
            "M2S1": [(0, 19000), (0, 29000), (0, 39000), (0, "_all")],
            "M2S2": [(0, 19000), (0, 28900), (0, "_all")],
            "M3S1": [(0, 9000), (0, 18900), (0, 27900), (0, "_all")],
        }
        num_of_training_samples = {
            "M1S1": 34000,
            "M1S2": 50700,
            "M1S3": 31670,
            "M2S1": 49000,
            "M2S2": 48900,
            "M3S1": 34900,
        }
        num_of_neurons = {
            "M1S1": 302,
            "M1S2": 330,
            "M1S3": 175,
            "M2S1": 299,
            "M2S2": 259,
            "M3S1": 324,
        }
        if sitename==None or name==None:
            all_dict={}
            all_dict['loss_name']=loss_name
            all_dict['everyday_training_label']=everyday_training_label
            all_dict['num_of_training_samples']=num_of_training_samples
            all_dict['num_of_neurons']=num_of_neurons
            return all_dict
        # with model training
        elif name=='everyday_training_label':
            return everyday_training_label[sitename.upper()]
        elif name=='num_of_training_samples':
            return num_of_training_samples[sitename.upper()]
        elif name=='num_of_neurons':
            return num_of_neurons[sitename.upper()]



    def group_barchart(datamat,xticks_name,legend_name,xlabelname,ylabelname, titlename, dpi=150):
        '''
        group number = datamat's column number. Bar number in each group = datamat's row number
        '''
        assert len(datamat.shape)==2
        w = 1.0
        num_col = datamat.shape[1] #stats.shape[1] - 1
        num_rows = datamat.shape[0] #stats.shape[0]
        assert num_col==len(xticks_name), 'column number must equal to len xticks_name'
        first_tick = int(np.ceil((num_rows*w/2))) 
        gap = num_rows*w + 1
        x = np.array([first_tick + i*gap for i in range(num_col)])
        colors = plt.cm.get_cmap('inferno',num_rows)
        fig,ax = plt.subplots(1,1, figsize=(10,10),dpi=dpi)
        b = []
        for i in range(num_rows):
            b.append(ax.bar(x - (i - num_rows/2 + 0.5)*w, 
                    datamat[i,:], 
                    width=w, 
                    color=colors(i), 
                    align='center', 
                    edgecolor = 'black', 
                    linewidth = 1.0, 
                    alpha=0.5))
        ax.legend([b_ for b_ in b], 
                legend_name, 
                ncol = 3, 
                loc = 'best', 
                framealpha = 0.1)

        ax.set_ylabel(ylabelname)
        ax.set_xlabel(xlabelname)
        ax.set_title(titlename)
        ax.set_xticks(x)
        ax.set_xticklabels(xticks_name)
        for i in range(num_rows):
            ax.bar_label(b[i], 
                        padding = 3, 
                        label_type='edge', 
                        rotation = 'horizontal')

    def find_matrix_max_index(matrix):
        ind = np.unravel_index(np.argmax(matrix, axis=None), matrix.shape) 
        return ind
    def checkKey(dic, key): 
        if key in dic:
            judge=1
        else:
            judge=0
        return judge
    def matrix_corrcoef(Matrix,vector,cal_mode=0,mmcal_dim=1,mode='numpy',device='cpu'):
        '''
        in:
        cal_mode=0
        Matrix: (N * P)
        vector: (N * 1)/(N,)
        out:
        numpy array corr: (P,) or torch tensor shape([P])
        
        in:
        cal_mode=1 -> input mmcal_dim
        Matrix: (N * P)
        vector: (N * P)
        if dim=1, Out: (N,) , compute corr with a[i,:], b[i,:]
        
        mode: you can use pytorch tensor to use gpu, which will be faster

        '''
        assert vector.shape[0]==Matrix.shape[0]
        if cal_mode==0:
            if len(vector.shape)==1:
                vector=vector.reshape((len(vector),1))
            if mode=='numpy' or mode==None:
                vector_avg=np.mean(vector,axis=0)
                Matrix_avg=np.mean(Matrix,axis=0,keepdims=True)
                upper_=np.sum((Matrix-Matrix_avg)*(vector-vector_avg),axis=0)
                lower_=np.sqrt(np.sum((Matrix-Matrix_avg)**2,axis=0)*np.sum((vector-vector_avg)**2,axis=0))
                corr=upper_/lower_
            else:
                vector=torch.tensor(vector)
                Matrix=torch.tensor(Matrix)
                vector=vector.to(device)
                Matrix=Matrix.to(device)
                vector_avg=torch.mean(vector,axis=0)
                Matrix_avg=torch.mean(Matrix,axis=0,keepdims=True)
                upper_=torch.sum((Matrix-Matrix_avg)*(vector-vector_avg),axis=0)
                lower_=torch.sqrt(torch.sum((Matrix-Matrix_avg)**2,axis=0)*torch.sum((vector-vector_avg)**2,axis=0))
                corr=upper_/lower_
                corr=corr.cpu().detach()
        else:
            # print(f'input shape (N*P), if mmcal_dim=1, output shape:(N,)')
            vector_avg=np.mean(vector,axis=mmcal_dim,keepdims=True)
            Matrix_avg=np.mean(Matrix,axis=mmcal_dim,keepdims=True)
            upper_=np.sum((Matrix-Matrix_avg)*(vector-vector_avg),axis=mmcal_dim)
            lower_=np.sqrt(np.sum((Matrix-Matrix_avg)**2,axis=mmcal_dim)*np.sum((vector-vector_avg)**2,axis=mmcal_dim))
            corr=upper_/lower_

        return corr.flatten()


    def round_number(vector,digit=2):
        vector=vector.flatten()
        nums=len(vector)
        newv=np.zeros(vector.shape)
        for i in range(nums):
            newv[i]=round(vector[i],digit)
        return newv
    class SCR: # sparse code rsp
        def batch_crop_resize(image,cropvalue=None,finalsize=None):
            '''
            IN: image.shape->(BHW),first crop then resize
            '''
            print('!!! First crop then resize')
            assert len(image.shape)==3
            numofimgs,H,W=image.shape
            if cropvalue is None:
                cropvalue=16
            if finalsize is None:
                finalsize=16
            crop_resize_img=np.zeros((numofimgs,finalsize,finalsize))
            for i in range(image.shape[0]):
                img1=image[i,:,:]
                croppedimg=GF.center_crop_(image=img1,cropsize=(cropvalue,cropvalue))
                resized = cv2.resize((croppedimg), (finalsize,finalsize), interpolation = cv2.INTER_CUBIC)
                crop_resize_img[i,:,:]=resized
            return crop_resize_img
        def create_rotated_images(pics, degrees_num=18):
            '''
            In: val_pics->(1000,50,50,1)
            Out: all_pics with degree num -> (1000,50,50,degrees_num*2), 2 polarity

            '''
            assert len(pics.shape)==4
            each_degree=360/degrees_num
            val_pics=pics.squeeze() # (1000,50,50)
            num_of_imgs,H,W=val_pics.shape[0],val_pics.shape[1],val_pics.shape[2]

            types=degrees_num*2 # 2 polarity
            all_imgs=np.zeros((num_of_imgs,H,W,types))
            degree_list=np.arange(0,360,each_degree)
            assert degree_list.shape[0]==degrees_num
            type=0
            for degree in degree_list:
                rotatedimg=GF.SCR.rotate_or_reverse(image=val_pics,mode='rotate',degree_num=degree)
                all_imgs[:,:,:,type]=rotatedimg
                all_imgs[:,:,:,type+1]=GF.SCR.rotate_or_reverse(image=rotatedimg,mode='reverse')
                type+=2
            print('Orient type',type,'\nTotal types with reverse',types)
            assert(types==type)
            return all_imgs

        def rotate_or_reverse(image,mode='rotate',degree_num=None):
            '''
            In: image->(BHW) or (HW); mode:rotate/reverse; degree_num: degree(rotate), reverse will use 1-image pixels

            '''
            mode=mode.lower()
            assert mode=='rotate' or mode=='reverse'

            num_dim=len(image.shape)
            assert num_dim==3 or num_dim==2
            if num_dim==3:
                num_of_imgs,H,W=image.shape
                newimage=np.zeros(image.shape)
                for i in range(num_of_imgs):
                    if mode=='rotate':
                        newimage[i,:,:]=scipy.ndimage.rotate(image[i,:,:],angle=degree_num,reshape=False)
                    elif mode=='reverse':
                        newimage[i,:,:]=1-image[i,:,:]
            elif num_dim==2:
                H,W=image.shape
                newimage=np.zeros(image.shape)
                if mode=='rotate':
                    newimage=scipy.ndimage.rotate(image,angle=degree_num,reshape=False)
                elif mode=='reverse':
                    newimage=1-image
            return newimage

    def get_VE(pred,real):
        pred=pred.flatten()
        real=real.flatten()
        assert pred.shape==real.shape
        VE=1 - np.var(real-pred) / (np.var(real))
        return VE
    def matrix_VE(Matrix,vector):
        '''
        In
        Matrix: (N * P)
        vector: (N * 1)
        Out
        VE vector: (P,)
        '''
        assert vector.shape[0]==Matrix.shape[0]
        if len(vector.shape)==1:
            vector=vector.reshape((len(vector),1))
        var_vector=np.var(vector)
        diff=vector-Matrix #(P * N)
        VE_vector=1-np.var(diff,axis=0)/(var_vector)
        return VE_vector.flatten()
    def get_model_rsp_cphw3(model,img_subp_mat,batch_size=512,device='cpu',norm_1=0, mode='torch'):
        '''
        img_subp_mat shape: (batchnum, 1, subcropsize, subcropsize), e.g. (44540, 1, 50, 50)
        '''
        mode=mode.lower()
        if mode=='torch':
            if norm_1==1:
                img_subp_mat=GF.norm_to_1(img_subp_mat)
            img_subp_mat=torch.tensor(img_subp_mat,dtype=torch.float)
            assert len(img_subp_mat.shape)==4
            all_rsp=[]
            valpics=GF.ImageDataset_cphw3(img_subp_mat=img_subp_mat)
            val_loader=DataLoader(valpics,batch_size=batch_size,shuffle=False)
            for num,batch_pics in enumerate(val_loader):
                with torch.no_grad():
                    print(num)
                    model=model.to(device)
                    batch_pics=batch_pics.to(device)
                    rsp=model(batch_pics)
                    all_rsp.append(rsp.detach().cpu().numpy())
            all_rsp=np.vstack(all_rsp)
        elif mode=='tf':
            pass
        else:
            raise RuntimeError('\nPlease input mode:\n1.torch\n2.tf')

        return all_rsp

    class ImageDataset_cphw3(Dataset):
        def __init__(self,img_subp_mat ):
            """
            cell_start: start from 1
            mode=num_neurons/startend
            """
            self.data=img_subp_mat 
        def __len__(self):
            return self.data.shape[0]  # number of images
        def __getitem__(self, index):
            img = self.data[index]
            return img
    def compute_oimg_size_for_subparts(
        num_of_subparts_HW=(2, 2), subparts_size=16, stride=1
    ):
        """
        use with GF.crop_img_subparts
        This will use number of crop subparts you need to compute what the size of orginal image should be.
        subparts_size is like a kernel size
        """
        size_H = (num_of_subparts_HW[0] - 1) * stride + subparts_size
        size_W = (num_of_subparts_HW[1] - 1) * stride + subparts_size
        size = (size_H, size_W)
        return size
    def crop_img_subparts(img,crop_size=50,stride=1):
        '''
        responsemap,rspmap
        Input: img->grayscale; crop_size->crop size of subpart image you want; stride-> cropping stride. 
        crop_size is the model inpute size
        Output: return (num_of_subparts,crop_size,crop_size)
        Note: cropping is from: topleft -> topright -> bottomleft -> bottomright
        Num of block: (Imgsize-crop_size)/stride+1
        '''
        if isinstance(crop_size,int):
            Hcrop,Wcrop=crop_size,crop_size
        elif isinstance(crop_size,tuple):
            Hcrop,Wcrop=crop_size[0],crop_size[1]
        assert len(img.shape)==2
        H,W=img.shape
        
        # do not need this
        # assert (H,W)>(100,100)
        # if stride&1 != H&1:
        #     img=img[:-1,:]
        # if stride&1 != W&1:
        #     img=img[:,:-1]
        H,W=img.shape
        print('Image shape (H,W):',(H,W))
        H_num,W_num=GF.compute_num_ofsubpart((H,W),crop_size,stride)
        print('Number of blocks:',(H_num,W_num))
        H_stridelist=list(range(0,H,stride))
        W_stridelist=list(range(0,W,stride))
        # print(H_stridelist)
        img_subp_mat=[]
        for i in range(H_num):
            for j in range(W_num):
                # print(i)
                x0,x1=H_stridelist[i], H_stridelist[i]+Hcrop
                y0,y1=W_stridelist[j], W_stridelist[j]+Wcrop
                # print(y0) 
                img_R=img[x0:x1, y0:y1]
                # print(img_R.shape)
                img_subp_mat.append(img_R)
        img_subp_mat=np.stack(img_subp_mat)
        Number_of_blocks=(H_num,W_num)
        return img_subp_mat,Number_of_blocks

    def compute_num_ofsubpart(imgshape,crop_size,stride):
        '''
        crop_size is like a kernel size. 
        '''
        if isinstance(crop_size,int):
            Hcrop,Wcrop=crop_size,crop_size
        elif isinstance(crop_size,tuple):
            Hcrop,Wcrop=crop_size[0],crop_size[1]
        if isinstance(imgshape,int):
            imgshape=(imgshape,imgshape)
        assert len(imgshape)==2
        assert isinstance(imgshape,tuple)
        H,W=imgshape[0],imgshape[1]
        H_num=int((H-Hcrop)/stride+1)
        W_num=int((W-Wcrop)/stride+1)
        return H_num,W_num
    def slice_max(oneDarray,slice_num):
        '''
        In: oneDarray, slice_num=2
        Example: a=[1,2,54,5,6,7], slice_num=2, Out=[0,2,54,0,0,7]
        '''
        oneDarray=oneDarray.flatten()
        oneDarray=oneDarray.reshape((-1,slice_num))
        print('oneDarray reshape:',oneDarray,'\n')
        print('oneDarray shape',oneDarray.shape,'\n')
        slice_max_v=np.max(oneDarray,axis=1)
        NewArray=np.zeros(oneDarray.shape)
        for i in range(len(slice_max_v)):
            max_1row=slice_max_v[i]
            Array_1row=oneDarray[i,:]
            bool_idx=(Array_1row>=max_1row)+0
            bool_idx_final=bool_idx
            if len(np.where(bool_idx==1)[0])>=2:
                print('exist same value, choose the first one...')
                First_idx=np.where(bool_idx==1)[0][0]
                bool_idx2=np.zeros(bool_idx.shape)
                bool_idx2[First_idx]=1
                bool_idx_final=bool_idx2
            NewArray[i,:]=bool_idx_final*max_1row
            
        print('NewArray',NewArray)
        return NewArray.flatten()
        
    def make_mask(ic, jc,RFsize, img_size=(50,50),criteria='zero',gaussian_radius=11,gaussian_sigma=2.2):
        '''
        img_size:tuple
        criteria='zero'/'gaussian'/
        '''
        # set up a mask
        m = 0
        mask = np.zeros(img_size, dtype=np.float32)
        temp = np.zeros(img_size).astype(np.float32)
        for i in range(img_size[0]):
            for j in range(img_size[1]):
                if (i - ic) ** 2 + (j - jc) ** 2 < (RFsize + 1) ** 2:
                    temp[i, j] = 1
        if criteria.lower()=='gaussian':
            mask= cv2.GaussianBlur(1 - temp, (gaussian_radius, gaussian_radius), gaussian_sigma)
        elif criteria.lower()=='zero':
            mask=1-temp
        return mask
    def conv2(img,kernel,mode='same'):
        """
        From: https://www.codegrepper.com/code-examples/python/conv2+python
        Emulate the function conv2 from Mathworks.
        Usage:
        z = conv2(img,kernel,mode='same')
        - Support other modes than 'same' (see conv2.m)
        """
        img=np.array(img,dtype=np.float)
        kernel=np.array(kernel,dtype=np.float)
        if not(mode == 'same'):
            raise Exception("Mode not supported")

        # Add singleton dimensions
        if (len(img.shape) < len(kernel.shape)):
            dim = img.shape
            for i in range(len(img.shape),len(kernel.shape)):
                dim = (1,) + dim
            img = img.reshape(dim)
        elif (len(kernel.shape) < len(img.shape)):
            dim = kernel.shape
            for i in range(len(kernel.shape),len(img.shape)):
                dim = (1,) + dim
            kernel = kernel.reshape(dim)

        origin = ()

        # Apparently, the origin must be set in a special way to reproduce
        # the results of scipy.signal.convolve and Matlab
        for i in range(len(img.shape)):
            if ( (img.shape[i] - kernel.shape[i]) % 2 == 0 and
                img.shape[i] > 1 and
                kernel.shape[i] > 1):
                origin = origin + (-1,)
            else:
                origin = origin + (0,)

        z = scipy.ndimage.filters.convolve(img,np.flip(kernel), mode='constant', origin=origin)

        return z
    def goodsubplot(name_list,x_list,y_list,subplotsize=(3,2),subtitlename_list=None,bigtitlename=None,bigxlabel=None,bigylabel=None):
        nl='\n'
        fig, axs = plt.subplots(subplotsize[0],subplotsize[1], figsize=(15, 15), sharex=True, sharey=True) #这个可以画个好尺寸
        axs=axs.flatten()     
        for i in range(len(name_list)):
            axs[i].scatter(x_list[i],y_list[i],s=20, c="k", marker="o")
            axs[i].set_title(subtitlename_list[i])
            plt.tight_layout()
            plt.suptitle(bigtitlename,fontsize=23)
            fig.supxlabel(bigxlabel,fontsize=20)
            fig.supylabel(bigylabel,fontsize=20) #大的子标题

    def pie_chart(labels,sizes,shadow=0,startangle=90):
        '''
        labels:list or np array of strings.
        sizes: list or np array of numbers.
        '''
        assert len(labels) 
        fig1, ax1 = plt.subplots(dpi=100)
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                shadow=shadow, startangle=startangle)
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax1.axis('equal')  
        plt.tight_layout()

    def img2matrix(imgmainpath,suffix='.png'):
        '''
        make sure all imgs are the same size
        '''
        pathlist=GF.filelist_suffix(imgmainpath,suffix)
        im_shape=np.array(PIL.Image.open(f"{imgmainpath}/{pathlist[0]}").convert('L')).shape[0]
        im_matrix=np.zeros((len(pathlist),im_shape,im_shape))
        for i in range(len(pathlist)):
            # print('Change img:',i+1)
            im_matrix[i,:,:]= np.array(PIL.Image.open(f"{imgmainpath}/{pathlist[i]}").convert('L'))
        
        return im_matrix
    def center_crop_(image,cropsize=(20,20),return_1_var=1):
        ''' 
        if image dimension is even, and crop is odd or image odd crop even. We will throw away the last row or(and) last column
        '''
        assert len(image.shape)==2
        if isinstance(cropsize,int):
            cropsize=(cropsize,cropsize)
        assert isinstance(cropsize,tuple)
        newimage=copy.copy(image)
        H,W=image.shape
        croph,cropw=cropsize
        if ((H%2)!=0 and (croph%2)==0) or ((H%2)==0 and (croph%2)!=0):
            print('throw away last row')
            # newimage=newimage[:-1,:] # with this or not will be the same, because we are using int() below. 
            # By using int, it actually has the same effect as throw away last row or column
        if ((W%2)!=0 and (cropw%2)==0) or ((W%2)==0 and (cropw%2)!=0):
            print('throw away last column')
            # newimage=newimage[:,:-1]
        start_h,start_w=int((H-croph)/2), int((W-cropw)/2) # 
        cropHidx=(start_h,start_h+croph)
        cropWidx=(start_w,start_w+cropw)
        crop_img = newimage[cropHidx[0]:cropHidx[1],cropWidx[0]:cropWidx[1]]
        if return_1_var==1:
            return crop_img
        else:
            return crop_img,cropHidx,cropWidx
    def show_imgs_in1Page(img_matrix,cmap='gray',showsize=(10,10),columns=None,rows=None,padding=False,title=None):
        '''
        shape: (numbers,H,W,C) or (numbers,H,W)
        from imshowtools import imshow
        img_matrix(B,H,W)
        fullversion: imshowtools.imshow(*img_matrix,cmap='gray',size=(15,15),columns=18,rows=rows,padding=padding,title=title,return_image=False)
        (use this) e.g. imshowtools.imshow(*img_matrix,cmap='gray',size=(15,15),columns=18,return_image=False)
        '''
        pass


    def filelist_suffix(filepath, suffix=None):
        """
        this is to find all the file with certain suffix, and order it.
        REMEMBER: filenames has to be numbers (beside the suffix)
        """
        filelist = os.listdir(filepath)
        assert isinstance(suffix, (str, tuple, type(None)))
        if suffix != None:
            filelist = [f for f in filelist if f.endswith((suffix))]
            try:
                
                if suffix=='.hdf5':
                    filelist.sort(key=lambda x: int(x[4:].replace(suffix,'')))
                else:
                    filelist.sort(key=lambda x: int(x[:-4]))
            except:
                filelist.sort()
            print("There are ", len(filelist), " files in this directory")
        else:
            try:
                filelist.sort(key=lambda x: int(x[:-4]))
            except:
                filelist.sort()
            print("There are ", len(filelist), " files in this directory")
            
        filelist_final = np.array(filelist)
        return filelist_final

    def gen_range(start, stop, step,mode='accumulate'):
        """
        Generate list
        mode=accumulate/separate
        * For function gen_list_tuple
        """
        mode=mode.lower()
        if mode not in ('accumulate','separate'):
            raise RuntimeError('mode=accumulate/separate')
        if mode=='accumulate':
            current = start
            while current < stop:
                next_current = current + step
                if next_current < stop:
                    yield (int(start), int(next_current))
                else:
                    yield (int(start), int(stop))
                current = next_current
        elif mode=='separate':
            current = start
            while current < stop:
                next_current = current + step
                if next_current < stop:
                    yield (int(current), int(next_current))
                else:
                    yield (int(current), int(stop))
                current = next_current

    def gen_list_tuple(start, stop, step,mode='accumulate'):
        '''
        mode=accumulate/separate; 
        default: accumulate
        '''
        a = []
        for i in GF.gen_range(start, stop, step,mode):
            a.append(i)
        return a

    # def gen_range(start, stop, step):
    #     """Generate list"""
    #     current = start
    #     while current < stop:
    #         next_current = current + step
    #         if next_current < stop:
    #             yield (current, next_current)
    #         else:
    #             yield (current, stop)
    #         current = next_current

    def norm_to_1(imagemat):
        """
        In: Input shape should be 4[BHWC or BCHW] or 3[CHW or HWC] or 2[HW] or 1[vector], tensor or numpy arrary.
        Out: Norm to 1 version , Batch and Channel seperate
        """
        assert (
            len(imagemat.shape) == 2
            or len(imagemat.shape) == 1
            or len(imagemat.shape) == 4,
            len(imagemat.shape) == 3,
        ), "Input shape should be 4[BHWC or BCHW] or 3[CHW or HWC] or 2[HW] or 1[vector]"
        assert isinstance(imagemat, torch.Tensor) or isinstance(
            imagemat, np.ndarray
        ), "input data should be torch tensor or numpy array"
        grad_mode = None
        if isinstance(imagemat, torch.Tensor):
            if imagemat.requires_grad:
                grad_mode = "True"
                GG = imagemat.grad
            else:
                grad_mode = "False"
                GG = None

            imagemat_new = imagemat.detach().clone()  # .detach().clone()
            print("---------------------------")
            imagemat_new = torch.tensor(imagemat_new, dtype=torch.float)  ### new line
        else:
            imagemat_new = imagemat.copy()
            imagemat_new = np.array(imagemat_new, dtype=np.float32)  ### new line
        if len(imagemat_new.shape) == 3:
            C, H, W = imagemat_new.shape
            assert (
                C == 1 or C == 3 or W == 1 or W == 3
            ), "Input should be CHW or HWC, and channel can only be 1 or 3"
            if C == 1 or C == 3:
                new_img = GF.channel_norm1(imagemat_new, mode="CHW")
            elif W == 1 or W == 3:
                new_img = GF.channel_norm1(imagemat_new, mode="HWC")
            else:
                raise RuntimeError("Check input")

        if len(imagemat_new.shape) == 2 or len(imagemat_new.shape) == 1:
            if imagemat_new.max() == imagemat_new.min():
                new_img = imagemat_new
            else:
                new_img = (imagemat_new - imagemat_new.min()) / (
                    imagemat_new.max() - imagemat_new.min()
                )

        if len(imagemat_new.shape) == 4:
            B, H, W, C = imagemat_new.shape
            assert H == 1 or H == 3 or C == 1 or C == 3, "Input should be BHWC or BCHW"
            if C == 1 or C == 3:
                mode = "HWC"
                for i in range(B):
                    imagemat_new[i, :, :, :] = GF.channel_norm1(
                        imagemat_new[i, :, :, :], mode=mode
                    )
                new_img = imagemat_new
            elif H == 1 or H == 3:
                mode = "CHW"
                for i in range(B):

                    imagemat_new[i, :, :, :] = GF.channel_norm1(
                        imagemat_new[i, :, :, :], mode=mode
                    )
                new_img = imagemat_new
            else:
                raise RuntimeError("Check whether your image channel is 1 or 3")
        if grad_mode == "True":
            new_img.requires_grad = True
            new_img.grad = GG
            # new_img = torch.tensor(new_img, requires_grad=True)
        return new_img

    def channel_norm1(mat, mode="CHW(HWC)"):
        if isinstance(mat, torch.Tensor):
            mat_new = mat.clone()
        else:
            mat_new = mat.copy()
        assert len(mat_new.shape) == 3, "Input shape should be 3D(CHW or HWC)"
        assert isinstance(mat_new, np.ndarray) or isinstance(
            mat_new, torch.Tensor
        ), "input should be numpy or torch tensor"
        if mode == "CHW(HWC)" or mode == "CHW":
            for i in range(mat_new.shape[0]):
                mat_new[i, :, :] = (mat_new[i, :, :] - mat_new[i, :, :].min()) / (
                    mat_new[i, :, :].max() - mat_new[i, :, :].min()
                )
            F_mat = mat_new
        elif mode == "HWC":
            for i in range(mat_new.shape[2]):
                mat_new[:, :, i] = (mat_new[:, :, i] - mat_new[:, :, i].min()) / (
                    mat_new[:, :, i].max() - mat_new[:, :, i].min()
                )
            F_mat = mat_new
        else:
            raise RuntimeError("Input mode: CHW or HWC")
        return F_mat

    def npy2mat(varname, npyfilepath, matsavepath):
        gg = np.load(npyfilepath)
        io.savemat(matsavepath, {varname: gg})
    def mat2npy(matfilename,varname):
        '''
        this method only works for matlab > v7 file.
        '''
        mat = io.loadmat(matfilename)
        rsp=mat[varname]
        return rsp
    def copy_allfiles(src, dest):
        """
        src:folder path
        dest: folder path
        this will not keep moving the folder to another folder
        this is moving the files in that folder to another folder
        """
        src_files = os.listdir(src)
        for file_name in src_files:
            full_file_name = os.path.join(src, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, dest)

    def mkdir(mainpath, foldername):
        """
        mainpath: path you want to create folders
        foldername: foldername, str, list or tuple
        Return: the path you generate.
        """
        assert isinstance(foldername, (str, tuple, list))
        if isinstance(foldername, str):
            pathname = GF.mkdir0(mainpath, foldername)
        if isinstance(foldername, (list, tuple)):
            pathname = []
            for i in foldername:
                pathname0 = GF.mkdir0(mainpath, i)
                pathname.append(pathname0)
        return pathname

    def mkdir0(mainpath, foldername):
        if mainpath[-1] == "/" or mainpath[-1] == "\\":
            pathname = mainpath + foldername + "/"
            folder = os.path.exists(mainpath + foldername + "/")
            if not folder:
                os.makedirs(f"{mainpath}/{foldername}")
                print("Create folders ing")
                print("done !")
            else:
                print("folder existed")
        else:
            pathname = mainpath + "/" + foldername + "/"
            folder = os.path.exists(mainpath + "/" + foldername + "/")
            if not folder:
                os.makedirs(f"{mainpath}/{foldername}")
                print("Create folders ing")
                print("done !")
            else:
                print("folder already existed")
        return pathname

    def sortTC(vector, sort_mode="Top_down"):
        """
        sort_mode: Top_down/Bottom_up(default:Top_down)

        """
        
        sort_mode=sort_mode.lower()
        if sort_mode not in ("top_down", "bottom_up"):
            raise RuntimeError(
                "sort_mode args incorrect:\nPlease input:\n1.Top_down\n2.Bottom_up"
            )

        if sort_mode == "top_down":
            value = np.sort(vector)[::-1]
            index = np.argsort(vector)[::-1]
        elif sort_mode == "bottom_up":
            value = np.sort(vector)
            index = np.argsort(vector)
        return value, index

    def save_mat_file(filename, var, varname="data"):
        io.savemat(filename + ".mat", {varname: var})

    def tf_to_torch_shape(tf_img):
        assert len(tf_img.shape) == 4, "Shape should be 4 dimensional"
        B, H, W, C = tf_img.shape
        assert (
            C == 1 or C == 3
        ), "Input image shape should be BHWC, Channel judgement is 1 or 3"
        # newi = torch.from_numpy(tf_img).unsqueeze_(0).view(B, C, H, W)
        newi = torch.from_numpy(tf_img).view(B, C, H, W)
        newi = newi.type(torch.float)
        B1, C1, H1, W1 = newi.shape
        assert C1 == 1 or C1 == 3, "Check whether the input shape is BHWC "
        return newi

    def load_data(train_pic_path, train_rsp_path, val_pic_path, val_rsp_path):
        """
        load data from tf format(BHWC)-> norm picture to 1 -> Out is torch format(BCHW)
        """
        train_rsp = np.load(train_rsp_path)
        val_rsp = np.load(val_rsp_path)
        train_pics = np.load(train_pic_path)
        val_pics = np.load(val_pic_path)

        train_pics = GF.norm_to_1(train_pics)
        val_pics = GF.norm_to_1(val_pics)
        train_pics = GF.tf_to_torch_shape(train_pics)
        val_pics = GF.tf_to_torch_shape(val_pics)  # [B,H,W,C] -> [B,C,H,W]
        train_rsp, val_rsp = torch.tensor(train_rsp, dtype=torch.float), torch.tensor(
            val_rsp, dtype=torch.float
        )

        print("val pic shape:", val_pics.shape, "\n train pics shape", train_pics.shape)
        print("val rsp shape", val_rsp.shape, "\n train rsp shape", train_rsp.shape)

        return train_pics, val_pics, train_rsp, val_rsp  # [B/imgs,C,H,W] or [pics,cell]

    def get_all_metrics(pred, real, num_neurons, img_samples_size):
        # pred/real -> (imgs,cells)/(imgs,epoch)
        # assert pred.shape[0] > 500, "First dimension should be images"
        R_square = GF.get_all_R2(pred, real)
        R, VE = GF.get_corr_VE(pred, real, num_neurons, img_samples_size)
        return R_square, R, VE

    def get_corr_VE(pred, real, num_neurons, img_samples_size):
        # I: pred/real -> (imgs,cells), O: (cells,)
        assert (
            len(pred.shape) == 2 and len(real.shape) == 2
        ), "Input shape: (imgs, cells)"
        assert isinstance(pred, np.ndarray) and isinstance(
            real, np.ndarray
        ), "Data input type should be numpy array"
        R = np.zeros(num_neurons)
        VE = np.zeros(num_neurons)
        for neuron in range(num_neurons):
            val_pred = pred[:, neuron]
            val_real = real[:, neuron]
            u2 = np.zeros((2, img_samples_size))
            u2[0, :] = np.reshape(val_pred, (img_samples_size))
            u2[1, :] = np.reshape(val_real, (img_samples_size))
            c2 = np.corrcoef(u2)
            R[neuron] = c2[0, 1]
            VE0 = 1 - (np.var(val_pred - val_real) / np.var(val_real))
            VE[neuron] = VE0
        return R, VE

    def adjust_R_square(pred, real, sample_size=None, label_size=None):
        if isinstance(pred, np.ndarray) and isinstance(real, np.ndarray):
            # pred,real -> numpy, shape:(xx,)
            RSS = np.sum((real - pred) ** 2)
            TSS = np.sum((real - real.mean()) ** 2)
        if isinstance(pred, torch.Tensor) and isinstance(real, torch.Tensor):
            # pred,real -> numpy, shape:(xx,)
            RSS = torch.sum((real - pred) ** 2)
            TSS = torch.sum((real - real.mean()) ** 2)

        R_square = 1 - RSS / TSS

        if not label_size == None:
            n = sample_size
            p = label_size
            R_square_adjust = 1 - ((1 - R_square) * (n - 1)) / (n - p - 1)
        else:
            R_square_adjust = "None"
        return R_square

    def get_all_R2(pred, real):
        # pred or real, (imgs,cells)
        assert (
            len(pred.shape) == 2 and len(real.shape) == 2
        ), "Input shape: (imgs, cells)"
        assert isinstance(pred, np.ndarray) and isinstance(
            real, np.ndarray
        ), "Data input type should be numpy array"
        R2 = []
        for i in range(pred.shape[1]):
            pred1 = pred[:, i]
            real1 = real[:, i]
            # sample_size=pred1.shape[0]
            R = GF.adjust_R_square(pred1, real1)
            R2.append(R)
        return np.stack(R2)

    def different_loss(lossname, real, predict, val_or_train=None):
        if lossname in ("mix_corr_MAE", "mix_corr_MSE"):
            assert val_or_train != None
        if lossname == "mix_corr_MAE":
            criterion = torch.nn.L1Loss(reduction="none")
            if val_or_train == "train":
                loss = (
                    -pearsonr(predict, real, batch_first=True)
                    + 0.1 * torch.mean(torch.abs(real))
                    + 0.1 * torch.mean(criterion(predict, real))
                )
            if val_or_train == "val":
                loss = (
                    torch.mean(-pearsonr(predict, real, batch_first=False))
                    + 0.1 * torch.mean(torch.abs(real))
                    + 0.1 * torch.mean(criterion(predict, real))
                )
        elif lossname == "mix_corr_MSE":
            criterion = torch.nn.MSELoss(reduction="mean")
            if val_or_train == "train":
                loss = (
                    -pearsonr(predict, real, batch_first=True)
                    + 0.1 * torch.mean(torch.abs(real))
                    + 0.1 * torch.mean(criterion(predict, real))
                )
            if val_or_train == "val":
                loss = (
                    torch.mean(-pearsonr(predict, real, batch_first=False))
                    + 0.1 * torch.mean(torch.abs(real))
                    + 0.1 * torch.mean(criterion(predict, real))
                )
        elif lossname == "MAE":
            criterion = torch.nn.L1Loss(reduction="mean")
            loss = criterion(predict, real)
        elif lossname == "MSE":
            # criterion = torch.nn.MSELoss(reduction="sum")  # O have reduction "mean"
            criterion = torch.nn.MSELoss(reduction="mean")
            loss = criterion(predict, real)

        elif lossname == "RMSLE":
            criterion = RMSLELoss()
            loss = criterion(predict, real)
        elif lossname == "MSEN":
            print("_---------", real.shape)
            loss = torch.mean(
                torch.square(torch.relu(torch.abs(predict - real) - 0.1)), axis=-1
            )
            # 保证输入real和predict是（1，xxxx）
        return loss

    def torch_train(
        *,
        net,
        lossname,
        train_loader,
        mainsavepath,
        modelsavepath,
        val_loader,
        device,
        batch_size,
        num_epoch,
        lr,
        Param_dict=None,
        lr_min=0,
        optim="sgd",
        weight_decay=0,
        init=True,
        scheduler_type=None,  # "ReduceLROnPlateau",
        saveModelMode="eveEpoch",  # eveEpoch/lastEpoch
    ):
        """
        this is the train function for pytorch of our project.
        net should be many subnet, individual cnn (or scnn) for different cells.
        """
        t.tic()
        if Param_dict != None:
            assert isinstance(Param_dict, dict)
            valsamplesize = Param_dict["valsamplesize"]
            trainsamplesize = Param_dict["trainsamplesize"]
            num_neurons = Param_dict["num_neurons"]
        else:
            raise RuntimeError("Input Param_dict")
        test_loss_path = f"{mainsavepath}loss_test.npy"
        train_loss_path = f"{mainsavepath}loss_train.npy"

        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.xavier_uniform_(m.weight)

        if init:
            net.apply(init_weights)
        print("training on:", device)
        net.to(device)
        optim = optim.lower()
        if optim == "sgd":
            optimizers = [
                torch.optim.SGD(
                    (param for param in sub_net.parameters() if param.requires_grad),
                    lr=lr,
                    weight_decay=weight_decay,
                    # betas=(0.9, 0.999),
                    # eps=1e-08,
                )
                for sub_net in net
            ]
        elif optim == "adam":
            optimizers = [
                torch.optim.Adam(
                    (param for param in sub_net.parameters() if param.requires_grad),
                    lr=lr,
                    weight_decay=weight_decay,
                )
                for sub_net in net
            ]
        elif optim == "adamW":
            optimizers = [
                torch.optim.AdamW(
                    (param for param in sub_net.parameters() if param.requires_grad),
                    lr=lr,
                    weight_decay=weight_decay,
                )
                for sub_net in net
            ]
        # elif optim == "ranger":
        #     optimizers = [
        #         Ranger(
        #             (param for param in sub_net.parameters() if param.requires_grad),
        #             lr=lr,
        #             weight_decay=weight_decay,
        #         )
        #         for sub_net in net
        #     ]
        if scheduler_type == "Cosine":
            schedulers = [
                CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=lr_min)
                for optimizer in optimizers
            ]
        if scheduler_type == "ReduceLROnPlateau":
            schedulers = [
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=0.1,
                    patience=10,
                    verbose=False,
                    threshold=0.0001,
                    threshold_mode="rel",
                    cooldown=0,
                    min_lr=0,
                    eps=1e-08,
                )
                for optimizer in optimizers
            ]
        if scheduler_type == "CosineAnnealingWarmRestarts":
            schedulers = [
                CosineAnnealingWarmRestarts(
                    optimizer, T_0=5, T_mult=1, eta_min=0, last_epoch=-1, verbose=False
                )
            ]
        test_all_loss = []
        train_all_loss = []
        all_corr = []
        all_VE = []
        all_R2 = []
        for epoch in range(num_epoch):
            if saveModelMode == "eveEpoch":
                ep_path_every_EP = f"{modelsavepath}_ep{epoch+1}"
            else:
                ep_path_every_EP = f"{modelsavepath}_ep{num_epoch}"
            if os.path.exists(ep_path_every_EP):
                saveorload = "load"
                print("Model exists, loading....")
                net.load_state_dict(torch.load(ep_path_every_EP))
                pass
            else:
                train_avg_loss0 = 0.0
                print(f"Training, epoch{epoch+1}")
                train_avg_loss_allcell = np.zeros(
                    num_neurons,
                )
                for subnet in net:
                    subnet.train()
                for i, (subnet, optimizer) in enumerate(zip(net, optimizers)):
                    for batch_num, (x, y) in enumerate(
                        train_loader
                    ):  # NOTE, what order?
                        x, y = x.to(device), y.to(device)
                        predict_train = subnet(x)
                        predict_train = torch.reshape(
                            predict_train, (predict_train.shape[0],)
                        )
                        print("predict_train shape:", predict_train.shape)
                        real_train = y[:, i]
                        loss = GF.different_loss(
                            lossname=lossname,
                            real=real_train,
                            predict=predict_train,
                            val_or_train="train",
                        )

                        loss = loss.mean()
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        train_avg_loss0 += loss.item()
                        print("loss", loss)
                    train_avg_loss1 = train_avg_loss0 / np.floor(
                        trainsamplesize / batch_size
                    )
                    print("train_avg_loss1", train_avg_loss1)
                    train_avg_loss_allcell[i] = train_avg_loss1
                    # scheduler.step(loss)  # NOTE
                print("train_avg_loss_allcell shape:", train_avg_loss_allcell)
                train_avg_loss = np.mean(train_avg_loss_allcell)
                train_all_loss.append(train_avg_loss)
                torch.save(net.state_dict(), ep_path_every_EP)
                np.save(train_loss_path, np.stack(train_all_loss))

            # Validate
            print("------validation ON---------")
            with torch.no_grad():
                net.eval()
                test_loss = 0
                pred_val = []
                real_val = []
                for batch_num_val, (x_val, y_val) in enumerate(val_loader):
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    output_val = torch.stack([subnet(x_val) for subnet in net])
                    # torch.Size([num_neurons, batchsize, 1])
                    output_val = torch.reshape(
                        output_val, (output_val.shape[0], output_val.shape[1])
                    )
                    output_val = torch.transpose(output_val, 0, 1)
                    print("output val shape:", output_val.shape)
                    """
                    output_val -> shape(batch,neurons)
                    y_val -> shape(batch,neurons)
                    """
                    loss = GF.different_loss(
                        lossname=lossname,
                        real=y_val,
                        predict=output_val,
                        val_or_train="val",
                    )  # pytorch mse criterion, will find average loss of all neurons,and all samples in 1 batch.
                    loss = loss.mean()
                    print(loss, "******************")
                    test_loss += loss.item()
                    pred_val.extend(output_val.cpu().numpy())
                    print("pred_val:", len(pred_val))
                    real_val.extend(y_val.cpu().numpy())
                test_loss /= np.floor(valsamplesize / batch_size)
                test_all_loss.append(test_loss)
                pred_val = np.stack(pred_val)
                real_val = np.stack(real_val)
                R_square, R, VE = GF.get_all_metrics(
                    pred=pred_val,
                    real=real_val,
                    num_neurons=num_neurons,
                    img_samples_size=valsamplesize,
                )
                all_corr.append(R)
                all_VE.append(VE)
                all_R2.append(R_square)
                print("-----------------------------------------------")
                print("Number of epoch:", num_epoch)
                print("Number of neurons:", num_neurons)
                print("Number of train samples:", trainsamplesize)
                print("Number of validation samples:", valsamplesize)
                print("Response shape:", pred_val.shape, ",", real_val.shape)
                print(
                    "Corr shape:",
                    np.stack(all_corr).shape,
                    ", loss shape:",
                    np.stack(test_all_loss).shape,
                )
        np.save(test_loss_path, np.stack(test_all_loss))
        t.toc()
        # return: pred_val,real_val,all_corr,all_VE,all_R2
        return (
            pred_val,
            real_val,
            np.stack(all_corr),
            np.stack(all_VE),
            np.stack(all_R2),
        )

class ImageDataset(Dataset):
    def __init__(self, data, labels, cell_start=None, cell_end=None, num_neurons=None):
        """
        cell_start: start from 1
        mode=num_neurons/startend
        """
        self.data = data
        self.labels = labels
        self.cell_start = cell_start
        self.cell_end = cell_end
        self.num_neurons = num_neurons

    def __len__(self):
        return self.data.shape[0]  # number of images
    def __getitem__(self, index):
        cell_start = self.cell_start
        cell_end = self.cell_end
        num_neurons = self.num_neurons
        assert ((cell_start == None and cell_end == None) and num_neurons != None) or (
            (cell_start != None and cell_end != None) and num_neurons == None
        )

        img = self.data[index]

        if num_neurons != None:
            label = self.labels[index, 0 : self.num_neurons]
        elif cell_start != None:
            label = self.labels[index, self.cell_start - 1 : self.cell_end]
        # print("img shape", img.shape, "rsp shape", label.shape)
        return img, label
class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))
