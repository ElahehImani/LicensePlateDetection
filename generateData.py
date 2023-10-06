import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import listdir,mkdir
from os.path import isfile, join, exists
import math

## loading images ------------------------------
rootpath='dataset/'
background_path=rootpath+'backgrounds/'
plate_path=rootpath+'plates/'
outpath=rootpath+'syntheticData/'
if(not exists(outpath)):
    mkdir(outpath)

background_imgs=[f for f in listdir(background_path) if isfile(join(background_path,f))]
plate_imgs=[f for f in listdir(plate_path) if isfile(join(plate_path,f))]

## transformation settings ---------------------
background_shape=(500,500) #(width,height)
plate_shape=(180,45) #(width,height)
rotation_bnd=np.arange(-40,40,30)
shear_bnd=np.arange(-0.2,0.3,0.3)
zero_img_size=(500,500,3)
smpl_per_image=1 #number of synthetic images based on paird background & plate images
generateData=True
plate_corner=np.array([[0,0],[0,plate_shape[0]],[plate_shape[1],0],[plate_shape[1],plate_shape[0]]])

## get affine transform for specific setting ------
def get_affineT(type, param):
    affine=[]
    if(type=='rotation'):
        affine=np.zeros((2,3))
        affine = cv2.getRotationMatrix2D(center=(zero_img_size[1]/2,zero_img_size[0]/2), angle=param, scale=1)

    elif(type=='shear_h'):
        affine=np.zeros((2,3))
        affine[0,0]=1
        affine[1,1]=1
        affine[0,1]=param
        affine[0,2] = -affine[0,1] * zero_img_size[0]/2
        affine[1,2] = -affine[1,0] * zero_img_size[1]/2
        
    elif(type=='shear_v'):
        affine=np.zeros((2,3))
        affine[0,0]=1
        affine[1,1]=1
        affine[1,0]=param
        affine[0,2] = -affine[0,1] * (zero_img_size[0])/2
        affine[1,2] = -affine[1,0] * (zero_img_size[1])/2

    elif(type=='scale'):
        affine=np.zeros((2,3))
        affine[0,0]=param
        affine[1,1]=param

    return affine

## apply affine transformation to the plate image -----------
def applyTransform(plate,affine,showImg):
    zero_img=np.zeros(zero_img_size,dtype=np.uint8)
    start_row=int(np.round(zero_img.shape[0]/2)-np.round(plate.shape[0]/2))
    start_col=int(np.round(zero_img.shape[1]/2)-np.round(plate.shape[1]/2))
    end_row=start_row+plate.shape[0]
    end_col=start_col+plate.shape[1]
    zero_img[start_row:end_row,start_col:end_col,:]=plate

    transformed_plate=cv2.warpAffine(zero_img,affine,(zero_img.shape[1],zero_img.shape[0]))
    points=np.zeros((4,2))
    points[0,0]=start_row
    points[0,1]=start_col

    points[1,0]=start_row
    points[1,1]=end_col

    points[2,0]=end_row
    points[2,1]=start_col

    points[3,0]=end_row
    points[3,1]=end_col


    corners=np.round(np.matmul(points,affine))
    corners=corners[:,0:2]

    new_corner=np.zeros((4,2))
    new_corner[0,:]=corners[0,:]-corners[0,:]
    new_corner[1,:]=corners[1,:]-corners[0,:]
    new_corner[2,:]=corners[2,:]-corners[0,:]
    new_corner[3,:]=corners[3,:]-corners[0,:]

    new_corner=new_corner.astype(np.int16)
    
    # print(new_corner)
    if(showImg):
        plt.figure()
        plt.subplot(2,1,1)
        plt.imshow(zero_img)
        plt.subplot(2,1,2)
        plt.imshow(transformed_plate)
        plt.show()

    return (transformed_plate,new_corner)

## overlaying transformed plate image on the background ----------
def augmentImage(background,plate,corners):
    gray=cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
    ret,binary=cv2.threshold(gray,10,255,cv2.THRESH_BINARY)
    output=cv2.connectedComponentsWithStats(binary,4,cv2.CV_32S)
    stat=output[2]
    boundingBox=binary[stat[1,1]:stat[1,1]+stat[1,3],stat[1,0]:stat[1,0]+stat[1,2]]
    croped_plate=plate[stat[1,1]:stat[1,1]+stat[1,3],stat[1,0]:stat[1,0]+stat[1,2],:]

    row_boundary=(1,background.shape[0]-boundingBox.shape[0])
    col_boundary=(1,background.shape[1]-boundingBox.shape[1])
    start_row=np.random.randint(row_boundary[0],row_boundary[1])
    start_col=np.random.randint(col_boundary[0],col_boundary[1])
    end_row=start_row+boundingBox.shape[0]-1
    end_col=start_col+boundingBox.shape[1]-1
    row=-1
    top_left=[]
    sw=0
    for i in range(start_row,end_row):
        row+=1
        col=-1
        for j in range(start_col,end_col):
            col+=1
            if(boundingBox[row,col]==0):                
                continue

            if(sw==0):
                top_left=[i,j]
                sw=1

            background[i,j,:]=croped_plate[row,col,:]

    if(corners[1,0]<corners[0,0]):
        top_left[0]=top_left[0]-corners[1,0]
        top_left[1]=top_left[1]-corners[1,1]

    new_corner=np.zeros((4,2))
    new_corner[0,0]=top_left[0]
    new_corner[0,1]=top_left[1]

    new_corner[1,0]=top_left[0]+corners[1,0]
    new_corner[1,1]=top_left[1]+corners[1,1]

    new_corner[2,0]=top_left[0]+corners[2,0]
    new_corner[2,1]=top_left[1]+corners[2,1]

    new_corner[3,0]=top_left[0]+corners[3,0]
    new_corner[3,1]=top_left[1]+corners[3,1]
    
    return (new_corner,background)

def save_info(imagePath,infoPath,image,corner,filename):
    cv2.imwrite(imagePath+filename+".jpg",image)
    np.savetxt(infoPath+filename+".txt", corner)
    

# apply inverse transformation to the background image containing plate
def inverse_transformation(image,corner):
    corner=corner.astype(int)
    min_row=np.min(corner[:,0])
    min_col=np.min(corner[:,1])
    max_row=np.max(corner[:,0])
    max_col=np.max(corner[:,1])

    croped_img=image[min_row:max_row,min_col:max_col,:]
    input_pts = np.float32([[corner[0,0]-min_row,corner[0,1]-min_col], [corner[1,0]-min_row,corner[1,1]-min_col],
                             [corner[2,0]-min_row,corner[2,1]-min_col]])
    output_pts = np.float32([[plate_corner[0,0],plate_corner[0,1]], 
                             [plate_corner[1,0],plate_corner[1,1]],
                             [plate_corner[2,0],plate_corner[2,1]]])
    
    inv_affine=cv2.getAffineTransform(output_pts,input_pts)
    transformed_image=cv2.warpAffine(croped_img,inv_affine,(2*croped_img.shape[1],2*croped_img.shape[0]))

    plt.figure()
    plt.subplot(2,1,1)
    plt.imshow(transformed_image)
    plt.subplot(2,1,2)
    plt.imshow(croped_img)
    plt.show()
    
    return transformed_image

def extractNum(plate,filename):
    x_offset_l=int(round(plate.shape[1]/12))
    x_offset_r=5
    y_offset=5
    plate=plate[y_offset:plate.shape[0]-y_offset,x_offset_l:plate.shape[1]-x_offset_r,:]
    gray_plate=cv2.cvtColor(plate,cv2.COLOR_RGB2GRAY)
    med=np.median(gray_plate,axis=[0,1])
    ret,binary=cv2.threshold(gray_plate,med-10,255,cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    erode_binary=cv2.erode(binary,kernel,iterations=1)
    close_binary=cv2.dilate(erode_binary,kernel,iterations=1)
    output=cv2.connectedComponentsWithStats(close_binary)
    (totalLabels, label_ids, stat, centroid) = output
    indices = np.argsort(centroid[:,0],axis=0)

    center_bnd=[plate.shape[0]/2-7,plate.shape[0]/2+7]
    num=1
    min_pixel=0.02*(plate.shape[0]*plate.shape[1])
    number_boundingbox=[]
    plt.figure()
    for i in range(stat.shape[0]):
        idx=indices[i]
        if(idx==0):
            continue
        if(centroid[idx,1]>center_bnd[0] and centroid[idx,1]<center_bnd[1] and stat[idx,4]>min_pixel):
            number_boundingbox.append(stat[idx])
            number=plate[stat[idx,1]:stat[idx,1]+stat[idx,3],stat[idx,0]:stat[idx,0]+stat[idx,2],:]
            plt.subplot(2,4,num)
            plt.imshow(number)
            num+=1

    plt.savefig(filename)

## generate synthetic dataset ------------------
showImg=False
counter=1

if(generateData):
    for i in range(len(background_imgs)):
        background_img=cv2.imread(background_path+background_imgs[i])
        background_img=cv2.resize(background_img,background_shape)
        
        ## create subfolders due to the google colab problem in reading huge files
        subfolder=outpath+str((i+1))+'/'
        if(exists(subfolder)==0):
            mkdir(subfolder)

        outpath_img=subfolder+'image/'
        outpath_info=subfolder+'info/'
        outpath_number=subfolder+'number/'

        if(exists(outpath_img)==0):
            mkdir(outpath_img)

        if(exists(outpath_info)==0):
            mkdir(outpath_info)

        if(exists(outpath_number)==0):
            mkdir(outpath_number)

        for j in range(len(plate_imgs)):
            plate=cv2.imread(plate_path+plate_imgs[i])
            plate=cv2.resize(plate,plate_shape)
            extractNum(plate,'test.jpg')
            for r in range(len(rotation_bnd)):
                affine=get_affineT('rotation',rotation_bnd[r])
                (img,corners)=applyTransform(plate,affine,showImg)
                for k in range(smpl_per_image):
                    background_img_cp=np.copy(background_img)
                    (corner,generatedImage)=augmentImage(background_img_cp,img,corners)
                    save_info(outpath_img,outpath_info,generatedImage,corner,str(counter))
                    counter+=1

            for r in range(len(shear_bnd)):
                affine=get_affineT('shear_h',shear_bnd[r])
                (img,corners)=applyTransform(plate,affine,showImg)
                for k in range(smpl_per_image):
                    background_img_cp=np.copy(background_img)
                    (corner,generatedImage)=augmentImage(background_img_cp,img,corners)
                    save_info(outpath_img,outpath_info,generatedImage,corner,str(counter))
                    counter+=1

            for r in range(len(shear_bnd)):
                affine=get_affineT('shear_v',shear_bnd[r])
                (img,corners)=applyTransform(plate,affine,showImg)
                for k in range(smpl_per_image):
                    background_img_cp=np.copy(background_img)
                    (corner,generatedImage)=augmentImage(background_img_cp,img,corners)
                    save_info(outpath_img,outpath_info,generatedImage,corner,str(counter))
                    counter+=1