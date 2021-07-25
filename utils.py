import numpy as np
import math
import cv2
import os
def convert_xy(axis_x,axis_y):
    min_x=np.min(axis_x)
    min_y=np.min(axis_y)
    axis_x-=min_x
    axis_y-=min_y
    max_x = np.max(axis_x)
    max_y = np.max(axis_y)
    axis_x*=255/max_x
    axis_y*=255/max_y
    return axis_x,axis_y
def relative_pos(x0,y0,x,y):
    distance=np.sqrt((x0-x)**2+(y0-y)**2)
    angle=math.atan2(y-y0,x-x0)
    return distance,angle

def copy_data(cir_coord):
    t=cir_coord.shape[2]
    batch=cir_coord.shape[0]
    true_coord=np.zeros((cir_coord.shape[0],cir_coord.shape[1],224,224,cir_coord.shape[4]))
    for b in range(batch):
        for i in range(cir_coord.shape[1]):
            for j in range(cir_coord.shape[4]):
                #map=np.tile(cir_coord[b,i,:,:,j],14)
                map1=np.zeros((224,224))
                if t>224:
                    map1=np.resize(cir_coord[b,i,:,:,j],(224,224))
                else:
                    p = np.zeros((224,16))
                    mod=224%t
                    div=224//t
                    for q in range(div):
                        p[q*t:(q+1)*t,:]=cir_coord[b,i,:,:,j]
                    p[div*t:div*t+mod,:]=cir_coord[b,i,:,:,j][:mod,:]
                    map1=np.resize(p,(224,224))
                true_coord[b,i,:,:,j]=map1
    return true_coord


def pre_treat_train(path):
    data=np.load(path)
    t = data.shape[2]
    batch=data.shape[0]
    cir_coord = np.zeros((batch,4, t, 16, 2))#4为四个关键点，t为帧数，16为剩下的关节，2为两个圆坐标
    for b in range(batch):

        imcoords=[5,6,11,12]#四个关键点5,6,11,12
        for n,imcoord in enumerate(imcoords):

            for i in range(t):
                data[b, 0, i, :, :][imcoord][0] += np.random.randint(-3, 3)
                data[b, 2, i, :, :][imcoord][0] += np.random.randint(-3, 3)
                for coord_num in range(17):
                    if coord_num<imcoord:
                        data[b, 0, i, :, :][coord_num][0] += np.random.randint(-3, 3)
                        data[b, 2, i, :, :][coord_num][0] += np.random.randint(-3, 3)
                        a=data[b,2,i,:,0]
                        distance,angle=relative_pos(data[b,0,i,:,:][imcoord][0],data[b,2,i,:,:][imcoord][0]+np.random.randint(-3, 3),data[b,0,i,:,:][coord_num][0],data[b,2,i,:,:][coord_num][0])
                        cir_coord[b,n,i,coord_num,0],cir_coord[b,n,i,coord_num,1]=distance,angle
                    elif coord_num>imcoord:
                        data[b, 0, i, :, :][coord_num][0] += np.random.randint(-3, 3)
                        data[b, 2, i, :, :][coord_num][0] += np.random.randint(-3, 3)
                        distance, angle = relative_pos(data[b, 0, i, :, :][imcoord][0], data[b, 2, i, :, :][imcoord][0],
                                                       data[b, 0, i, :, :][coord_num][0],
                                                       data[b, 2, i, :, :][coord_num][0])
                        cir_coord[b,n, i,coord_num-1,0],cir_coord[b,n, i,coord_num-1,1]=distance,angle
        cir_coord[b,n,:,:,0],cir_coord[b,n,:,:,1]=convert_xy(cir_coord[b,n,:,:,0],cir_coord[b,n,:,:,1])
    # 显示灰度图
    for b in range(batch):
        for n in range(4):
            cir_coord[b, n, :, :, 0] /= 255
            cir_coord[b, n, :, :, 1] /= 255
    cir_coord = copy_data(cir_coord)
    return cir_coord

def pre_treat_test(path):
    data = np.load(path)
    t = data.shape[2]
    batch = data.shape[0]
    cir_coord = np.zeros((batch, 4, t, 16, 2))  # 4为四个关键点，t为帧数，16为剩下的关节，2为两个圆坐标
    for b in range(batch):

        imcoords = [5, 6, 11, 12]  # 四个关键点5,6,11,12
        for n, imcoord in enumerate(imcoords):

            for i in range(t):
                for coord_num in range(17):
                    if coord_num < imcoord:
                        a = data[b, 2, i, :, 0]
                        distance, angle = relative_pos(data[b, 0, i, :, :][imcoord][0],
                                                       data[b, 2, i, :, :][imcoord][0] + np.random.randint(-3, 3),
                                                       data[b, 0, i, :, :][coord_num][0],
                                                       data[b, 2, i, :, :][coord_num][0])
                        cir_coord[b, n, i, coord_num, 0], cir_coord[b, n, i, coord_num, 1] = distance, angle
                    elif coord_num > imcoord:
                        distance, angle = relative_pos(data[b, 0, i, :, :][imcoord][0],
                                                       data[b, 2, i, :, :][imcoord][0],
                                                       data[b, 0, i, :, :][coord_num][0],
                                                       data[b, 2, i, :, :][coord_num][0])
                        cir_coord[b, n, i, coord_num - 1, 0], cir_coord[b, n, i, coord_num - 1, 1] = distance, angle
        cir_coord[b, n, :, :, 0], cir_coord[b, n, :, :, 1] = convert_xy(cir_coord[b, n, :, :, 0],
                                                                        cir_coord[b, n, :, :, 1])

    #显示灰度图
    for b in range(batch):
        for n in range(4):
            cir_coord[b, n, :, :, 0]/=255
            cir_coord[b, n, :, :, 1]/=255

    # cv2.imshow('1',cir_coord[0,3,:,:,0])
    # cv2.imshow('2', cir_coord[0,3,:,:,1])
    # cv2.waitKey(0)
    cir_coord=copy_data(cir_coord)
    return cir_coord

if __name__ == '__main__':
    sample_path = './data/train/000/P000S00G10B10H50UC022000LC021000A000R0_08251609.npy'
    sample = pre_treat(sample_path)