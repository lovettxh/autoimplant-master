import numpy as np
import scipy
import scipy.ndimage
import random
import nrrd
from glob import glob

data_dir = r"D:\autoimplant-master\training_set\defective_skull"
label_dir = r"D:\autoimplant-master\training_set\implant"

data_list =glob('{}/*.nrrd'.format(data_dir))
label_list=glob('{}/*.nrrd'.format(label_dir))

for i in range(len(data_list)):
    data,hd=nrrd.read(data_list[i])
    label,hl=nrrd.read(label_list[i])

    if i < 10:
        data1=data_dir+r"\10%d.nrrd"%i
        data2=data_dir+r"\20%d.nrrd"%i
        data3=data_dir+r"\30%d.nrrd"%i
        data4=data_dir+r"\40%d.nrrd"%i
        data5=data_dir+r"\50%d.nrrd"%i
        label1=label_dir+r"\10%d.nrrd"%i
        label2=label_dir+r"\20%d.nrrd"%i
        label3=label_dir+r"\30%d.nrrd"%i
        label4=label_dir+r"\40%d.nrrd"%i
        label5=label_dir+r"\50%d.nrrd"%i
    elif i < 100:
        data1=data_dir+r"\1%d.nrrd"%i
        data2=data_dir+r"\2%d.nrrd"%i
        data3=data_dir+r"\3%d.nrrd"%i
        data4=data_dir+r"\4%d.nrrd"%i
        data5=data_dir+r"\5%d.nrrd"%i
        label1=label_dir+r"\1%d.nrrd"%i
        label2=label_dir+r"\2%d.nrrd"%i
        label3=label_dir+r"\3%d.nrrd"%i
        label4=label_dir+r"\4%d.nrrd"%i
        label5=label_dir+r"\5%d.nrrd"%i
    print("data shape:",data.shape)
    print("label shape:",label.shape)
    nrrd.write(data1, np.rot90(data,1,(0,1)).astype('int32'),hd)
    nrrd.write(data2, np.rot90(data,2,(0,1)).astype('int32'),hd)
    nrrd.write(data3, np.rot90(data,3,(0,1)).astype('int32'),hd)
    nrrd.write(data4, np.rot90(data,2,(1,2)).astype('int32'),hd)
    nrrd.write(data5, np.rot90(data,2,(2,0)).astype('int32'),hd)
    nrrd.write(label1, np.rot90(label,1,(0,1)).astype('int32'),hl)
    nrrd.write(label2, np.rot90(label,2,(0,1)).astype('int32'),hl)
    nrrd.write(label3, np.rot90(label,3,(0,1)).astype('int32'),hl)
    nrrd.write(label4, np.rot90(label,2,(1,2)).astype('int32'),hl)
    nrrd.write(label5, np.rot90(label,2,(2,0)).astype('int32'),hl)