import numpy as np
import scipy
import scipy.ndimage
import random
import nrrd
from glob import glob
from data_loader import *

data_dir = r"D:\autoimplant-master\training_set\defective_skull"
label_dir = r"D:\autoimplant-master\training_set\implant"
bbox_dir = r"D:\autoimplant-master\predictions_n1"
data_save_dir = r"D:\autoimplant-master\training_set\defective_skull_p2"
label_save_dir = r"D:\autoimplant-master\training_set\implant_p2"

data_list =glob('{}/*.nrrd'.format(data_dir))
label_list=glob('{}/*.nrrd'.format(label_dir))
bbox_list=glob('{}/*.nrrd'.format(bbox_dir))
################################################ n1
# for i in range(len(data_list)):
#     data,hd=nrrd.read(data_list[i])
#     label,hl=nrrd.read(label_list[i])
#     print("data:",data_list[i])
#     print("label:",label_list[i])
#     data_=resizing(data)
#     label_=resizing(label)
#     if i < 10:
#         data1=data_save_dir+r"\00%d.nrrd"%i
#         label1=label_save_dir+r"\00%d.nrrd"%i
#     elif i < 100:
#         data1=data_save_dir+r"\0%d.nrrd"%i
#         label1=label_save_dir+r"\0%d.nrrd"%i
#     else:
#         data1=data_save_dir+r"\%d.nrrd"%i
#         label1=label_save_dir+r"\%d.nrrd"%i

#     nrrd.write(data1, data_, hd)
#     nrrd.write(label1, label_, hl)

############################################## n2
for i in range(len(data_list)):
    data,hd=nrrd.read(data_list[i])
    label,hl=nrrd.read(label_list[i])
    bbox,hb=nrrd.read(bbox_list[i])
    print("data:",data_list[i])
    print("label:",label_list[i])
    print("bbox:",bbox_list[i])

    resx,resxx,resy,resyy,resz,reszz=bbox_cal(bbox,data.shape[2])
    data_inp=data[resx-margin:512-resxx+margin,resy-margin:521-resyy+margin,data.shape[2]-128:data.shape[2]]
    data_lb=label[resx-margin:512-resxx+margin,resy-margin:512-resyy+margin,data.shape[2]-128:data.shape[2]]

    data_inp=padding(data_inp)
    data_lb=padding(data_lb)
    if i < 10:
        data1=data_save_dir+r"\00%d.nrrd"%i
        label1=label_save_dir+r"\00%d.nrrd"%i
    elif i < 100:
        data1=data_save_dir+r"\0%d.nrrd"%i
        label1=label_save_dir+r"\0%d.nrrd"%i
    else:
        data1=data_save_dir+r"\%d.nrrd"%i
        label1=label_save_dir+r"\%d.nrrd"%i

    nrrd.write(data1, data_inp, hd)
    nrrd.write(label1, data_lb, hl)