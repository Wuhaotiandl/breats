from shelper import *
from helper import load_data_classify
import os
import pydicom as dicom
import numpy as np
from shutil import copyfile

def test(data_path):
    """
        从下到上排序的 CT_01 对应最下面这一张
    """
    slices = []
    paths = os.listdir(data_path)
    for path in paths:
        img_path = os.path.join(data_path, path)
        slice = dicom.read_file(img_path)
        slices.append(slice)
    slices = [s for s in slices if s.Modality == 'CT']
    images = np.stack([s.pixel_array for s in slices])
    for image in images:
        ShowImage(1, image)

def extract_one(data_path, json_store, model_path, save_path):
    imgs_original, SOPInstanceUIDs, ImagePositionPatients, spacing, Space, classess_value, load_time, classify_time, pname = load_data_classify(
        data_path, json_store, model_path)
    # 删除json文件
    os.remove(json_store + os.sep + 'series_classess.json')
    indexSun = []
    # 0对应CT_001.dcm, 1对应CT_002.dcm
    for i, value in enumerate(classess_value):
        if value in [7]:
            indexSun.append(i)

    paths = make_path(indexSun)
    s_path = os.path.join(save_path, pname)
    if not os.path.exists(s_path):
        os.makedirs(s_path)
    for path in paths:
        src_path = os.path.join(data_path, path)
        tar_path = os.path.join(s_path, path)
        copyfile(src_path, tar_path)
        print('finish {}'.format(tar_path))

def make_path(indexSun):
    a = []
    for index in indexSun:
        if index + 1 < 10:
            a.append('CT_00{}.dcm'.format(str(index+1)))
        elif index + 1 < 100:
            a.append('CT_0{}.dcm'.format(str(index+1)))
        else:
            a.append('CT_{}.dcm'.format(str(index + 1)))
    return a


def extract_all_7(data_path, save_path, json_store, model_path):
    all_patients_paths = os.listdir(data_path)
    for p_path in all_patients_paths:
        fin_path = os.path.join(data_path, p_path, 'CT 1')
        extract_one(fin_path, json_store, model_path, save_path)
data_path = r'G:\data\esophagus\GE NING data\CTV'
save_path = r'G:\data\esophagus\GE NING data\classify_7'
json_store = r'G:'
model_path = r'E:\PycharmProjects\SBSS-CNN\new_program\prog\model' + os.sep
extract_all_7(data_path, save_path, json_store, model_path)