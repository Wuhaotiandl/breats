import time
import pydicom as dicom
import os
import numpy as np
from pydicom.filereader import InvalidDicomError
import json
import sys
from scipy import interpolate
from keras import models
from skimage.measure import label as label_function
from skimage.measure import regionprops
import skimage.morphology as sm
import random
import math

def load_data_classify(path_test, path_ofclassess_json, path_of_weights):
    #    slices = [dicom.read_file(path_test + os.sep + s, force = True) for s in os.listdir(path_test)]
    tic = time.time()
    slices = []
    # 读取文件中的某一个dicom文件
    for s in os.listdir(path_test):
        try:
            one_slices = dicom.read_file(path_test + os.sep + s)
        except IOError:
            print('No such file')
            continue
        except InvalidDicomError:
            print('Invalid Dicom file')
            continue
        slices.append(one_slices)

    slices = [s for s in slices if s.Modality == 'CT']
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]), reverse=False)
    SOPInstanceUIDs = [x.SOPInstanceUID for x in slices]
    classify_tic = time.time()
    if os.path.exists(os.path.join(path_ofclassess_json, 'series_classess.json')):
        information_classess = read_classify_json(path_ofclassess_json)
        classess_value = [information_classess[i.SOPInstanceUID] for i in slices]
        print('json file of whole body classification has existed and loaded successfully !')
    else:
        classess_value = main_classify(path_test, path_of_weights)
        rst_json_list = []  # 2019.1.29 lixn修改
        for i in range(len(classess_value)):
            temp_dict = {}
            temp_dict["SOPInstanceUID"] = SOPInstanceUIDs[i]
            temp_dict["classess_value"] = str(classess_value[i])
            rst_json_list.append(temp_dict)
        json_file_name = os.path.join(path_ofclassess_json, 'series_classess.json')
        with open(json_file_name, "w") as file:
            json.dump(rst_json_list, file)
        print('load the network of whole body classification to get the json file !')
    classify_toc = time.time()

    #    print(slices[0].ImagePositionPatient, slices[0].PixelSpacing)

    ImagePositionPatients = np.array([x.ImagePositionPatient for x in slices]).reshape(len(slices), len(
        slices[0].ImagePositionPatient))
    patient_name = slices[0].PatientName.family_name
    Space = slices[0].PixelSpacing

    if len(slices) > 1:
        try:
            slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    elif len(slices) == 1:
        slice_thickness = slices[0].Slice_Thickness
    else:
        sys.exit(1)

    for s in slices:
        s.SliceThickness = slice_thickness

    #############################################################
    # 对于使用dicom不需要这一段,直接使用spacing就可以
    # pydicom需要把spacing放到一个暂存的数组中,因为他的spacing数据类型问题
    #############################################################
    temp_spacing = []
    for i in range(len(Space)):
        temp_spacing.append(float(Space[i]))

    #    spacing = slices[1].PixelSpacing
    spacing = map(float, ([slices[2].SliceThickness] + temp_spacing))
    spacing = np.array(list(spacing))

    #    spacing[0], spacing[2] = spacing[2], spacing[0]

    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    intercept = slices[0].RescaleIntercept
    slope = slices[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)
    image = image.astype(np.float32)
    # 防止有些数据有金属伪影
    image[image < -1000] = -1000
    image[image > 1000] = 1000
    #    print(image.shape)
    #    print(spacing)
    #    print(SOPInstanceUIDs)
    #    print(ImagePositionPatients)

    # # 归一化
    # for i in range(image.shape[0]):
    #     image[i, :, :] = np.array(image[i, :, :], np.float32)
    #     image[i, :, :] = (image[i, :, :] - np.mean(image[i, :, :])) / np.std(image[i, :, :] + 1e-10)

    toc = time.time()

    classify_time = classify_toc - classify_tic
    load_time = toc - tic - classify_time

    return image, SOPInstanceUIDs, ImagePositionPatients, spacing, Space, classess_value, load_time, classify_time, patient_name

def read_classify_json(path_ofclassess_json):
    path_ofclassess_json = os.path.join(path_ofclassess_json, 'series_classess.json')
    all_sopinstanceuid = []
    all_classess = []
    information_classess = {}
    with open(path_ofclassess_json, 'r') as fp:
        classfy_result = json.load(fp)
        for one_classfy_result in classfy_result:
            uid = one_classfy_result['SOPInstanceUID']
            all_sopinstanceuid.append(uid)
            value = one_classfy_result['classess_value']
            all_classess.append(int(value))
            information_classess = dict(zip(all_sopinstanceuid, all_classess))
    return information_classess

def main_classify(path_of_ct, path_of_weights):
    imgs_data, sop_uid = load_imgs_ct(path_of_ct)
    net = load_net(path_of_weights)
    result = predict_main(imgs_data, net)
    return result


# =============================================================================
# 通过CT导入原图,为防止金属伪影,处理方式为:
# image[image<-1500] = -1024
# image[image>2976] = 2976
# 然后进行插值,将平面插值为(1,1)
# =============================================================================
def load_imgs_ct(imgs_path):
    slices = []
    for s in os.listdir(imgs_path):
        try:
            one_slices = dicom.read_file(os.path.join(imgs_path, s))
        except IOError:
            print('No such file')
            continue
        except InvalidDicomError:
            print('Invalid Dicom file')
            continue
        slices.append(one_slices)
    slices = [s for s in slices if s.Modality == 'CT']
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]), reverse=False)
    sop_uid = [i.SOPInstanceUID for i in slices]

    if len(slices) > 1:
        try:
            slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    elif len(slices) == 1:
        slice_thickness = slices[0].SliceThickness
    else:
        sys.exit(1)

    for s in slices:
        s.SliceThickness = slice_thickness

    spacing = slices[1].PixelSpacing
    spacing = np.array(spacing, dtype=np.float32)
    #        spacing顺序是xy
    spacing[0], spacing[1] = spacing[1], spacing[0]

    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    intercept = slices[0].RescaleIntercept
    slope = slices[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
    image += np.int16(intercept)
    image = image.astype(np.float32)
    image = image.swapaxes(0, 2)
    image[image < -1500] = -1024
    image[image > 2976] = 2976
    image = nearest_interpolation(image, spacing, [1, 1])
    return image, sop_uid


def nearest_interpolation(data, spacing, std=(1, 1)):
    print('start interpolation......')
    assert len(spacing) == 2
    pixel = np.array(spacing, dtype=np.float32) / np.array(std, dtype=np.float32)
    new_x = int(data.shape[0] * pixel[0])
    new_y = int(data.shape[1] * pixel[1])
    img_new = np.zeros([new_x, new_y, data.shape[2]], dtype=np.float32)
    for i in range(data.shape[2]):
        img = data[:, :, i]

        linspace_original_x = np.linspace(-1, 1, data.shape[0])
        linspace_original_y = np.linspace(-1, 1, data.shape[1])

        linspace_new_x = np.linspace(-1, 1, new_x)
        linspace_new_y = np.linspace(-1, 1, new_y)

        newfunc = interpolate.interp2d(linspace_original_x, linspace_original_y, img, kind='linear')
        new_one_layer = newfunc(linspace_new_x, linspace_new_y)
        img_new[:, :, i] = new_one_layer
    print('interpolation done......')
    return img_new

# =============================================================================
# 导入网络
# =============================================================================
def load_net(weights_path):
    print ('start load model ......')
    with open(os.path.join(weights_path, 'full_body_classify.json')) as file:
        classify_net = models.model_from_json(file.read())
    classify_net.load_weights(os.path.join(weights_path, 'full_body_classify.h5'))
#    classify_net = load_model(os.path.join(weights_path, 'full_body_classify.h5'))
    print ('load model done ......')
    return classify_net

# =============================================================================
# 预测总入口
# 输入网络
# 可以输入一个完整的一套数据,一可以输入一张图片;
# 如果是一套数据,按照规定好的,shape=(x,y,z),人是像左躺的,输出list,对应每张图片的类别(0~6).
# 如果是一张图片,则输出一个list,只有一个元素(0~6).
# =============================================================================
def predict_main(imgs_data, net):
    if len(imgs_data.shape) == 3:
        result_all = []
        for z in range(imgs_data.shape[2]):
            predicted_result = predict_one_slice(imgs_data[:,:,z], net)
            result_all += predicted_result
        result_all = after_process(result_all)
        result_all = np.array(result_all)+1
        result_all = list(result_all)
        return result_all
    elif len(imgs_data.shape) == 2:
        predicted_result = predict_one_slice(imgs_data, net)
        predicted_result = np.array(predicted_result)+1
        predicted_result = list(predicted_result)
        return predicted_result
    else:
        print('input data is wrong')
#        print ('输入数据错误,必须为3维或者2维')
        raise ValueError

# =============================================================================
# 处理一张图片,包括预处理,将预测的结果返回一个list
# =============================================================================
def predict_one_slice(imgs, net, batch_size=(224, 224)):
    center_coordinates = find_body_center(imgs)
    one_img_to_predict = standerdize(imgs)
    cutted_data_to_predict = cut_data(one_img_to_predict, batch_size, center_coordinates)
    cutted_data_to_predict = cutted_data_to_predict.astype(np.float32)
    cutted_data_to_predict_after_reshape = cutted_data_to_predict.reshape(1, cutted_data_to_predict.shape[0], cutted_data_to_predict.shape[1], 1)
    result_label = predict_data(net, cutted_data_to_predict_after_reshape)
#将概率小于0.5的值0,如果这个样本输出概率都小于0.5,这个样本类别暂时赋值None
    clacess_of_one = []
    result_label[result_label<0.5] = 0
    if result_label.max() == 0:
#        clacess_of_one.append(None)
        clacess_of_one.append(None)
    else:
        clacess_of_one.append(int(np.argmax(result_label, axis=1)))
    return clacess_of_one

# =============================================================================
# 找一张图片中身体的中心
# =============================================================================
def find_body_center(one_slice_img):
    x, y = np.shape(one_slice_img)
    x_center = math.ceil(x/2)
    y_center = math.ceil(y/2)
    threshold = 0.3 * x
    binary = one_slice_img > -400
    image = np.array(binary, dtype = np.uint8)
    image = sm.binary_opening(image, selem=sm.disk(12))
    labeled_image = label_function(image, connectivity = 2)
    marked_image = regionprops(labeled_image)
    center_to_select = []
    for area_property in marked_image:
        if area_property.centroid[0]>threshold and area_property.centroid[0]<x-threshold and area_property.centroid[1]>threshold and area_property.centroid[1]<y-threshold:
            center_to_select.append(area_property.centroid)
    if center_to_select:
        distance_all = []
        for one_coordinate in center_to_select:
            distence = np.square(x_center - one_coordinate[0]) + np.square(y_center - one_coordinate[1])
            distance_all.append(distence)
        i = np.argmin(distance_all)
        center = center_to_select[i]
    else:
        center = None
    return center
# =============================================================================
# 标准化
# =============================================================================
def standerdize(imgs):
#避免除以0,,epsilo和keras的BN层设置相同,为0.001
    epsilo = 0.001
    standerdized_imgs = (imgs - np.mean(imgs))/(np.std(imgs) + epsilo)
    return standerdized_imgs


# =============================================================================
# 将原图剪切为256,为眼睛区域预测生成数据
# =============================================================================
def cut_data(one_slice_img, batch_size, center_coordinates, shake_stride=0):
    # 还要考虑原图没有batch_size大的情况
    input_size = batch_size
    if center_coordinates:
        if input_size[0] < one_slice_img.shape[0]:
            x_center = int(center_coordinates[0])
            random_data_x = random.randint(shake_stride * (-1), shake_stride)
            x_start = x_center - math.ceil(input_size[0] / 2) + random_data_x
            if x_start < 0:
                x_start = 0
            elif x_start + input_size[0] > one_slice_img.shape[0]:
                x_start = one_slice_img.shape[0] - input_size[0]
            resized_data_1 = one_slice_img[x_start:x_start + input_size[0], :]
        else:
            random_data_x = random.randint(shake_stride * (-1), shake_stride)
            x_start = int((input_size[0] - one_slice_img.shape[0]) / 2) + random_data_x
            if x_start < 0:
                x_start = 0
            elif x_start + one_slice_img.shape[0] > input_size[0]:
                x_start = input_size[0] - one_slice_img.shape[0]
            resized_data_1 = np.zeros((input_size[0], one_slice_img.shape[1]))
            resized_data_1[x_start:x_start + one_slice_img.shape[0], :] = one_slice_img[:, :]

        if input_size[1] < one_slice_img.shape[1]:
            y_center = int(center_coordinates[1])
            random_data_y = random.randint(shake_stride * (-1), shake_stride)
            y_start = y_center - math.ceil(input_size[1] / 2) + random_data_y
            if y_start < 0:
                y_start = 0
            elif y_start + input_size[1] > one_slice_img.shape[1]:
                y_start = one_slice_img.shape[1] - input_size[1]
            resized_data_2 = resized_data_1[:, y_start:y_start + input_size[1]]
        else:
            random_data_y = random.randint(shake_stride * (-1), shake_stride)
            y_start = int((input_size[1] - one_slice_img.shape[1]) / 2) + random_data_y
            if y_start < 0:
                y_start = 0
            elif y_start + one_slice_img.shape[1] > input_size[1]:
                y_start = input_size[1] - one_slice_img.shape[1]
            resized_data_2 = np.zeros((input_size[1], input_size[1]))
            resized_data_2[:, y_start:y_start + one_slice_img.shape[1]] = resized_data_1[:, :]
        assert resized_data_2.shape == (input_size[0], input_size[1])
    else:
        if input_size[0] < one_slice_img.shape[0]:
            x_center = math.ceil(one_slice_img.shape[0] / 2)
            random_data_x = random.randint(shake_stride * (-1), shake_stride)
            x_start = x_center - math.ceil(input_size[0] / 2) + random_data_x
            if x_start < 0:
                x_start = 0
            elif x_start + input_size[0] > one_slice_img.shape[0]:
                x_start = one_slice_img.shape[0] - input_size[0]
            resized_data_1 = one_slice_img[x_start:x_start + input_size[0], :]
        else:
            random_data_x = random.randint(shake_stride * (-1), shake_stride)
            x_start = int((input_size[0] - one_slice_img.shape[0]) / 2) + random_data_x
            if x_start < 0:
                x_start = 0
            elif x_start + one_slice_img.shape[0] > input_size[0]:
                x_start = input_size[0] - one_slice_img.shape[0]
            resized_data_1 = np.zeros((input_size[0], one_slice_img.shape[1]))
            resized_data_1[x_start:x_start + one_slice_img.shape[0], :] = one_slice_img[:, :]

        if input_size[1] < one_slice_img.shape[1]:
            y_center = math.ceil(one_slice_img.shape[1] / 2)
            random_data_y = random.randint(shake_stride * (-1), shake_stride)
            y_start = y_center - math.ceil(input_size[1] / 2) + random_data_y
            if y_start < 0:
                y_start = 0
            elif y_start + input_size[1] > one_slice_img.shape[1]:
                y_start = one_slice_img.shape[1] - input_size[1]
            resized_data_2 = resized_data_1[:, y_start:y_start + input_size[1]]
        else:
            random_data_y = random.randint(shake_stride * (-1), shake_stride)
            y_start = int((input_size[1] - one_slice_img.shape[1]) / 2) + random_data_y
            if y_start < 0:
                y_start = 0
            elif y_start + one_slice_img.shape[1] > input_size[1]:
                y_start = input_size[1] - one_slice_img.shape[1]
            resized_data_2 = np.zeros((input_size[1], input_size[1]))
            resized_data_2[:, y_start:y_start + one_slice_img.shape[1]] = resized_data_1[:, :]
        assert resized_data_2.shape == (input_size[0], input_size[1])
    return resized_data_2

# =============================================================================
# 预测一张图片的类别
# =============================================================================
def predict_data(net, imgs):
    result = net.predict(imgs)
    return result


# =============================================================================
# 预测完一套数据后的后处理
# =============================================================================
def after_process(clacess_of_all):
    """
    parameters:
        clacess_of_all: 模型预测出的初始分类结果
    function:
        对模型预测结果进行后处理，使得结果更为合理
    """
    final_result = []
    ###1、去除预测出None的值：
    pointer = 0
    for i, class_number in enumerate(clacess_of_all):
        if class_number != None:
            final_result.append(class_number)
        else:
            front = []
            back = []
            count = 1
            while pointer == 0:
                ### 如果None前后有相同的值，则赋值为此值
                if i-count >= 0 or i+count < len(clacess_of_all):
                    if i-count >= 0 and clacess_of_all[i-count] != None:
                        front.append(clacess_of_all[i-count])
                    if i+count < len(clacess_of_all) and clacess_of_all[i+count] != None:
                        back.append(clacess_of_all[i+count])
                    if len(front) and front[-1] in back:
#                            print(front, back)
                        final_result.append(front[-1])
                        pointer = 1
                        break
                    elif len(back) and back[-1] in front:
#                            print(front, back)
                        final_result.append(back[-1])
                        pointer = 1
                        break
                    count += 1
                ### 如果None前后没有相同的值，则赋值为None后的值
                else:
                    for ii in range(i, len(clacess_of_all)):
                        if clacess_of_all[ii] != None:
                            final_result.append(clacess_of_all[ii])
                            pointer = 1
                            break
                    ### 如果None后无值，则赋值为None前的值
                    if pointer != 1:
                        for ii in np.arange(i, -1, -1):
                            if clacess_of_all[ii] != None:
                                final_result.append(clacess_of_all[ii])
                                pointer = 1
                                break
            assert pointer == 1
        pointer = 0
#    print('no None result: ', '\n', final_result)
### 2、去除同一类里不是这一类的个别数据，>4个数判定为一类
    threshold_class = 4
#    for clacess_every in range(9,-1,-1):
    for clacess_every in range(10):
        if clacess_every not in final_result:
            continue
        clacess_all_index = [i for i, k in enumerate(final_result) if k == clacess_every]
        correct_clacess = [clacess_all_index[0]]
        for class_every_index in range(len(clacess_all_index)-1):
            if clacess_all_index[class_every_index+1] - clacess_all_index[class_every_index] <= threshold_class:
                correct_clacess.append(clacess_all_index[class_every_index+1])
            else:
                for correct_index in range(correct_clacess[0], correct_clacess[-1]+1):
                    final_result[correct_index] = clacess_every
                correct_clacess = [clacess_all_index[class_every_index+1]]

            for correct_index in range(correct_clacess[0], correct_clacess[-1]+1):
                final_result[correct_index] = clacess_every
#    print('delete little wrong result: ', '\n',  final_result)
#简单判断是递增还是递减
#如果是递增,pointer_up = 1
#如果是递减,pointer_down = 1
#将数据转换为从大到小
    threshold_up_down = 20
    if threshold_up_down > len(final_result):
        threshold_up_down = len(final_result)
    start_sum = np.sum(final_result[:threshold_up_down])
    counts_start = np.bincount(final_result[:threshold_up_down])
    number_start = np.argmax(counts_start)
    end_sum = np.sum(final_result[len(final_result)-threshold_up_down:])
    counts_end = np.bincount(final_result[len(final_result)-threshold_up_down:])
    number_end = np.argmax(counts_end)
#    print(number_start, number_end, start_sum, end_sum)
### 用前后端最大值计算
#    if number_start > number_end:
    ### 用前后端和计算
    if start_sum > end_sum:
        pointer_down = 1
        pointer_up = 0
    else:
        pointer_down = 0
        pointer_up = 1
        final_result.reverse()
#    print('sort from 9-0: ',  '\n', final_result)
    threshold = 20
    if threshold > len(final_result):
        threshold = len(final_result)
        ### 解决头部分类错误问题
    for singal_number in range(len(final_result)-threshold, len(final_result)-1):    #取最后20层
        if final_result[singal_number]-final_result[singal_number+1] >= 0:
            continue
        else:
#            print('final_result[singal_number]', singal_number, final_result[singal_number])
            if final_result[singal_number] == 1:
#                final_result[singal_number+1] == 0
                final_result[singal_number+1] = 0
            else:
#                final_result[singal_number+1] = final_result[singal_number] - 1
                final_result[singal_number+1] = final_result[singal_number]
#            print('final_result[singal_number+1]', singal_number+1, final_result[singal_number+1])
    if pointer_up:
        final_result.reverse()
#保证从小到大
    if pointer_down:
        final_result.reverse()
#    print('0 class fix: ',  '\n', final_result)
#到这一步,假设已经解决了开头第0类别分类错误问题
#然后解决临界处类别错误,将其归为前一类
    pointer = 1
    while pointer == 1:
        pointer = 0
        for i in range(len(final_result)-1):
            if final_result[i] - final_result[i+1] == -1 or final_result[i] - final_result[i+1] == 0:
                continue
            elif final_result[i] - final_result[i+1] < -1:
                for j in range(final_result[i+1]):
                    if j in final_result[i+1:]:
                        final_result[i+1] = final_result[i]
                        pointer = 1
                        break
                else:
                    wrongNum = final_result[i]
                    for k in range(i, -1, -1):
                        if final_result[k] == wrongNum:
                            final_result[k] = final_result[i+1] - 1
                            pointer = 1
            else:
                final_result[i+1] = final_result[i]
                pointer = 1
#    print('joint fix: ',  '\n', final_result)
    if pointer_down:
        final_result.reverse()
    return final_result