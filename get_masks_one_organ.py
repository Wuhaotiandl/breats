import os
import glob
import numpy as np
import pydicom
import re
import sys
from skimage import measure

# 查找roi的下标
def find_roi_id(rois, roi_pattern):
    for i, roi in enumerate(rois):
        if re.match(roi_pattern, roi, re.I):
            return i
    return -1

#从struct文件中读取轮廓信息
def read_data_from_dicom(rs_file, roi_pattern, spacing,masks, start_point):
    """
    :param rs_file: rtstructure 文件
    :param roi_pattern:  器官名称
    :param spacing: 像素间空隙
    :param masks: 原图大小的全1矩阵
    :param start_point: 序列的起始位置
    :return:
    """
    rois = [roi.ROIName for roi in rs_file.StructureSetROISequence]
    print(rois)
    print(roi_pattern)
    id = find_roi_id(rois, roi_pattern)
    print(id)
    count = 0
    if id > -1:
        # 判断当前的rs_file.ROIContourSequence[id]是否具有ContourSequence属性
        if hasattr(rs_file.ROIContourSequence[id], 'ContourSequence'):
            if masks.shape[0] < len(rs_file.ROIContourSequence[id].ContourSequence):
                print('-----------')
                return masks
            for contour in rs_file.ROIContourSequence[id].ContourSequence:
                contour_data = contour.ContourData
                if len(contour_data) < 9:
                    return masks
                contour_data = np.array(contour_data).reshape(-1, 3)
                X = (np.round((contour_data[:, 0] - start_point[0]) / spacing[2]).astype(np.int))
                Y = (np.round((contour_data[:, 1] - start_point[1]) / spacing[1]).astype(np.int))
                z = (np.round((contour_data[0, 2] - start_point[2]) / spacing[0]).astype(np.int))
                V_poly = np.stack([Y, X], axis=1)
                masks[z, :, :] = measure.grid_points_in_poly([512, 512], V_poly)
                count = count +1
        else:
            print('no this organ')
    else:
        print('no this organ')
    print(count)
    print((np.where(masks == 1))[0])
    return masks

def main(data_path):
    save_path = os.path.join(os.path.dirname(data_path))
    # 存放路径
    save_file = save_path + os.sep + 'masks'
    if not os.path.exists(save_file):
        os.makedirs(save_file)
    patient_list = os.listdir(data_path)
    patient_list = [os.path.join(data_path, _path) for _path in patient_list]
    for patient_path in patient_list:
        dicom_list = os.listdir(patient_path)

        all_slices = [pydicom.dcmread(os.path.join(patient_path, s)) for s in dicom_list]
        ct_slices = [s for s in all_slices if s.Modality == 'CT']
        ct_slices.sort(key=lambda x: int(x.ImagePositionPatient[2]), reverse=False)
        if len(ct_slices) > 1:
            try:
                slice_thickness = np.abs(ct_slices[0].ImagePositionPatient[2] - ct_slices[1].ImagePositionPatient[2])
            except:
                slice_thickness = np.abs(ct_slices[0].SliceLocation - ct_slices[1].SliceLocation)
        elif len(ct_slices) == 1:
            slice_thickness = ct_slices[0].SliceThickness
        else:
            sys.exit(1)
        start_point = np.array(ct_slices[0].ImagePositionPatient)
        Space = (ct_slices[0].PixelSpacing)
        temp_spacing = []
        for i in range(len(Space)):
            temp_spacing.append(float(Space[i]))
        # [slice_thickness] + temp_spacing 其中的temp_spacing一定要是个list，这行代码等于append效果
        spacing = map(float, ([slice_thickness] + temp_spacing))
        spacing = np.array(list(spacing))  # 分辨率和空间层厚
        image = np.stack([s.pixel_array for s in ct_slices])
        image = image.astype(np.int16)

        truth_files = glob.glob(os.path.join(patient_path, "RS*.dcm"))
        print(truth_files)
        if len(truth_files) == 0:
            print(patient_path)
            print('no struct')
            continue
        truth_rs = pydicom.dcmread(truth_files[0], force=True)
        patient_name = truth_rs.PatientName
        save_patient_path = save_file + os.sep + str(patient_name)
        if not os.path.exists(save_patient_path):
            os.makedirs(save_patient_path)
        np.save(os.path.join(save_patient_path, "ct_data"), image)

        rois_truth = [roi.ROIName for roi in truth_rs.StructureSetROISequence]

        for roi in rois_truth:
            roi_key = 'ctv'
            if re.search(roi_key, roi, re.IGNORECASE):
                masks = np.zeros(image.shape, np.int8)
                truth_masks = read_data_from_dicom(truth_rs, roi, spacing, masks, start_point)
                if np.max(truth_masks)>0:
                    np.save(os.path.join(save_patient_path, str(roi)), truth_masks)
                else:
                    print(patient_name)
                    print(roi)
            else:
                continue




if __name__ == '__main__':
    truth_path = r'F:\t1\left'
    main(truth_path)
