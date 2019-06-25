import numpy as np
from shelper import ShowImage, find_counters_by,draw_coords_img, ExtractInfo,findBodyCenter,cutting,DataNormalize,ResizeChannel,Get_coords, draw_coords_img_public
from keras import models
from metrics import dice_coefficient_loss, dice_coef
import os
from PIL import Image
from skimage import io
import cv2

def generate(data_path):
    """
        生成数据
    """
    paths = os.listdir(data_path)
    train_image = []
    train_mask = []
    for path in paths:
        try:
            temp_train = np.load( os.path.join(data_path, path, 'ct_data.npy') )
            temp_mask = np.load( os.path.join(data_path, path, 'CTV 1.npy'))

            temp_train += -1024
            temp_train[temp_train < -1500] = -1024
            temp_train[temp_train > 2976] = 2976
            temp_train[temp_train > 200] = 200
            temp_train[temp_train < -150] = -150

            temp_train, temp_mask = ExtractInfo(temp_train, temp_mask)

            center = findBodyCenter(temp_train[temp_train.shape[0] // 2])
            bias_x = int(center[0] - temp_train[0].shape[0] / 2)
            bias_y = int(center[1] - temp_train[0].shape[1] / 2)

            cut_img = cutting(288, 0, temp_train, bias_x, 50)
            cut_mask = cutting(288, 0, temp_mask, bias_x, 50)

            cut_img = np.expand_dims(cut_img, 3)
            cut_mask = np.expand_dims(cut_mask, 3)

            train_image.append(cut_img)
            train_mask.append(cut_mask)
        except:
            print("提取{}出现异常，已略过...".format(str(path)))
    train_image = np.concatenate(([_image for _image in train_image]), axis=0)
    train_mask = np.concatenate(([_mask for _mask in train_mask]), axis=0)
    return train_image, train_mask

def eval2d(data_path, model_path):
    eval_img, eval_mask = generate(data_path)
    model = models.load_model(model_path, custom_objects={'dice_coefficient_loss': dice_coefficient_loss, 'dice_coef': dice_coef})
    prediction = model.predict(eval_img, batch_size=8)

    prediction = np.squeeze(prediction, 3)
    eval_img = np.squeeze(eval_img, 3)
    eval_mask = np.squeeze(eval_mask, 3)
    _sum = 0
    pic_save_path = r'E:\rel2'
    for i in range(prediction.shape[0]):

        tep_pre = prediction[i]
        tep_org = eval_mask[i]
        tep_pre[tep_pre < 0.5] = 0
        tep_pre[tep_pre > 0.5] = 1
        tot = tep_pre * tep_org
        dice = 2* sum(tot[tot == 1]) / (sum(tep_pre[tep_pre ==1]) + sum(tep_org[tep_org == 1]))
        _sum += dice
        print('当前第{}张的dice系数为{}'.format(str(i), str(dice)))
        _image = np.zeros_like(eval_img[i])
        _image=_image.astype(np.uint8)
        _image = ResizeChannel(_image)
        pre_coords = Get_coords(tep_pre)
        eval_coords = Get_coords(tep_org)
        _image = draw_coords_img_public(_image, pre_coords, eval_coords, (0, 180, 255), (238, 104, 123), (60, 20, 220))

        # 提取轮廓，并且在原图中画出来
        # pre_coords = find_counters_by(prediction[i], value=1)
        # eval_coords = find_counters_by(eval_mask[i])
        # _image = draw_coords_img(eval_img[i], pre_coords, eval_coords, 50, 250)
        # _image = eval_img[i]
        # _image = DataNormalize(_image)
        # _image=_image.astype(np.uint8)
        # _image = ResizeChannel(_image)
        # _image = draw_coords_img(_image, pre_coords, eval_coords, (0, 180, 255), (238, 104, 123))


        # ShowImage(1, _image)
        cv2.imwrite(os.path.join(pic_save_path, str(i) +'.jpg'), _image)
    ave = _sum / prediction.shape[0]
    print('平均dice系数为{}'.format(str(ave)))

if __name__ == '__main__':
    """
        验证乳腺癌的BN-Unet模型
    """
    data_path = r'E:\masks'
    model_path = r'F:\Breast-test\mask right model\unet2d.h5'

    eval2d(data_path, model_path)
