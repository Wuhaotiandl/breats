import numpy as np
from shelper import *
from unet2d_model import unet
import os
from keras import backend as K



def get_first_index(mask):
    for i in range(mask.shape[0]):
        varifyImg=mask[i, :, :]
        numbers=sum(varifyImg[varifyImg == 1])
        if numbers > 0:
            return i


def assembly(data_path):
    """
        组装数据
    """
    path_list=os.listdir(data_path)
    original_img=[]
    original_mask=[]
    for path in path_list:
        img_path=os.path.join(data_path, path, 'ct_data.npy')
        mask_path=os.path.join(data_path, path, 'CTV 1.npy')
        if not os.path.exists(mask_path):
            mask_path=os.path.join(data_path, path, 'CTV.npy')
        img=np.load(img_path)
        mask=np.load(mask_path)
        img_1, mask_1=ExtractInfo(img, mask)
        img_1[img_1 < -1500]=-1024
        img_1, mask_1=ExtractInfo(img, mask)
        img_1+=-1024
        img_1[img_1 > 2976]=2976
        img_1[img_1 > 200]=200
        img_1[img_1 < -150]=-150
        center=findBodyCenter(img_1[img_1.shape[0] // 2])
        bias_x=int(center[0] - img_1[0].shape[0] / 2)
        bias_y=int(center[1] - img_1[0].shape[1] / 2)

        cut_img=cutting(288, 0, img_1, bias_x, bias_y)
        cut_mask=cutting(288, 0, mask_1, bias_x, bias_y)
        # for i in  range(cut_img.shape[0]):
        #     ShowImage(1, cut_img[i], cut_mask[i])
        original_img.append(cut_img)
        original_mask.append(cut_mask)
    train_mask=np.concatenate([_mask for _mask in original_mask], axis=0)
    train_img=np.concatenate([_img for _img in original_img], axis=0)
    return train_img, train_mask


def train(data_path, model_save_path):
    train_img, train_mask=assembly(data_path)
    train_img=np.expand_dims(train_img, 3)
    train_mask=np.expand_dims(train_mask, 3)

    model=unet()
    model.fit(train_img, train_mask,epochs=10, batch_size=2, validation_split=0.3, shuffle=True)
    model.save(model_save_path)

if __name__ == '__main__':
    data_path=r'F:\Breast-test\masks right'
    model_save_path=r'F:\Breast-test\masks right\unet2d.h5'
    train(data_path, model_save_path)




