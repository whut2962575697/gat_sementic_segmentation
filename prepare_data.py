# -*- encoding: utf-8 -*-
'''
@File    :   prepare_data.py.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2020, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/7/15 14:20   xin      1.0         None
'''

import numpy as np
from skimage.io import imread, imsave
import pickle
import os
import json
import random
import shutil
from PIL import Image
import cv2


# get node features and edge adj matrix


def calculate_feature(filename, save_path, small_roi, large_roi, gt, img):
    small_roi_img = imread(small_roi)
    large_roi_img = imread(large_roi)
    gt_img= imread(gt)
    rs_img = imread(img)
    obj_map = {}
    node_num = 0
    feature_dim = 3
    # n_cls = 12
    for i, small_roi_row, large_roi_row, gt_row, rs_row in zip(range(small_roi_img.shape[0]), small_roi_img, large_roi_img, gt_img, rs_img):
        for j, small_roi_cell, large_roi_cell, gt_cell, rs_cell in zip(range(small_roi_img.shape[1]), small_roi_row, large_roi_row, gt_row, rs_row):
            if large_roi_cell not in obj_map:
                obj_map[large_roi_cell] = {}
            if small_roi_cell not in obj_map[large_roi_cell]:
                node_num = node_num + 1
                obj_map[large_roi_cell][small_roi_cell] = {'feature_idx':[(i, j)], 'x_min': i, 'y_min': j, 'x_max': i, 'y_max': j, 'gt': {gt_cell: 1}, 'features':[rs_cell]}
            else:
                obj_map[large_roi_cell][small_roi_cell]['feature_idx'].append((i, j))
                obj_map[large_roi_cell][small_roi_cell]['features'].append(rs_cell)
                if i > obj_map[large_roi_cell][small_roi_cell]['x_max']:
                    obj_map[large_roi_cell][small_roi_cell]['x_max'] = i
                if j > obj_map[large_roi_cell][small_roi_cell]['y_max']:
                    obj_map[large_roi_cell][small_roi_cell]['y_max'] = j
                if gt_cell not in obj_map[large_roi_cell][small_roi_cell]['gt']:
                    obj_map[large_roi_cell][small_roi_cell]['gt'][gt_cell] = 1
                else:
                    obj_map[large_roi_cell][small_roi_cell]['gt'][gt_cell] = obj_map[large_roi_cell][small_roi_cell]['gt'][gt_cell] + 1
    adj_mat = np.zeros((node_num, node_num)).astype(np.uint8)
    feature_mat = np.zeros((node_num, feature_dim)).astype(np.float32)
    label_mat = np.zeros((node_num)).astype(np.uint8)
    roi_mat = np.zeros((node_num, 5)).astype(np.uint8)
    n_d = 0
    mask_json = []
    for large_obj_id, large_obj in obj_map.items():
        n_id_list = []

        for small_obj_id, small_obj in large_obj.items():
            mask_json.append(small_obj['feature_idx'])
            n_id_list.append(n_d)
            fea = [0, 0, 0]
            for feature in small_obj['features']:
                fea[0] = fea[0] + feature[0]/ 255.0
                fea[1] = fea[1] + feature[1]/ 255.0
                fea[2] = fea[2] + feature[2]/ 255.0
            fea[0] = fea[0] / len(small_obj['features'])
            fea[1] = fea[1] / len(small_obj['features'])
            fea[2] = fea[2] / len(small_obj['features'])
            feature_mat[n_d] = fea
            roi_mat[n_d] = [0, small_obj['x_min'], small_obj['y_min'], small_obj['x_max'], small_obj['y_max']]
            main_cls = [0, 0]
            for _cls, count in small_obj['gt'].items():
                if count>main_cls[1]:
                    main_cls[0] = _cls
                    main_cls[1] = count
            label_mat[n_d] = main_cls[0]-1

            n_d = n_d + 1
        for n_id_1 in n_id_list:
            for n_id_2 in n_id_list:
                adj_mat[n_id_1, n_id_2] = 1
    print(adj_mat)
    print(feature_mat)
    print(label_mat)
    print(roi_mat)
    if not os.path.exists(os.path.join(save_path, 'imgs')):
        os.mkdir(os.path.join(save_path, 'imgs'))
    if not os.path.exists(os.path.join(save_path, 'node_features')):
        os.mkdir(os.path.join(save_path, 'node_features'))
    if not os.path.exists(os.path.join(save_path, 'roi')):
        os.mkdir(os.path.join(save_path, 'roi'))
    if not os.path.exists(os.path.join(save_path, 'edge_adjs')):
        os.mkdir(os.path.join(save_path, 'edge_adjs'))
    if not os.path.exists(os.path.join(save_path, 'obj_masks')):
        os.mkdir(os.path.join(save_path, 'obj_masks'))
    if not os.path.exists(os.path.join(save_path, 'labels')):
        os.mkdir(os.path.join(save_path, 'labels'))
    shutil.copy(img, os.path.join(save_path, 'imgs', filename+'.tif'))
    with open(os.path.join(save_path, 'node_features', filename+'.pkl'), 'wb') as f:
        pickle.dump(feature_mat, f)  # 序列化
    with open(os.path.join(save_path, 'roi', filename+'.pkl'), 'wb') as f:
        pickle.dump(roi_mat, f)  # 序列化
    with open(os.path.join(save_path, 'edge_adjs', filename+'.pkl'), 'wb') as f:
        pickle.dump(adj_mat, f)  # 序列化
    with open(os.path.join(save_path, 'labels', filename+'.pkl'), 'wb') as f:
        pickle.dump(label_mat, f)  # 序列化
    with open(os.path.join(save_path, 'obj_masks', filename+'.json'), 'w') as f:
        json.dump(mask_json, f)


def calculate_obj(filename, save_path, small_roi, large_roi, gt, img):
    small_roi_img = imread(small_roi)
    large_roi_img = imread(large_roi)
    gt_img = imread(gt)
    rs_img = imread(img)
    obj_map = {}
    node_num = 0
    feature_dim = 3
    # n_cls = 12
    for i, small_roi_row, large_roi_row, gt_row, rs_row in zip(range(small_roi_img.shape[0]), small_roi_img,
                                                               large_roi_img, gt_img, rs_img):
        for j, small_roi_cell, large_roi_cell, gt_cell, rs_cell in zip(range(small_roi_img.shape[1]), small_roi_row,
                                                                       large_roi_row, gt_row, rs_row):
            if large_roi_cell not in obj_map:
                obj_map[large_roi_cell] = {}
            if small_roi_cell not in obj_map[large_roi_cell]:
                node_num = node_num + 1
                # if small_roi_cell == 25897:
                #     print(i, j)
                obj_map[large_roi_cell][small_roi_cell] = {'feature_idx': [(i, j)], 'x_min': j, 'y_min': i, 'x_max': j,
                                                           'y_max': i, 'gt': {gt_cell: 1}, 'features': [rs_cell]}
            else:
                obj_map[large_roi_cell][small_roi_cell]['feature_idx'].append((i, j))
                obj_map[large_roi_cell][small_roi_cell]['features'].append(rs_cell)
                if j < obj_map[large_roi_cell][small_roi_cell]['x_min']:
                    obj_map[large_roi_cell][small_roi_cell]['x_min'] = j
                if j > obj_map[large_roi_cell][small_roi_cell]['x_max']:
                    obj_map[large_roi_cell][small_roi_cell]['x_max'] = j
                if i < obj_map[large_roi_cell][small_roi_cell]['y_min']:
                    obj_map[large_roi_cell][small_roi_cell]['y_min'] = i
                if i > obj_map[large_roi_cell][small_roi_cell]['y_max']:
                    obj_map[large_roi_cell][small_roi_cell]['y_max'] = i
                if gt_cell not in obj_map[large_roi_cell][small_roi_cell]['gt']:
                    obj_map[large_roi_cell][small_roi_cell]['gt'][gt_cell] = 1
                else:
                    obj_map[large_roi_cell][small_roi_cell]['gt'][gt_cell] = \
                    obj_map[large_roi_cell][small_roi_cell]['gt'][gt_cell] + 1

    for large_obj_id, large_obj in obj_map.items():
        for small_obj_id, small_obj in large_obj.items():
            print(small_obj['x_min'], small_obj['y_min'], small_obj['x_max'], small_obj['y_max'])
            if small_obj['x_max'] - small_obj['x_min'] == 0 or small_obj['y_max'] - small_obj['y_min'] == 0:
                node_num = node_num - 1

    adj_mat = np.zeros((node_num, node_num)).astype(np.uint8)
    feature_mat = np.zeros((node_num, feature_dim)).astype(np.float32)
    label_mat = np.zeros((node_num)).astype(np.uint8)
    roi_mat = np.zeros((node_num, 5)).astype(np.uint8)
    n_d = 0
    mask_json = []
    mask_objs = np.zeros((node_num, 224, 224)).astype(np.uint8)
    resized_mask_objs = []
    for large_obj_id, large_obj in obj_map.items():
        n_id_list = []

        for small_obj_id, small_obj in large_obj.items():

            if small_obj['x_max'] - small_obj['x_min'] == 0 or small_obj['y_max'] - small_obj['y_min'] == 0:
                continue

            mask_json.append(small_obj['feature_idx'])
            print(len(small_obj['feature_idx']))
            print(small_obj['x_min'], small_obj['y_min'], small_obj['x_max'], small_obj['y_max'])
            for (i_x, j_y) in small_obj['feature_idx']:
                # print(i_x, j_y)
                mask_objs[n_d, i_x, j_y] = 1
            print(np.sum(mask_objs[n_d]))
            cv2.imwrite(r'D:\new_dataset\new_dataset\gat\temp/'+filename+'_0_'+str(n_d)+'.jpg', mask_objs[n_d])
            # scipy.misc.toimage(mask_objs[n_d], cmin=0.0, cmax=...).save('outfile.jpg')
            # scipy.misc.imsave(r'D:\new_dataset\new_dataset\gat\temp/'+filename+'_'+str(n_d)+'.jpg', mask_objs[n_d])
            # imsave(r'D:\new_dataset\new_dataset\gat\temp/'+filename+'_'+str(n_d)+'.jpg', mask_objs[n_d])
            new_mask_obj = mask_objs[n_d, small_obj['y_min']:small_obj['y_max'], small_obj['x_min']:small_obj['x_max']]
            print(n_d, new_mask_obj.shape)
            # new_img = Image.fromarray(new_mask_obj).resize((7, 7))
            new_img = cv2.resize(new_mask_obj, (7, 7))
            # tt = Image.fromarray(mask_objs[n_d]).save(r'D:\new_dataset\new_dataset\gat\temp/'+filename+'_'+str(n_d)+'.jpg')
            cv2.imwrite(r'D:\new_dataset\new_dataset\gat\temp/' + filename + '_' + str(n_d) + '.jpg', new_img)
            # with open(r'D:\new_dataset\new_dataset\gat\temp/'+filename+'_'+str(n_d)+'.jpg', 'w') as f:
            #     tt.save(f)
            resized_mask_objs.append(np.array(new_img))

            n_id_list.append(n_d)
            fea = [0, 0, 0]
            for feature in small_obj['features']:
                fea[0] = fea[0] + feature[0] / 255.0
                fea[1] = fea[1] + feature[1] / 255.0
                fea[2] = fea[2] + feature[2] / 255.0
            fea[0] = fea[0] / len(small_obj['features'])
            fea[1] = fea[1] / len(small_obj['features'])
            fea[2] = fea[2] / len(small_obj['features'])
            feature_mat[n_d] = fea
            roi_mat[n_d] = [0, small_obj['x_min'], small_obj['y_min'], small_obj['x_max'], small_obj['y_max']]
            main_cls = [0, 0]
            for _cls, count in small_obj['gt'].items():
                if count > main_cls[1]:
                    main_cls[0] = _cls
                    main_cls[1] = count
            label_mat[n_d] = main_cls[0] - 1

            n_d = n_d + 1
        for n_id_1 in n_id_list:
            for n_id_2 in n_id_list:
                adj_mat[n_id_1, n_id_2] = 1
    resized_mask_objs = np.array(resized_mask_objs)
    print(adj_mat)
    print(resized_mask_objs)
    print(feature_mat)
    print(label_mat)
    print(roi_mat)
    if not os.path.exists(os.path.join(save_path, 'imgs')):
        os.mkdir(os.path.join(save_path, 'imgs'))
    if not os.path.exists(os.path.join(save_path, 'node_features')):
        os.mkdir(os.path.join(save_path, 'node_features'))
    if not os.path.exists(os.path.join(save_path, 'mask_objs')):
        os.mkdir(os.path.join(save_path, 'mask_objs'))
    if not os.path.exists(os.path.join(save_path, 'roi')):
        os.mkdir(os.path.join(save_path, 'roi'))
    if not os.path.exists(os.path.join(save_path, 'edge_adjs')):
        os.mkdir(os.path.join(save_path, 'edge_adjs'))
    if not os.path.exists(os.path.join(save_path, 'obj_masks')):
        os.mkdir(os.path.join(save_path, 'obj_masks'))
    if not os.path.exists(os.path.join(save_path, 'labels')):
        os.mkdir(os.path.join(save_path, 'labels'))
    shutil.copy(img, os.path.join(save_path, 'imgs', filename + '.tif'))
    with open(os.path.join(save_path, 'node_features', filename + '.pkl'), 'wb') as f:
        pickle.dump(feature_mat, f)  # 序列化
    with open(os.path.join(save_path, 'mask_objs', filename + '.pkl'), 'wb') as f:
        pickle.dump(resized_mask_objs, f)  # 序列化
    with open(os.path.join(save_path, 'roi', filename + '.pkl'), 'wb') as f:
        pickle.dump(roi_mat, f)  # 序列化
    with open(os.path.join(save_path, 'edge_adjs', filename + '.pkl'), 'wb') as f:
        pickle.dump(adj_mat, f)  # 序列化
    with open(os.path.join(save_path, 'labels', filename + '.pkl'), 'wb') as f:
        pickle.dump(label_mat, f)  # 序列化
    with open(os.path.join(save_path, 'obj_masks', filename + '.json'), 'w') as f:
        json.dump(mask_json, f)


def main(roi_small_path, roi_large_path, gt_path, rs_img_path, save_path):
    filenames = [x for x in os.listdir(rs_img_path) if x.endswith('.tif')]
    for filename in filenames:
        calculate_obj(filename.strip('.tif'), save_path,
                          os.path.join(roi_small_path, filename),
                          os.path.join(roi_large_path, filename),
                          os.path.join(gt_path, filename),
                          os.path.join(rs_img_path, filename))

def split_trainval(roi_small_path, roi_large_path, gt_path, rs_img_path, save_path):
    filenames = [x for x in os.listdir(rs_img_path) if x.endswith('.tif')]
    random.shuffle(filenames)
    os.mkdir(os.path.join(save_path, 'train'))
    os.mkdir(os.path.join(save_path, 'train', 'roi_small'))
    os.mkdir(os.path.join(save_path, 'train', 'roi_large'))
    os.mkdir(os.path.join(save_path, 'train', 'gt'))
    os.mkdir(os.path.join(save_path, 'train', 'rs'))
    os.mkdir(os.path.join(save_path, 'val'))
    os.mkdir(os.path.join(save_path, 'val', 'roi_small'))
    os.mkdir(os.path.join(save_path, 'val', 'roi_large'))
    os.mkdir(os.path.join(save_path, 'val', 'gt'))
    os.mkdir(os.path.join(save_path, 'val', 'rs'))
    for filename in filenames[:int(0.7*len(filenames))]:
        shutil.copy(os.path.join(roi_small_path, filename), os.path.join(save_path, 'train', 'roi_small', filename))
        shutil.copy(os.path.join(roi_large_path, filename), os.path.join(save_path, 'train', 'roi_large', filename))
        shutil.copy(os.path.join(gt_path, filename), os.path.join(save_path, 'train', 'gt', filename))
        shutil.copy(os.path.join(rs_img_path, filename), os.path.join(save_path, 'train', 'rs', filename))

    for filename in filenames[int(0.7*len(filenames)):]:
        shutil.copy(os.path.join(roi_small_path, filename), os.path.join(save_path, 'val', 'roi_small', filename))
        shutil.copy(os.path.join(roi_large_path, filename), os.path.join(save_path, 'val', 'roi_large', filename))
        shutil.copy(os.path.join(gt_path, filename), os.path.join(save_path, 'val', 'gt', filename))
        shutil.copy(os.path.join(rs_img_path, filename), os.path.join(save_path, 'val', 'rs', filename))





if __name__ == "__main__":
    # a = imread(r'C:\Users\xin\Pictures/4cee953dc58bff6f31fef61e58cd92cc.png')
    # print(a.shape)
    # calculate_obj('0.tif'.strip('.tif'), r'',
    #               os.path.join(r'D:\new_dataset\new_dataset\roi_small1\roi_small1\raster_output_16', '0.tif'),
    #               os.path.join(r'D:\new_dataset\new_dataset\roi_large\raster_output_16', '0.tif'),
    #               os.path.join(r'D:\new_dataset\new_dataset\gt\raster_output_8', '0.tif'),
    #               os.path.join(r'D:\new_dataset\new_dataset\img\raster_output_8', '0.tif'))
    # main(r'D:\new_dataset\new_dataset\trainval_datatset\train\roi_small', r'D:\new_dataset\new_dataset\trainval_datatset\train\roi_large',
    #      r'D:\new_dataset\new_dataset\trainval_datatset\train\gt', r'D:\new_dataset\new_dataset\trainval_datatset\train\rs',
    #      r'D:\new_dataset\new_dataset\gat\train')
    #
    # main(r'D:\new_dataset\new_dataset\trainval_datatset\val\roi_small',
    #      r'D:\new_dataset\new_dataset\trainval_datatset\val\roi_large',
    #      r'D:\new_dataset\new_dataset\trainval_datatset\val\gt',
    #      r'D:\new_dataset\new_dataset\trainval_datatset\val\rs',
    #      r'D:\new_dataset\new_dataset\gat\val')
    main(r'D:\trainval_datatset\train\roi_small',
         r'D:\trainval_datatset\train\roi_large',
         r'D:\trainval_datatset\train\gt',
         r'D:\trainval_datatset\train\rs',
         r'D:\gat_dataset\train')

    main(r'D:\trainval_datatset\val\roi_small',
         r'D:\trainval_datatset\val\roi_large',
         r'D:\trainval_datatset\val\gt',
         r'D:\trainval_datatset\val\rs',
         r'D:\gat_dataset\val')
    # split_trainval(r'D:\new_dataset\new_dataset\roi_small\raster_output_16', r'D:\new_dataset\new_dataset\roi_large\raster_output_16',
    #                r'D:\new_dataset\new_dataset\gt\raster_output_8', r'D:\new_dataset\new_dataset\img\raster_output_8', r'D:\new_dataset\new_dataset\trainval_datatset')
    # calculate_feature('0', r'D:\new_dataset\new_dataset\test', r'D:\new_dataset\new_dataset\roi_small\raster_output_16/0.tif',
    #                   r'D:\new_dataset\new_dataset\roi_large\raster_output_16/0.tif', r'D:\new_dataset\new_dataset\gt\raster_output_8/0.tif',
    #                   r'D:\new_dataset\new_dataset\img\raster_output_8/0.tif')













