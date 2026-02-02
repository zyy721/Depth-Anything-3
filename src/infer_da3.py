import glob, os, torch
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.export.depth_vis import export_to_depth_vis

import os
import argparse
import pickle as pkl
from tqdm import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from pyquaternion import Quaternion
import numpy as np

import gc

class NuscenesDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def rt2mat(translation, quaternion=None, inverse=False, rotation=None):
    R = Quaternion(quaternion).rotation_matrix if rotation is None else rotation
    T = np.array(translation)
    if inverse:
        R = R.T
        T = -R @ T
    mat = np.eye(4)
    mat[:3, :3] = R
    mat[:3, 3] = T
    return mat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--single-gpu", action='store_true')

    args = parser.parse_args()

    rela_target_path = os.path.join('data', 'da3_rela_nusc')
    metric_target_path = os.path.join('data', 'da3_metric_nusc')

    torch.cuda.set_device(args.local_rank)
    if args.single_gpu:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', world_size=1, rank=0) 
    else:
        dist.init_process_group(backend='nccl') 
    device = torch.device("cuda", args.local_rank)

    with open('data/nuscenes_cam_w_plan/nuscenes_infos_train_sweeps_occ.pkl', "rb") as f:
        train_nusc_data = pkl.load(f)['infos']

    with open('data/nuscenes_cam_w_plan/nuscenes_infos_val_sweeps_occ.pkl', "rb") as f:
        val_nusc_data = pkl.load(f)['infos']

    nusc_data = {**train_nusc_data, **val_nusc_data}

    nusc_dataset = NuscenesDataset(list(range(len(nusc_data))))
    train_sampler = DistributedSampler(nusc_dataset, shuffle=False)
    trainloader = DataLoader(nusc_dataset, batch_size=1, num_workers=4, sampler=train_sampler)


    # device = torch.device("cuda")
    # model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1", cache_dir="/home/yzhu/.cache/huggingface/hub", local_files_only=True).to(device)

    model = model.to(device=device)
    # example_path = "assets/examples/SOH"
    # images = sorted(glob.glob(os.path.join(example_path, "*.png")))

    camera_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT']

    scene_name_list = list(nusc_data.keys())

    for index_data in tqdm(trainloader):
        scene = scene_name_list[index_data]
        for sample_idx, sample in enumerate(nusc_data[scene]):
            rela_scene_path = os.path.join(rela_target_path, sample['scene_name'])
            os.makedirs(rela_scene_path, exist_ok=True)

            metric_scene_path = os.path.join(metric_target_path, sample['scene_name'])
            os.makedirs(metric_scene_path, exist_ok=True)

            if sample['is_key_frame']:
                all_img_path_list, all_global2cam_list, all_intrinsics_list = [], [], []
                all_sweeps_idx_list = []
                # if (sample_idx - 1) >= 0:
                #     all_sweeps_idx_list.append(sample_idx - 1)
                all_sweeps_idx_list.append(sample_idx)
                # if (sample_idx + 1) < len(nusc_data[scene]):
                #     all_sweeps_idx_list.append(sample_idx + 1)

                for sweep_idx in all_sweeps_idx_list:
                    for cam in camera_names:
                        cur_img_path = nusc_data[scene][sweep_idx]['data'][cam]['filename']
                        cur_img_path = os.path.join('data/nuscenes', cur_img_path)
                        all_img_path_list.append(cur_img_path)

                        camera_intrinsic = nusc_data[scene][sweep_idx]['data'][cam]['calib']['camera_intrinsic']
                        camera_intrinsic = np.array(camera_intrinsic)
                        all_intrinsics_list.append(camera_intrinsic)

                        # global2cam
                        sensor2ego_translation = nusc_data[scene][sweep_idx]['data'][cam]['calib']['translation']
                        sensor2ego_rotation = nusc_data[scene][sweep_idx]['data'][cam]['calib']['rotation']
                        ego2global_translation = nusc_data[scene][sweep_idx]['data'][cam]['pose']['translation']
                        ego2global_rotation = nusc_data[scene][sweep_idx]['data'][cam]['pose']['rotation']
                        curr_ego2cam = rt2mat(sensor2ego_translation,
                                              sensor2ego_rotation,
                                              inverse=True)
                        curr_global2ego = rt2mat(ego2global_translation,
                                                 ego2global_rotation,
                                                 inverse=True)
                        curr_global2cam = curr_ego2cam @ curr_global2ego
                        all_global2cam_list.append(curr_global2cam)


                # path = "data/nuscenes/samples/CAM_FRONT"
                # path_A = os.path.join(path, "n008-2018-07-26-12-13-50-0400__CAM_FRONT__1532621862162404.jpg")
                # path_B = os.path.join(path, "n008-2018-07-26-12-13-50-0400__CAM_FRONT__1532621862662404.jpg")
                # path_C = os.path.join(path, "n008-2018-07-26-12-13-50-0400__CAM_FRONT__1532621863162404.jpg")

                # images = [path_A, path_B, path_C]  

                images = all_img_path_list
                extrinsics = np.stack(all_global2cam_list, axis=0)
                intrinsics = np.stack(all_intrinsics_list, axis=0)

                prediction_w_ext_int = model.inference(
                    images, extrinsics, intrinsics
                )

                sample_token = sample['token']
                # np.save(os.path.join(rela_scene_path, sample_token + '.npy'), prediction_w_ext_int.depth.astype(np.float16))
                np.save(os.path.join(metric_scene_path, sample_token + '.npy'), prediction_w_ext_int.depth.astype(np.float16))

                # export_to_depth_vis(prediction, "out/w_ext_int")


                # prediction_wo_ext_int = model.inference(
                #     images
                # )

                # export_to_depth_vis(prediction, "out/wo_ext_int")

                # # prediction.processed_images : [N, H, W, 3] uint8   array
                # print(prediction.processed_images.shape)
                # # prediction.depth            : [N, H, W]    float32 array
                # print(prediction.depth.shape)  
                # # prediction.conf             : [N, H, W]    float32 array
                # print(prediction.conf.shape)  
                # # prediction.extrinsics       : [N, 3, 4]    float32 array # opencv w2c or colmap format
                # print(prediction.extrinsics.shape)
                # # prediction.intrinsics       : [N, 3, 3]    float32 array
                # print(prediction.intrinsics.shape)


