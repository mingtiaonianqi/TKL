import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import numpy as np
from typing import Tuple, Optional
import logging
from tqdm import tqdm
import sys

class ScannetDatasetWholeScene:
    def __init__(self, root, block_points=4096, split='test', test_area=2, stride=0.5, block_size=1.0, padding=0.001):
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.root = root
        self.split = split
        self.stride = stride
        self.scene_points_num = []
        assert split in ['train', 'test']
        if self.split == 'train':
            self.file_list = [d for d in os.listdir(root) if 'Area_{}'.format(test_area) not in d]
        else:
            self.file_list = [d for d in os.listdir(root) if 'Area_{}'.format(test_area) in d]
            
        self.scene_points_list = []
        self.semantic_labels_list = []
        self.room_coord_min, self.room_coord_max = [], []
        for file in self.file_list:
            data = np.load(root + file)
            points = data[:, :3]
            self.scene_points_list.append(data[:, :7])  
            self.semantic_labels_list.append(data[:, 7])  
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_coord_min.append(coord_min)
            self.room_coord_max.append(coord_max)
            
        labelweights = np.zeros(5)
        for seg in self.semantic_labels_list:
            tmp, _ = np.histogram(seg, bins=range(6))
            self.scene_points_num.append(seg.shape[0])
            labelweights += tmp
            
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        points = point_set_ini[:,:7]    #修改为了6
        labels = self.semantic_labels_list[index]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
        #加了两行网格大小的输出
        print("网格的X和Y")
        print(grid_x)
        print(grid_y)
        data_room, label_room, sample_weight, index_room = np.array([]), np.array([]), np.array([]),  np.array([])
        for index_y in range(0, grid_y):
            for index_x in range(0, grid_x):
                s_x = coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, coord_max[0])
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size
                point_idxs = np.where(
                    (points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) & (points[:, 1] >= s_y - self.padding) & (points[:, 1] <= e_y + self.padding))[0]
                if point_idxs.size == 0:
                    continue
                num_batch = int(np.ceil(point_idxs.size / self.block_points))
                point_size = int(num_batch * self.block_points)
                replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
                point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
                np.random.shuffle(point_idxs)
                data_batch = points[point_idxs, :]
                normlized_xyz = np.zeros((point_size, 3))
                normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
                normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
                normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
                data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size / 2.0)
                data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size / 2.0)
                # data_batch[:, 3:6] /= 255.0
                data_batch[:, 3:6] /= 255.0     #把注释的这一行加上了  应该是对颜色的处理
                data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
                label_batch = labels[point_idxs].astype(int)
                batch_weight = self.labelweights[label_batch]

                data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
                label_room = np.hstack([label_room, label_batch]) if label_room.size else label_batch
                sample_weight = np.hstack([sample_weight, batch_weight]) if label_room.size else batch_weight
                index_room = np.hstack([index_room, point_idxs]) if index_room.size else point_idxs
        data_room = data_room.reshape((-1, self.block_points, data_room.shape[1]))
        label_room = label_room.reshape((-1, self.block_points))
        sample_weight = sample_weight.reshape((-1, self.block_points))
        index_room = index_room.reshape((-1, self.block_points))
        return data_room, label_room, sample_weight, index_room

    def __len__(self):
        return len(self.scene_points_list)


class get_model(nn.Module):
    def __init__(self, num_classes):
        super(get_model, self).__init__()
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 10 + 3, [32, 32, 64], False)     # ys这里为12+3， 我修改为了9+3 ys输入为9维数据，我这里与S3DIS一致，使用的是6维数据
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
        self.args = self.arg()


    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l4_points

    def predict(self, data_path):
        def log_string(str):
            logger.info(str)
            print(str)
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        ROOT_DIR = BASE_DIR
        sys.path.append(os.path.join(ROOT_DIR, 'models'))

        classes = ['ground', 'tree','cable','track']
        class2label = {cls: i for i, cls in enumerate(classes)}
        seg_classes = class2label
        seg_label_to_cat = {}

        for i, cat in enumerate(seg_classes.keys()):
            seg_label_to_cat[i] = cat

        args = self.args
        NUM_CLASSES = 4
        BATCH_SIZE = args.batch_size
        NUM_POINT = args.num_point

        root = data_path
        TEST_DATASET_WHOLE_SCENE = ScannetDatasetWholeScene(root, split='test', test_area=args.test_area, block_points=NUM_POINT, stride=4.6, block_size=5, padding=0.0066)
        
        scene_id = TEST_DATASET_WHOLE_SCENE.file_list
        scene_id = [x[:-4] for x in scene_id]
        num_batches = len(TEST_DATASET_WHOLE_SCENE)

        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
        ckpt = torch.load('best_model.pth')
        classifier = self.cuda()
        classifier.load_state_dict(ckpt['model_state_dict'])
        classifier.eval()

        logger = logging.getLogger("Model")
        logger.setLevel(logging.INFO)
        with torch.no_grad():
            flage = False
            scene_id = TEST_DATASET_WHOLE_SCENE.file_list
            scene_id = [x[:-4] for x in scene_id]
            num_batches = len(TEST_DATASET_WHOLE_SCENE)

            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

            log_string('---- EVALUATION WHOLE SCENE----')

            for batch_idx in range(num_batches):
                print("Inference [%d/%d] %s ..." % (batch_idx + 1, num_batches, scene_id[batch_idx]))
                total_seen_class_tmp = [0 for _ in range(NUM_CLASSES)]
                print("step1")
                total_correct_class_tmp = [0 for _ in range(NUM_CLASSES)]
                print("step2")
                total_iou_deno_class_tmp = [0 for _ in range(NUM_CLASSES)]
                print("step3")

                print("step4")
                whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx]
                whole_scene_label = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[batch_idx]
                vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))
                print("step50")
                for _ in tqdm(range(args.num_votes), total=args.num_votes):
                    print("step51")
                    scene_data, scene_label, scene_smpw, scene_point_index = TEST_DATASET_WHOLE_SCENE[batch_idx]
                    print("step52")
                    num_blocks = scene_data.shape[0]
                    print("step53")
                    s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
                    print("step54")
                    batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 10))   #  这里把 12 改成了  9

                    print("step55")
                    batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
                    print("step56")
                    batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
                    print("step57")
                    batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT))

                    print("step6")
                    for sbatch in range(s_batch_num):
                        start_idx = sbatch * BATCH_SIZE
                        end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                        real_batch_size = end_idx - start_idx
                        batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                        batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
                        batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
                        batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]
                        batch_data[:, :, 3:5] /= 1.0         #这里把 9 改成了  6
                        
                        torch_data = torch.Tensor(batch_data)
                        torch_data = torch_data.float().cuda()
                        torch_data = torch_data.transpose(2, 1)
                        seg_pred, _ = classifier(torch_data)
                        if(flage != True):
                            np_save_data = torch_data.cpu().numpy().astype(np.float32)
                            output = seg_pred.cpu().numpy().astype(np.float32)
                            np.save("save_data_output.npy", output)
                            np.save("save_data.npy", np_save_data)
                            flage = True
                            print("successfully saved npy file!")
                        batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()


                        vote_label_pool = self.add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                                batch_pred_label[0:real_batch_size, ...],
                                                batch_smpw[0:real_batch_size, ...])

                pred_label = np.argmax(vote_label_pool, 1)

                print("step7")
                for l in range(NUM_CLASSES):
                    total_seen_class_tmp[l] += np.sum((whole_scene_label == l))
                    total_correct_class_tmp[l] += np.sum((pred_label == l) & (whole_scene_label == l))
                    total_iou_deno_class_tmp[l] += np.sum(((pred_label == l) | (whole_scene_label == l)))
                    total_seen_class[l] += total_seen_class_tmp[l]
                    total_correct_class[l] += total_correct_class_tmp[l]
                    total_iou_deno_class[l] += total_iou_deno_class_tmp[l]

                print("step8")
                iou_map = np.array(total_correct_class_tmp) / (np.array(total_iou_deno_class_tmp, dtype=np.float32) + 1e-6)
                print(iou_map)
                arr = np.array(total_seen_class_tmp)
                tmp_iou = np.mean(iou_map[arr != 0])
                log_string('Mean IoU of %s: %.4f' % (scene_id[batch_idx], tmp_iou))
                print('----------------------------')

            IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float32) + 1e-6)
            iou_per_class_str = '------- IoU --------\n'
            for l in range(NUM_CLASSES):
                iou_per_class_str += 'class %s, IoU: %.3f \n' % (
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])),
                    total_correct_class[l] / float(total_iou_deno_class[l]))
            log_string(iou_per_class_str)
            log_string('eval point avg class IoU: %f' % np.mean(IoU))
            log_string('eval whole scene point avg class acc: %f' % (
                np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float32) + 1e-6))))
            log_string('eval whole scene point accuracy: %f' % (
                    np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)))

            print("Done!")

    
    def add_vote(self, vote_label_pool, point_idx, pred_label, weight):
        B = pred_label.shape[0]
        N = pred_label.shape[1]
        for b in range(B):
            for n in range(N):
                if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                    vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
        return vote_label_pool
        
    def arg(self):
        '''PARAMETERS'''
        parser = argparse.ArgumentParser('Model')
        parser.add_argument('--batch_size', type=int, default=2, help='batch size in testing [default: 32]')
        parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
        parser.add_argument('--num_point', type=int, default=32768, help='point number [default: 4096]')  #默认为4096
        parser.add_argument('--log_dir', type=str, default= '0626', help='experiment root')
        parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
        parser.add_argument('--test_area', type=int, default=2, help='area for testing, option: 1-6 [default: 5]')
        parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting [default: 5]')
        return parser.parse_args()

@torch.jit.script
def square_distance(src: torch.Tensor, dst: torch.Tensor)->torch.Tensor:
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


@torch.jit.script
def index_points(points: torch.Tensor, idx: torch.Tensor)->torch.Tensor:
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape = [view_shape[0]] +[1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


@torch.jit.script
def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


@torch.jit.script
def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor)-> torch.Tensor:
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


@torch.jit.script
def sample_and_group(npoint: int, radius: float, nsample: int, xyz: torch.Tensor, points: torch.Tensor, returnfps: bool=False
                     )-> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points, None, None


@torch.jit.script
def sample_and_group_all(xyz: torch.Tensor, points: torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points, _, _ = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]

        for bn, conv in zip(self.mlp_bns, self.mlp_convs):
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1: Optional[torch.Tensor], points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for bn, conv in zip(self.mlp_bns, self.mlp_convs):
            new_points = F.relu(bn(conv(new_points)))
        return new_points
    
if __name__ == '__main__':
    root = 'data_utils/data/'
    model = get_model(4)
    model.eval()
    # model_sc = torch.jit.script(model)
    # model_sc.save("best_model.pt")
    model.predict(root)