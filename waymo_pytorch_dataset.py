import os
import numpy as np
from torch.utils.data import IterableDataset
from torch.utils.data import Dataset
import dataset_utils.calibration_waymo as calibration
import dataset_utils.object3d as object3d
from PIL import Image
import tqdm
import math

from simple_waymo_open_dataset_reader import WaymoDataFileReader
from simple_waymo_open_dataset_reader import dataset_pb2, label_pb2
from simple_waymo_open_dataset_reader import utils

class WaymoDataset(Dataset):
    '''Wymo dataset for pytorch
    CURRENT:
        V Serialized data feeding
    TODO:
        X Implement shuffling
        X Implement IterableDataset/BatchSampler
        X Make Cache
    
    USAGE:
        DATA_PATH = '/home/jupyter/waymo-od/waymo_dataset'
        LOCATIONS = ['location_sf']
        
        dataset = WaymoDataset(DATA_PATH, LOCATIONS, 'train', True, "new_waymo")
        
        frame, idx = dataset.data, dataset.count
        calib = dataset.get_calib(frame, idx)
        pts =  dataset.get_lidar(frame, idx)
        target = dataset.get_label(frame, idx)
    
    :param root_dir: Root directory of the data
    :param split: Select if train/test/val
    :param use_cache: Select if you need to save a pkl file of the dataset for easy access 
    '''
    def __init__(self, root_dir, locations, split='train', use_cache=False, name="Waymo"):
        self._name=name
        self.split = split
        is_test = self.split == 'test'
        
        self._dataset_dir = os.path.join(root_dir,'kitti_dataset', 'testing' if is_test else 'training')
        
        self.__lidar_list = ['_FRONT', '_FRONT_RIGHT', '_FRONT_LEFT', '_SIDE_RIGHT', '_SIDE_LEFT']
        self.__type_list = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']
        self.get_file_names() # Storing file names in object 
        
        self._image = None 
        self._num_files = len(self.__file_names)
        self._curr_counter = 0
        self._num_frames = 0
        self._total_frames = 0
        self._idx_to_frame = []
        self._sample_list = []
        self._frame_counter = -1 # Count the number of frames used per file
        self._file_counter = -1 # Count the number of files used
        self._dataset_nums = [] # Number of frames to be considered from each file (records+files)
        self._dataset_itr = None # tfRecord iterator
        self.num_sample = self._total_frames
        
        if use_cache: self.make_cache()
    
    @property
    def name(self):
        return self._name
    
    @property
    def num_classes(self):
        return len(self._classes)
    
    @property
    def classes(self):
        return self._classes
    
    @property
    def count(self):
        return self._curr_counter
    
    @property
    def data(self):
        self._curr_counter+=1
        return self.__getitem__(self._curr_counter)
    
    @property
    def frame_count(self):
        return self._frame_counter
    
    @property
    def record_table(self):
        return self._sample_list
    
    @property
    def image_shape(self):
        if not self.image: return None
        width, height = self.image.shape
        return height, width, 3
        
    def __len__(self):
        if not self._total_frames:
            self.count_frames()
        return self._total_frames
    
    def __getitem__(self, idx):
        self._curr_counter = idx
        # Get the next dataset if frame number is more than table count
        if self._frame_counter == -1 or not len(self._dataset_nums) or self._frame_counter >= self._dataset_nums[self._file_counter]-1: 
            self.current_file = self.__file_names.pop() # get one filename
            dataset = WaymoDataFileReader(self.current_file) # get Dataloader
            self._sample_list = dataset.get_record_table() # get number of record table
            self._dataset_itr = iter(dataset) # Get next record iterator
            if self._frame_counter == -1:
                self._file_counter +=1
                self._dataset_nums.append(len(self._sample_list))
            self._frame_counter = 1
        else:
            self._frame_counter+=1
        self._num_frames+=1
        self._idx_to_frame.append((self._file_counter, self._frame_counter))
        return next(self._dataset_itr) # Send next frame from record 
    
    def count_frames(self):
        # Count total frames 
        for file_name in self.__file_names:
            print(file_name)
            dataset = WaymoDataFileReader(file_name)
            for frame in tqdm.tqdm(dataset):
                self._total_frames+=1
        print("[LOG] Total frames: ", self._total_frames)
        
    def get_file_names(self):
        self.__file_names = []
        for i in os.listdir(DATA_PATH):
            if i.split('.')[-1] == 'tfrecord':
                self.__file_names.append(DATA_PATH + '/' + i)
        print("[log] Number of files found {}".format(len(self.__file_names)))  
        
    def get_lidar(self, frame, idx, all_points=False):
        '''Get lidar pointcloud
            TODO: Get all 4 lidar points appeneded together
            :return pcl: (N, 3) points in (x,y,z)
        '''
        laser_name = dataset_pb2.LaserName.TOP # laser information
        laser = utils.get(frame.lasers, laser_name)
        laser_calibration = utils.get(frame.context.laser_calibrations, laser_name)
        ri, camera_projection, range_image_pose = utils.parse_range_image_and_camera_projection(laser)
        pcl, pcl_attr = utils.project_to_pointcloud(frame, ri, camera_projection, range_image_pose, laser_calibration)
        return pcl

    def get_image(self, frame, idx):
        '''Get image
        '''
        camera_name = dataset_pb2.CameraName.FRONT
        camera_calibration = utils.get(frame.context.camera_calibrations, camera_name)
        camera = utils.get(frame.images, camera_name)
        vehicle_to_image = utils.get_image_transform(camera_calibration) # Transformation
        img = utils.decode_image(camera)
        self.image=img
        return img

    def get_calib(self, frame, idx):
        '''Get calibration object
        '''
        return calibration.Calibration(frame, idx)

    def get_label(self, frame, idx):
        '''Get label as object3d
        {
            cls_type: Object class
            trucation: If truncated or not in image
            occlusion: If occluded or not in image
            box2d: 2d (x1, y1, x2, y2)
            h: box height
            w: box width
            l: box length
            pos: box center position in (x,y,z)
            ry: Heading theta about y axis
            score: Target score 
            alpha: 3D rotation azimuth angle
            level: hard/medium/easy
            dis_to_cam: range distance of point
        }
        '''
        # preprocess bounding box data
        id_to_bbox = dict()
        id_to_name = dict()
        for labels in frame.projected_lidar_labels:
            name = labels.name
            for label in labels.labels:
                bbox = [label.box.center_x - label.box.length / 2, label.box.center_y - label.box.width / 2,
                        label.box.center_x + label.box.length / 2, label.box.center_y + label.box.width / 2]
                id_to_bbox[label.id] = bbox
                id_to_name[label.id] = name - 1
        
        object_list = []
        for obj in frame.laser_labels:
            # caculate bounding box
            bounding_box = None
            name = None
            id = obj.id
            for lidar in self.__lidar_list:
                if id + lidar in id_to_bbox:
                    bounding_box = id_to_bbox.get(id + lidar)
                    name = str(id_to_name.get(id + lidar))
                    break
            if bounding_box == None or name == None:
                continue
            
            kitti_obj = object3d.Object3d()
            kitti_obj.cls_type = self.__type_list[obj.type]
            kitti_obj.trucation = 0
            kitti_obj.occlusion = 0
            kitti_obj.box2d = np.array(( float(bounding_box[0]), float(bounding_box[1]), float(bounding_box[2]), float(bounding_box[3])), dtype=np.float32)
            kitti_obj.h = obj.box.height
            kitti_obj.w = obj.box.width
            kitti_obj.l = obj.box.length
            x = obj.box.center_x
            y = obj.box.center_y
            z = obj.box.center_z
            kitti_obj.pos = np.array((float(x), float(y), float(z)), dtype=np.float32)
            kitti_obj.ry = obj.box.heading
            kitti_obj.score = 1
            beta = math.atan2(x, z)
            kitti_obj.alpha = (kitti_obj.ry + beta - math.pi / 2) % (2 * math.pi)
            kitti_obj.level = kitti_obj.get_obj_level()
            kitti_obj.dis_to_cam = np.linalg.norm(kitti_obj.pos)
            object_list.append(kitti_obj)
        return object_list
    
    def make_cache(self):
        return NotImplemented 
        
def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct

if __name__ == '__main__':
    import cv2
    DATA_PATH = '/home/monet/research/dataset/waymo/training'
    LOCATIONS = ['location_sf']

    dataset = WaymoDataset(DATA_PATH, LOCATIONS, 'train', True, "new_waymo")
    # 0-5 for training; 6-8 for val

    frame, idx = dataset.data, dataset.count
    calib = dataset.get_calib(frame, idx)
    pts =  dataset.get_lidar(frame, idx)
    target = dataset.get_label(frame, idx)
    image = dataset.get_image(frame, idx)
    
    print(image.shape, idx)
    cv2.imwrite('original.jpg', image)
    
    # groundtruth
    gtimg = image.copy()
    for obj in target:
        left,top,right,bottom = obj.box2d
        left,top,right,bottom = int(left),int(top),int(right),int(bottom)
        if bottom-top <=20: continue
        if obj.cls_type not in ['VEHICLE', 'PEDESTRIAN']: continue #[2,0]
        cv2.rectangle(gtimg,(left, top), (right, bottom),(0,255,0))
        cv2.putText(gtimg,obj.cls_type,(left,top-2),0,.6,(0,255,0))
    cv2.imwrite('groundtruth.jpg', gtimg)
    
    # inference with yolov5
    import torch
    from torchvision import transforms

    # Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom
    model.eval()
    
    # Get shape
    H,W,C = image.shape
    H0,W0 = 480,640
    shape = (H0,W0)
    
    # Resize 
    image = cv2.resize(image, (W0, H0), cv2.INTER_LINEAR)
    
    # Extract labels
    targets = []
    for obj in target:
        left,top,right,bottom = obj.box2d
        if bottom-top <=20: continue
        if obj.cls_type not in ['VEHICLE', 'PEDESTRIAN']: continue #[2,0]
        cls = 2 if obj.cls_type == 'VEHICLE' else 0
        x_center = (left + right)/W; y_center = (top + bottom)/H; width = (right - left)/W; height = (bottom - top)/H
        targets += [[0,cls,x_center,y_center,width,height]]
    targets = torch.FloatTensor(targets)

    # Inference
    ptimg = transforms.ToTensor()(image).unsqueeze(0)
    out, train_out = model(ptimg)

    # need to compute loss from results
    # normalized target label
    # input targets(image,class,x,y,w,h)
    # Loss func
    from utils.loss import ComputeLoss
    compute_loss = ComputeLoss(model.model)
    loss = compute_loss([x.float() for x in train_out], targets.to(out.device))[0]  # box, obj, cls
    
    # NMS
    from utils.general import non_max_suppression, scale_coords, xywh2xyxy
    targets[:, 2:] *= torch.Tensor([W, H, W, H]).to(targets.device)  # to pixels
    nb = 1
    conf_thres=0.001  # confidence threshold
    iou_thres=0.6  # NMS IoU threshold
    p, r, f1, mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    stats, ap, ap_class = [], [], []
    lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)]
    out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=False)
    
    # Plot
    from utils.plots import output_to_target, plot_images
    plot_images(ptimg, targets, '.', 'labels.jpg')
    plot_images(ptimg, output_to_target(out), '.', 'pred.jpg')
    
    # Metrics
    iouv = torch.linspace(0.5, 0.95, 10).cuda(0)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    seen = 0
    from pathlib import Path
    for si, pred in enumerate(out):
        labels = targets[targets[:, 0] == si, 1:]
        nl = len(labels)
        tcls = labels[:, 0].tolist() if nl else []  # target class
        seen += 1

        if len(pred) == 0:
            if nl:
                stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            continue

        # Predictions
        predn = pred.clone()
        scale_coords(image[si].shape[1:], predn[:, :4], shape)  # native-space pred

        # Evaluate
        if nl:
            tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
            scale_coords(image[si].shape[1:], tbox, shape)  # native-space labels
            labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
            correct = process_batch(predn, labelsn, iouv)
        else:
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
        stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)
        
    # Compute metrics
    nc = 2
    names = {0:'VEHICLE', 1:'PEDESTRIAN'}
    from utils.metrics import ap_per_class
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=True, save_dir=Path(''), names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))