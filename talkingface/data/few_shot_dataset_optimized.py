import numpy as np
import cv2
import tqdm
import copy
from talkingface.utils import *
import glob
import pickle
import torch
import torch.utils.data as data

# 优化：支持动态分辨率配置
def get_image(A_path, crop_coords, input_type, resize=128):  # 默认使用128分辨率
    """优化版本的图像处理函数，支持动态分辨率配置"""
    (x_min, y_min, x_max, y_max) = crop_coords
    size = (x_max - x_min, y_max - y_min)

    if input_type == 'mediapipe':
        if A_path.shape[1] == 2:
            pose_pts = (A_path - np.array([x_min, y_min])) * resize / size
            return pose_pts[:, :2]
        else:
            A_path[:, 2] = A_path[:, 2] - np.max(A_path[:, 2])
            pose_pts = (A_path - np.array([x_min, y_min, 0])) * resize / size[0]
            return pose_pts[:, :3]
    else:
        img_output = A_path[y_min:y_max, x_min:x_max, :]
        # 优化：使用双三次插值提高缩放质量
        img_output = cv2.resize(img_output, (resize, resize), interpolation=cv2.INTER_CUBIC)
        return img_output

def generate_input(img, keypoints, mask_keypoints, is_train=False, mode=["mouth_bias"], 
                  mouth_width=None, mouth_height=None, target_size=128):
    """优化版本的输入生成函数，支持动态分辨率"""
    # 根据关键点决定正方形裁剪区域
    crop_coords = crop_face(keypoints, size=img.shape[:2], is_train=is_train)
    target_keypoints = get_image(keypoints[:,:2], crop_coords, input_type='mediapipe', resize=target_size)
    target_img = get_image(img, crop_coords, input_type='img', resize=target_size)

    target_mask_keypoints = get_image(mask_keypoints[:,:2], crop_coords, input_type='mediapipe', resize=target_size)

    # source_img信息：扣出嘴部区域
    source_img = copy.deepcopy(target_img)
    source_keypoints = target_keypoints

    pts = source_keypoints.copy()
    face_edge_start_index = 2
    pts[INDEX_FACE_OVAL[face_edge_start_index:-face_edge_start_index], 1] = target_mask_keypoints[face_edge_start_index:-face_edge_start_index, 1]
    pts = pts[FACE_MASK_INDEX + INDEX_NOSE_EDGE[::-1], :2]
    pts = pts.reshape((-1, 1, 2)).astype(np.int32)
    cv2.fillPoly(source_img, [pts], color=(0, 0, 0))
    
    # 优化：根据目标分辨率调整嘴部特征图生成
    scale_factor = target_size / (crop_coords[2] - crop_coords[0])
    source_face_egde = draw_face_feature_maps(
        source_keypoints, 
        mode=mode, 
        im_edges=target_img,
        mouth_width=mouth_width * scale_factor if mouth_width else None, 
        mouth_height=mouth_height * scale_factor if mouth_height else None
    )
    source_img = np.concatenate([source_img, source_face_egde], axis=2)
    return source_img, target_img, crop_coords

def generate_ref(img, keypoints, is_train=False, alpha=None, beta=None, target_size=128):
    """优化版本的参考图像生成函数"""
    crop_coords = crop_face(keypoints, size=img.shape[:2], is_train=is_train, alpha=alpha, beta=beta)
    target_keypoints = get_image(keypoints[:,:2], crop_coords, input_type='mediapipe', resize=target_size)
    target_img = get_image(img, crop_coords, input_type='img', resize=target_size)
    return target_img, target_keypoints, crop_coords

def select_ref_index(driven_keypoints, n_ref=5, ratio=1/3.):
    """选择参考帧索引"""
    driven_keypoints = np.array(driven_keypoints)
    mouth_open_list = []
    for i in range(len(driven_keypoints)):
        mouth_open = np.mean(np.linalg.norm(driven_keypoints[i][INDEX_LIPS_INNER[0:6]] - driven_keypoints[i][INDEX_LIPS_INNER[6:12]], axis=1))
        mouth_open_list.append(mouth_open)
    mouth_open_list = np.array(mouth_open_list)
    ref_index_list = np.argsort(mouth_open_list)[::-1][:int(len(driven_keypoints) * ratio)]
    ref_index_list = np.random.choice(ref_index_list, n_ref, replace=True)
    return ref_index_list

def get_ref_images_fromVideo(cap, ref_img_index_list, ref_keypoints, target_size=128):
    """从视频中获取参考图像"""
    ref_img_list = []
    ref_keypoints_list = []
    ref_crop_coords_list = []
    for ref_img_index in ref_img_index_list:
        cap.set(cv2.CAP_PROP_POS_FRAMES, ref_img_index)
        ret, frame = cap.read()
        if ret:
            ref_img, ref_keypoint, ref_crop_coords = generate_ref(
                frame, ref_keypoints[ref_img_index], is_train=False, target_size=target_size
            )
            ref_img_list.append(ref_img)
            ref_keypoints_list.append(ref_keypoint)
            ref_crop_coords_list.append(ref_crop_coords)
    return ref_img_list, ref_keypoints_list, ref_crop_coords_list

class Few_Shot_Dataset_Optimized(data.Dataset):
    """优化版本的数据集类，支持动态分辨率配置"""
    def __init__(self, dict_info, n_ref=2, is_train=False, target_size=128):
        self.dict_info = dict_info
        self.n_ref = n_ref
        self.is_train = is_train
        self.target_size = target_size  # 新增：目标分辨率参数
        
        self.driven_images = dict_info["driven_images"]
        self.driven_keypoints = dict_info["driven_keypoints"]
        self.driving_keypoints = dict_info["driving_keypoints"]
        self.driven_mask_keypoints = dict_info["driven_mask_keypoints"]
        
        self.length = 0
        for video_index in range(len(self.driven_images)):
            self.length += len(self.driven_images[video_index])
        print(f"Dataset initialized with {self.length} samples at {self.target_size}x{self.target_size} resolution")

    def get_ref_images(self, video_index, ref_img_index_list):
        """获取参考图像"""
        ref_img_list = []
        ref_keypoints_list = []
        ref_crop_coords_list = []
        for ref_img_index in ref_img_index_list:
            ref_img, ref_keypoint, ref_crop_coords = generate_ref(
                self.driven_images[video_index][ref_img_index], 
                self.driven_keypoints[video_index][ref_img_index], 
                is_train=self.is_train,
                target_size=self.target_size  # 使用动态分辨率
            )
            ref_img_list.append(ref_img)
            ref_keypoints_list.append(ref_keypoint)
            ref_crop_coords_list.append(ref_crop_coords)
        return ref_img_list, ref_keypoints_list, ref_crop_coords_list

    def __getitem__(self, index):
        """获取数据项"""
        video_index = 0
        img_index = index
        for i in range(len(self.driven_images)):
            if img_index - len(self.driven_images[i]) < 0:
                video_index = i
                break
            else:
                img_index = img_index - len(self.driven_images[i])

        # 选择参考帧
        ref_img_index_list = select_ref_index(self.driven_keypoints[video_index], n_ref=self.n_ref)
        ref_img_list, ref_keypoints_list, ref_crop_coords_list = self.get_ref_images(video_index, ref_img_index_list)

        # 生成输入数据
        source_img, target_img, crop_coords = generate_input(
            self.driven_images[video_index][img_index], 
            self.driven_keypoints[video_index][img_index],
            self.driven_mask_keypoints[video_index][img_index], 
            is_train=self.is_train,
            target_size=self.target_size  # 使用动态分辨率
        )

        # 数据预处理
        source_img = source_img / 255.0
        target_img = target_img / 255.0
        ref_img_tensor = []
        for ref_img in ref_img_list:
            ref_img = ref_img / 255.0
            ref_img_tensor.append(ref_img)
        ref_img_tensor = np.concatenate(ref_img_tensor, axis=2)

        # 转换为张量格式
        source_img = torch.from_numpy(source_img).float().permute(2, 0, 1)
        target_img = torch.from_numpy(target_img).float().permute(2, 0, 1)
        ref_img_tensor = torch.from_numpy(ref_img_tensor).float().permute(2, 0, 1)

        return source_img, ref_img_tensor, target_img

    def __len__(self):
        return self.length

# 保持原有的data_preparation函数不变
def data_preparation(train_video_list):
    """数据准备函数"""
    img_all = []
    keypoints_all = []
    mask_all = []
    main_keypoints_index = list(range(0, 468))
    
    for model_name in tqdm.tqdm(train_video_list):
        Path_output_pkl = os.path.join(model_name, "keypoint_rotate.pkl")
        with open(Path_output_pkl, "rb") as f:
            images_info = pickle.load(f)
        keypoints_all.append(images_info[:, main_keypoints_index, :2])

        Path_output_pkl = os.path.join(model_name, "face_mat_mask.pkl")
        with open(Path_output_pkl, "rb") as f:
            mat_list, face_pts_mean_personal = pickle.load(f)

        face_pts_mean_personal = face_pts_mean_personal[INDEX_FACE_OVAL]
        face_mask_pts = np.zeros([len(mat_list), len(face_pts_mean_personal), 2])
        for index_ in range(len(mat_list)):
            rotationMatrix = mat_list[index_]
            keypoints = np.ones([4, len(face_pts_mean_personal)])
            keypoints[:3, :] = face_pts_mean_personal.T
            driving_mask = rotationMatrix.dot(keypoints).T
            face_mask_pts[index_] = driving_mask[:, :2]
        mask_all.append(face_mask_pts)

    print("train size: ", len(img_all))
    dict_info = {}
    dict_info["driven_images"] = img_all
    dict_info["driven_keypoints"] = keypoints_all
    dict_info["driving_keypoints"] = keypoints_all
    dict_info["driven_mask_keypoints"] = mask_all
    return dict_info