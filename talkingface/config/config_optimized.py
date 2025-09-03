import argparse

class DataProcessingOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def parse_args(self):
        self.parser.add_argument('--video_path', type=str, default='./asserts/examples/test4.mp4',
                                 help='path of video')
        self.parser.add_argument('--openface_landmark_path', type=str, default='./asserts/examples/test4.csv',
                                 help='path of detected openface landmark')
        self.parser.add_argument('--audio_path', type=str, default='./asserts/examples/driving_audio_1.wav',
                                 help='path of driving audio')
        self.parser.add_argument('--save_path', type=str, default='./asserts/examples/output.mp4',
                                 help='path of output')
        self.parser.add_argument('--pretrained_syncnet_path', type=str, default='./asserts/syncnet_256mouth.pth',
                                 help='path of pretrained syncnet')
        self.parser.add_argument('--pretrained_frame_DINet_path', type=str, default='./asserts/frame_trained_DINet.pth',
                                 help='path of pretrained frame trained DINet')
        self.parser.add_argument('--pretrained_clip_DINet_path', type=str, default='./asserts/clip_training_DINet_256mouth.pth',
                                 help='path of pretrained clip trained DINet')
        self.parser.add_argument('--deepspeech_model_path', type=str, default='./asserts/output_graph.pb',
                                 help='path of pretrained deepspeech model')
        return self.parser.parse_args()

class DINetTrainingOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def parse_args(self):
        self.parser.add_argument('--seed', type=int, default=456, help='random seed to use.')
        self.parser.add_argument('--source_channel', type=int, default=3, help='input source image channels')
        self.parser.add_argument('--ref_channel', type=int, default=15, help='input reference image channels')
        self.parser.add_argument('--audio_channel', type=int, default=29, help='input audio channels')
        self.parser.add_argument('--augment_num', type=int, default=32, help='augment training data')
        # 优化：将mouth_region_size从64提升到128，提高唇形清晰度
        self.parser.add_argument('--mouth_region_size', type=int, default=128, help='mouth region size for training - increased for better lip clarity')
        self.parser.add_argument('--train_data', type=str, default=r"./asserts/training_data/training_json.json",
                            help='path of training json')
        # 优化：减小batch_size以适应更高分辨率训练
        self.parser.add_argument('--batch_size', type=int, default=16, help='training batch size - reduced for higher resolution')
        # 优化：增加感知损失权重，提高视觉质量
        self.parser.add_argument('--lamb_perception', type=int, default=15, help='weight of perception loss - increased for better visual quality')
        self.parser.add_argument('--lamb_syncnet_perception', type=int, default=0.1, help='weight of syncnet perception loss')
        # 优化：增加像素损失权重，提高细节保真度
        self.parser.add_argument('--lamb_pixel', type=int, default=15, help='weight of pixel loss - increased for better detail preservation')
        # 优化：降低学习率以适应更高分辨率训练
        self.parser.add_argument('--lr_g', type=float, default=0.00006, help='initial learning rate for generator - reduced for stability')
        self.parser.add_argument('--lr_d', type=float, default=0.00006, help='initial learning rate for discriminator - reduced for stability')
        self.parser.add_argument('--start_epoch', default=1, type=int, help='start epoch in training stage')
        self.parser.add_argument('--non_decay', default=4, type=int, help='num of epoches with fixed learning rate')
        self.parser.add_argument('--decay', default=36, type=int, help='num of linearly decay epochs')
        self.parser.add_argument('--checkpoint', type=int, default=2, help='num of checkpoints in training stage')
        # 优化：更新结果路径以反映128分辨率训练
        self.parser.add_argument('--result_path', type=str, default=r"./asserts/training_model_weight/frame_training_128",
                                 help='result path to save model - updated for 128 resolution')
        self.parser.add_argument('--coarse2fine', action='store_true', help='If true, load pretrained model path.')
        self.parser.add_argument('--coarse_model_path',
                                 default='',
                                 type=str,
                                 help='Save data (.pth) of previous training')
        self.parser.add_argument('--pretrained_syncnet_path',
                                 default='',
                                 type=str,
                                 help='Save data (.pth) of pretrained syncnet')
        self.parser.add_argument('--pretrained_frame_DINet_path',
                                 default='',
                                 type=str,
                                 help='Save data (.pth) of frame trained DINet')
        # =========================  Discriminator ==========================
        self.parser.add_argument('--D_num_blocks', type=int, default=4, help='num of down blocks in discriminator')
        self.parser.add_argument('--D_block_expansion', type=int, default=64, help='block expansion in discriminator')
        self.parser.add_argument('--D_max_features', type=int, default=256, help='max channels in discriminator')
        return self.parser.parse_args()


class DINetInferenceOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def parse_args(self):
        self.parser.add_argument('--source_channel', type=int, default=3, help='channels of source image')
        self.parser.add_argument('--ref_channel', type=int, default=15, help='channels of reference image')
        self.parser.add_argument('--audio_channel', type=int, default=29, help='channels of audio feature')
        # 保持推理时的高分辨率
        self.parser.add_argument('--mouth_region_size', type=int, default=256, help='help to resize window')
        self.parser.add_argument('--source_video_path',
                                 default='./asserts/examples/test4.mp4',
                                 type=str,
                                 help='path of source video')
        self.parser.add_argument('--source_openface_landmark_path',
                                 default='./asserts/examples/test4.csv',
                                 type=str,
                                 help='path of detected openface landmark')
        self.parser.add_argument('--driving_audio_path',
                                 default='./asserts/examples/driving_audio_1.wav',
                                 type=str,
                                 help='path of driving audio')
        self.parser.add_argument('--pretrained_clip_DINet_path',
                                 default='./asserts/clip_training_DINet_256mouth.pth',
                                 type=str,
                                 help='path of pretrained clip trained DINet')
        self.parser.add_argument('--deepspeech_model_path',
                                 default='./asserts/output_graph.pb',
                                 type=str,
                                 help='path of pretrained deepspeech model')
        return self.parser.parse_args()