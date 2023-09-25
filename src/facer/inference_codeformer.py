import os
import time
import cv2
import argparse
import glob
import requests
import torch
import shutil
from torchvision.transforms.functional import normalize
from facer.basicsr.utils import imwrite, img2tensor, tensor2img
from facer.basicsr.utils.download_util import load_file_from_url
from facer.basicsr.utils.misc import gpu_is_available, get_device
from facer.facelib.utils.face_restoration_helper import FaceRestoreHelper
from facer.facelib.utils.misc import is_gray
from facer.basicsr.utils.registry import ARCH_REGISTRY
import math

pretrain_model_url = {
    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
}

def set_realesrgan():
    from facer.basicsr.archs.rrdbnet_arch import RRDBNet
    from facer.basicsr.utils.realesrgan_utils import RealESRGANer

    use_half = False
    if torch.cuda.is_available(): # set False in CPU/MPS mode
        no_half_gpu_list = ['1650', '1660'] # set False for GPUs that don't support f16
        if not True in [gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list]:
            use_half = True

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )
    upsampler = RealESRGANer(
        scale=2,
        model_path="weights/RealESRGAN_x2plus.pth",
        model=model,
        tile=args.bg_tile,
        tile_pad=40,
        pre_pad=0,
        half=use_half
    )

    if not gpu_is_available():  # CPU
        import warnings
        warnings.warn('Running on CPU now! Make sure your PyTorch version matches your CUDA.'
                        'The unoptimized RealESRGAN is slow on CPU. '
                        'If you want to disable it, please remove `--bg_upsampler` and `--face_upsample` in command.',
                        category=RuntimeWarning)
    return upsampler
def empty_folder(folder_path):
    try:
        # Check if the folder exists
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            # Iterate through all files within the folder
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"Folder '{folder_path}' has been emptied of files.")
        else:
            print(f"Folder '{folder_path}' does not exist or is not a directory.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def download_video(url,user_id):
    try:
        response=requests.get(url,stream=True)
        if response.status_code==200:
            save_dir="inputvideo"
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            file_name=f"upscaled_video_{user_id}.mp4"
            save_path=os.path.join(save_dir,file_name)
            print("downloading video")
            with open(save_path,'wb') as video_file:
                for chunk in response.iter_content(chunk_size=8192):
                    video_file.write(chunk)
            return save_path
        else:
            print(f"fail to downlaod video.Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"an error occured:{str(e)}")
        return None
    
class ImageRestorationConfig:
    def __init__(self):
        self.input_path = './inputs/whole_imgs'
        self.output_path = None
        self.fidelity_weight = 0.5
        self.upscale = 1
        self.has_aligned = False
        self.only_center_face = False
        self.draw_box = False
        self.detection_model = 'retinaface_resnet50'
        self.bg_upsampler = 'None'
        self.face_upsample = False
        self.bg_tile = 400
        self.suffix = None
        self.save_video_fps = None
        self.user_id=""

def main(link,name):
    print("in main")
    global args
    args = ImageRestorationConfig()
    args.user_id=name
    args.input_path=link

    if not os.path.isdir("weights"):
        os.makedirs("weights")
    if not os.path.isfile("weights/codeformer.pth"):
        try:
            response=requests.get(pretrain_model_url['restoration'])
            if response.status_code==200:
                with open("weights/codeformer.pth",'wb') as file:
                    file.write(response.content)
                print("codeformer.pth downloaded successfully")
            else:
                print("fail to download codeformer.pth")
        except:
            print("Error occure while downloading codeformer.pth ")
    if not os.path.isfile("weights/RealESRGAN_x2plus.pth"):
        try:
            url="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth"
            response=requests.get(url)
            if response.status_code==200:
                with open("weights/RealESRGAN_x2plus.pth",'wb') as file:
                    file.write(response.content)
                print("RealESRGAN_x2plus.pth downloaded successfully")
            else:
                print("fail to download RealESRGAN_x2plus.pth")
        except:
            print("Error occure while downloading RealESRGAN_x2plus.pth ")
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = get_device()
    # parser = argparse.ArgumentParser()

    # parser.add_argument('-i', '--input_path', type=str, default='./inputs/whole_imgs', 
    #         help='Input image, video or folder. Default: inputs/whole_imgs')
    # parser.add_argument('-o', '--output_path', type=str, default=None, 
    #         help='Output folder. Default: results/<input_name>_<w>')
    # parser.add_argument('-w', '--fidelity_weight', type=float, default=0.5, 
    #         help='Balance the quality and fidelity. Default: 0.5')
    # parser.add_argument('-s', '--upscale', type=int, default=2, 
    #         help='The final upsampling scale of the image. Default: 2')
    # parser.add_argument("-u_id","--user_id",type=str,default=None,help='please enter user id')
    # parser.add_argument('--has_aligned', action='store_true', help='Input are cropped and aligned faces. Default: False')
    # parser.add_argument('--only_center_face', action='store_true', help='Only restore the center face. Default: False')
    # parser.add_argument('--draw_box', action='store_true', help='Draw the bounding box for the detected faces. Default: False')
    # # large det_model: 'YOLOv5l', 'retinaface_resnet50'
    # # small det_model: 'YOLOv5n', 'retinaface_mobile0.25'
    # parser.add_argument('--detection_model', type=str, default='retinaface_resnet50', 
    #         help='Face detector. Optional: retinaface_resnet50, retinaface_mobile0.25, YOLOv5l, YOLOv5n, dlib. \
    #             Default: retinaface_resnet50')
    # parser.add_argument('--bg_upsampler', type=str, default='None', help='Background upsampler. Optional: realesrgan')
    # parser.add_argument('--face_upsample', action='store_true', help='Face upsampler after enhancement. Default: False')
    # parser.add_argument('--bg_tile', type=int, default=400, help='Tile size for background sampler. Default: 400')
    # parser.add_argument('--suffix', type=str, default=None, help='Suffix of the restored faces. Default: None')
    # parser.add_argument('--save_video_fps', type=float, default=None, help='Frame rate for saving video. Default: None')

    # args = parser.parse_args()

    # ------------------------ input & output ------------------------
    w = args.fidelity_weight
    user_id=args.user_id
    input_video = False
    # v=download_video(args.input_path,user_id)
    # args.input_path=v
    print(f"video path is {args.input_path}")
    if args.input_path.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')): # input single img path
        input_img_list = [args.input_path]
        result_root = f'results/test_img_{user_id}'
    elif args.input_path.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')): # input video path
        from facer.basicsr.utils.video_util import VideoReader, VideoWriter
        input_img_list = []
        vidreader = VideoReader(args.input_path)
        image = vidreader.get_frame()
        while image is not None:
            input_img_list.append(image)
            image = vidreader.get_frame()
        audio = vidreader.get_audio()
        fps = vidreader.get_fps() if args.save_video_fps is None else args.save_video_fps   
        
        result_root = f'upscaled_video'
        input_video = True
        vidreader.close()
    else: # input img folder
        if args.input_path.endswith('/'):  # solve when path ends with /
            args.input_path = args.input_path[:-1]
        # scan all the jpg and png images
        input_img_list = sorted(glob.glob(os.path.join(args.input_path, '*.[jpJP][pnPN]*[gG]')))
        result_root = f'results/{os.path.basename(args.input_path)}'

    if not args.output_path is None: # set output path
        result_root = f'{args.output_path}'

    test_img_num = len(input_img_list)
    if test_img_num == 0:
        raise FileNotFoundError('No input image/video is found...\n' 
            '\tNote that --input_path for video should end with .mp4|.mov|.avi')

    # ------------------ set up background upsampler ------------------
    if args.bg_upsampler == 'realesrgan':
        bg_upsampler = set_realesrgan()
    else:
        bg_upsampler = None

    # ------------------ set up face upsampler ------------------
    if args.face_upsample:
        if bg_upsampler is not None:
            face_upsampler = bg_upsampler
        else:
            face_upsampler = set_realesrgan()
    else:
        face_upsampler = None

    # ------------------ set up CodeFormer restorer -------------------
    net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
                                            connect_list=['32', '64', '128', '256']).to(device)
    
    # ckpt_path = 'weights/CodeFormer/codeformer.pth'
    # ckpt_path = load_file_from_url(url=pretrain_model_url['restoration'], 
    #                                 model_dir='weights/CodeFormer', progress=True, file_name=None)
    ckpt_path='weights/codeformer.pth'
    checkpoint = torch.load(ckpt_path)['params_ema']
    net.load_state_dict(checkpoint)
    net.eval()

    # ------------------ set up FaceRestoreHelper -------------------
    # large det_model: 'YOLOv5l', 'retinaface_resnet50'
    # small det_model: 'YOLOv5n', 'retinaface_mobile0.25'
    if not args.has_aligned: 
        print(f'Face detection model: {args.detection_model}')
    if bg_upsampler is not None: 
        print(f'Background upsampling: True, Face upsampling: {args.face_upsample}')
    else:
        print(f'Background upsampling: False, Face upsampling: {args.face_upsample}')

    face_helper = FaceRestoreHelper(
        args.upscale,
        face_size=512,
        crop_ratio=(1, 1),
        det_model = args.detection_model,
        save_ext='png',
        use_parse=True,
        device=device)

    # -------------------- start to processing ---------------------
    for i, img_path in enumerate(input_img_list):
        # clean all the intermediate results to process the next image
        face_helper.clean_all()
        
        if isinstance(img_path, str):
            img_name = os.path.basename(img_path)
            basename, ext = os.path.splitext(img_name)
            # print(f'[{i+1}/{test_img_num}] Processing: {img_name}')
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        else: # for video processing
            basename = str(i).zfill(6)
            img_name = f'{user_id}_{basename}' if input_video else basename
            # print(f'[{i+1}/{test_img_num}] Processing: {img_name}')
            img = img_path

        if args.has_aligned: 
            # the input faces are already cropped and aligned
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            face_helper.is_gray = is_gray(img, threshold=10)
            if face_helper.is_gray:
                print('Grayscale input: True')
            face_helper.cropped_faces = [img]
        else:
            face_helper.read_image(img)
            # get face landmarks for each face
            num_det_faces = face_helper.get_face_landmarks_5(
                only_center_face=args.only_center_face, resize=640, eye_dist_threshold=5)
            # print(f'\tdetect {num_det_faces} faces')
            # align and warp each face
            face_helper.align_warp_face()

        # face restoration for each cropped face
        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            # prepare data
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

            try:
                with torch.no_grad():
                    output = net(cropped_face_t, w=w, adain=True)[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except Exception as error:
                print(f'\tFailed inference for CodeFormer: {error}')
                restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

            restored_face = restored_face.astype('uint8')
            face_helper.add_restored_face(restored_face, cropped_face)

        # paste_back
        if not args.has_aligned:
            # upsample the background
            if bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = bg_upsampler.enhance(img, outscale=args.upscale)[0]
            else:
                bg_img = None
            face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            if args.face_upsample and face_upsampler is not None: 
                restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=args.draw_box, face_upsampler=face_upsampler)
            else:
                restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=args.draw_box)

        # save faces
        for idx, (cropped_face, restored_face) in enumerate(zip(face_helper.cropped_faces, face_helper.restored_faces)):
            # save cropped face
            if not args.has_aligned: 
                save_crop_path = os.path.join(f'cropped_faces', f'{basename}_{idx:02d}.png')
                imwrite(cropped_face, save_crop_path)
            # save restored face
            if args.has_aligned:
                save_face_name = f'{basename}.png'
            else:
                save_face_name = f'{basename}_{idx:02d}.png'
            if args.suffix is not None:
                save_face_name = f'{save_face_name[:-4]}_{args.suffix}.png'
            save_restore_path = os.path.join(f'restored_faces', save_face_name)
            imwrite(restored_face, save_restore_path)

        # save restored img
        if not args.has_aligned and restored_img is not None:
            if args.suffix is not None:
                basename = f'{basename}_{args.suffix}'
            save_restore_path = os.path.join(f'final_results', f'{basename}.png')
            imwrite(restored_img, save_restore_path)

    # save enhanced video
    if input_video:
        print('Video Saving...')
        # load images
        video_frames = []
        img_list = sorted(glob.glob(os.path.join(f'final_results', '*.[jp][pn]g')))
        for img_path in img_list:
            img = cv2.imread(img_path)
            video_frames.append(img)
        # write images to video
        height, width = video_frames[0].shape[:2]
        if args.suffix is not None:
            video_name = f'{user_id}_{args.suffix}.png'
        timestamp = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime())
        os.mkdir("temp_result")
        save_restore_path = f'temp_result/video_{user_id}_{timestamp}.mp4'
        vidwriter = VideoWriter(save_restore_path, height, width, fps, audio)
        print(len(video_frames))
        for f in video_frames:
            print("f.shape, f.dtype")
            print("f.shape=",f.shape)
            print("f.shape=",f.dtype)
            vidwriter.write_frame(f)
        vidwriter.close()

    print(f'\nAll results are saved in {result_root}')
    try:
        if os.path.isdir('cropped_faces'):
            empty_folder('cropped_faces')
        if os.path.isdir('final_results'):
            empty_folder('final_results')
        if os.path.isdir('restored_faces'):
            empty_folder('restored_faces')
        if os.path.isdir("inputvideo"):
            empty_folder("inputvideo")
    except OSError as e:
        print(f'Error:{e}')
    return save_restore_path
