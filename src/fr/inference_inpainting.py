import os
import cv2
import argparse
import glob
import requests
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import get_device
from basicsr.utils.registry import ARCH_REGISTRY


if __name__ == '__main__':
    if not os.path.isfile("weights/codeformer_inpainting.pth"):
        try:
            url="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer_inpainting.pth"
            response=requests.get(url)
            if response.status_code==200:
                with open("weights/codeformer_inpainting.pth",'wb') as file:
                    file.write(response.content)
                print("codeformer_inpainting.pth downloaded successfully")
            else:
                print("fail to download codeformer_inpainting.pth")
        except:
            print("Error occure while downloading codeformer_inpainting.pth ")
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = get_device()
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path', type=str, default='./inputs/masked_faces', 
                    help='Input image or folder. Default: inputs/masked_faces')
    parser.add_argument('-o', '--output_path', type=str, default=None, 
                    help='Output folder. Default: results/<input_name>')
    parser.add_argument('--suffix', type=str, default=None, 
                    help='Suffix of the restored faces. Default: None')
    args = parser.parse_args()

    # ------------------------ input & output ------------------------
    print('[NOTE] The input face images should be aligned and cropped to a resolution of 512x512.')
    if args.input_path.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')): # input single img path
        input_img_list = [args.input_path]
        result_root = f'results/test_inpainting_img'
    else: # input img folder
        if args.input_path.endswith('/'):  # solve when path ends with /
            args.input_path = args.input_path[:-1]
        # scan all the jpg and png images
        input_img_list = sorted(glob.glob(os.path.join(args.input_path, '*.[jpJP][pnPN]*[gG]')))
        result_root = f'results/{os.path.basename(args.input_path)}'

    if not args.output_path is None: # set output path
        result_root = args.output_path

    test_img_num = len(input_img_list)

    # ------------------ set up CodeFormer restorer -------------------
    net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=512, n_head=8, n_layers=9, 
                                            connect_list=['32', '64', '128']).to(device)
    
    # ckpt_path = 'weights/CodeFormer/codeformer.pth'
    ckpt_path = load_file_from_url(url="weights/codeformer_inpainting.pth", 
                                    model_dir='weights/CodeFormer', progress=True, file_name=None)
    checkpoint = torch.load(ckpt_path)['params_ema']
    net.load_state_dict(checkpoint)
    net.eval()

    # -------------------- start to processing ---------------------
    for i, img_path in enumerate(input_img_list):
        img_name = os.path.basename(img_path)
        basename, ext = os.path.splitext(img_name)
        print(f'[{i+1}/{test_img_num}] Processing: {img_name}')
        input_face = cv2.imread(img_path)
        assert input_face.shape[:2] == (512, 512), 'Input resolution must be 512x512 for inpainting.'
        # input_face = cv2.resize(input_face, (512, 512), interpolation=cv2.INTER_LINEAR)
        input_face = img2tensor(input_face / 255., bgr2rgb=True, float32=True)
        normalize(input_face, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        input_face = input_face.unsqueeze(0).to(device)
        try:
            with torch.no_grad():
                mask = torch.zeros(512, 512)
                m_ind = torch.sum(input_face[0], dim=0)
                mask[m_ind==3] = 1.0
                mask = mask.view(1, 1, 512, 512).to(device)
                # w is fixed to 1, adain=False for inpainting
                output_face = net(input_face, w=1, adain=False)[0]
                output_face = (1-mask)*input_face + mask*output_face
                save_face = tensor2img(output_face, rgb2bgr=True, min_max=(-1, 1))
            del output_face
            torch.cuda.empty_cache()
        except Exception as error:
            print(f'\tFailed inference for CodeFormer: {error}')
            save_face = tensor2img(input_face, rgb2bgr=True, min_max=(-1, 1))

        save_face = save_face.astype('uint8')

        # save face
        if args.suffix is not None:
            basename = f'{basename}_{args.suffix}'
        save_restore_path = os.path.join(result_root, f'{basename}.png')
        imwrite(save_face, save_restore_path)

    print(f'\nAll results are saved in {result_root}')

