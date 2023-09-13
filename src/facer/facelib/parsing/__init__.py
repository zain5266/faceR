import os
import requests
import torch

from facelib.utils import load_file_from_url
from .bisenet import BiSeNet
from .parsenet import ParseNet


def init_parsing_model(model_name='bisenet', half=False, device='cuda'):
    if model_name == 'bisenet':
        model = BiSeNet(num_class=19)
        model_url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_bisenet.pth'
    elif model_name == 'parsenet':
        model = ParseNet(in_size=512, out_size=512, parsing_ch=19)
        model_url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')
    if not os.path.isfile("weights/parsing_parsenet.pth"):
        try:
            response=requests.get(model_url)
            if response.status_code==200:
                with open("weights/parsing_parsenet.pth",'wb') as file:
                    file.write(response.content)
                print("parsing_parsenet.pth downloaded successfully")
            else:
                print("fail to download parsing_parsenet.pth")
        except:
            print("Error occure while downloading parsing_parsenet.pth ")    
    model_path = "weights/parsing_parsenet.pth"
    load_net = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(load_net, strict=True)
    model.eval()
    model = model.to(device)
    return model
