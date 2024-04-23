from yacs.config import CfgNode as CN

MODEL_PATH = dict()
MODEL_PATH = {
    "bninception": "~/.torch/models/bn_inception-52deb4733.pth",
    "resnet50": "~/.torch/models/pytorch_resnet50.pth",
    "googlenet": "~/.torch/models/googlenet-1378be20.pth",
}

MODEL_PATH = CN(MODEL_PATH)
