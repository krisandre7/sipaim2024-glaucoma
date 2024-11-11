from glob import glob
import os
from pathlib import Path
import torch
import yaml

PROJECT_DIR = Path(__file__).absolute().parent.parent
DATA_PATH = PROJECT_DIR / 'data' / 'JustRAIGS'

def load_model_and_config(path, inference = False, optimize = False):
    extra_files = {'config.yaml': ''}
    try:
        model = torch.jit.load(path, _extra_files=extra_files)
        if inference:
            if optimize:
                model = torch.jit.optimize_for_inference(model.eval())
            model.eval()
        config = yaml.safe_load(extra_files['config.yaml'])
    except RuntimeError:
        model_state = torch.load(path)
        config =  model_state['config']
        try:
            model = model_state['model'].model
        except:
            from src.bootstrap.registry import MODELS
            model = MODELS[model_state['config']['model']['name']]
            model = model(**model_state['config']['model']['args'])
            model.load_state_dict(model_state['model_state_dict'])
            model = model.model
        model.eval()
    return model, config

def clear_model_files(dir_name):
    files = os.listdir(dir_name)

    for file in files:
        if file.endswith(".pt"):
            os.remove(os.path.join(dir_name, file))