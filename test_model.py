import argparse
import glob
import os
import random
import numpy as np
import torch
from tqdm import trange
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from src.datamodules.datasets.justraigs import JustRAIGSDataset
from src.bootstrap.bootstrap import bootstrap_metrics
from src.bootstrap.registry import DATASETS
from src.datamodules import JustRAIGSDataModule
from src.enums import Stage
from src.infra.metric_monitor import MetricMonitor
from src.utils import load_model_and_config
import albumentations as A
from torch.func import vmap
import copy
import cv2

def save_grad_cam_images(cam, input_tensor, targets, output_dir, original_images, image_index):
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    
    for i, grayscale in enumerate(grayscale_cam):
        rgb_img = original_images[i]
        grayscale = 1 - grayscale
        visualization = show_cam_on_image(rgb_img, grayscale, use_rgb=True)
        cam_image_path = os.path.join(output_dir, f"cam_image_{image_index + i}.jpg")
        cv2.imwrite(cam_image_path, visualization)

def test(model, datamodule, metrics_monitor: MetricMonitor, device, subset, ensemble=False, grad_cam=False, 
         layer_name=None, output_dir=None):
    datamodule.prepare_data()
    
    if subset == 'test':
        datamodule.setup(Stage.TEST)
        dataloader = datamodule.test_dataloader()
    elif subset == 'val':
        datamodule.setup(Stage.FIT)
        dataloader = datamodule.val_dataloader()
    elif subset == 'train':
        datamodule.setup(Stage.FIT)
        dataloader = datamodule.train_dataloader()
    metrics_monitor.val_metrics.reset()
    
    if not hasattr(model, args.layer_name):
        raise ValueError(f"Model does not have layer {args.layer_name}")
    
    layer = getattr(model, args.layer_name)
    
    cam = GradCAM(model=model, target_layers=[layer])

    with trange(len(dataloader)) as progress_bar:
        for batch_idx, sample_batch in zip(progress_bar, dataloader):
            X, y = sample_batch
            X = X.to(device)
            y = y.to(device)
            
            # Forward Pass
            if ensemble:
                ensemble_model, params, buffers = model
                output = ensemble_model(params, buffers, X)
                output = torch.mean(output, axis=0).squeeze()
            else:
                output = model(X).squeeze()
            
            metrics_monitor.to_val(device)
            metrics_monitor.update_val(output, y)

            if grad_cam:
                # Convert tensor images to numpy images
                original_images = [X[i].numpy(force=True).transpose(1, 2, 0) for i in range(X.shape[0])]
                original_images = [(img - img.min()) / (img.max() - img.min()) for img in original_images]
                targets = [ClassifierOutputTarget(0) for label in y]
                save_grad_cam_images(cam, X, targets, output_dir, original_images, 
                                     batch_idx * datamodule.batch_size)
                
    metric_compute = metrics_monitor.compute_val(plot_conf_matrix=False)
    
    for metric_name, tensor in metric_compute.items():
        
        if tensor.dim() == 0:  # Scalar tensor
            value = tensor.item()
        elif tensor.dim() > 1:  # tensor with multiple elements
            value = tensor.numpy(force=True)
        
        print(f"{metric_name.replace('val','test')}: {value}")

def call_single_model(params, buffers, data):
    return torch.func.functional_call(base_model, (params, buffers), (data,))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test PyTorch JIT model for referral or justification task")
    parser.add_argument('model_path', type=str, help="Model path")
    parser.add_argument('--data_dir', '-data', type=str, help="Dataset directory")
    parser.add_argument('--task', '-t', type=str, choices=['referral', 'justification'], default='referral',
                        help="Specify the task for testing the model (referral or justification)")
    parser.add_argument('--batch_size', '-bs', type=int, help="Test batch size)")
    parser.add_argument('--test_samples', '-samples', default=None, type=int, help="Test sample size")
    parser.add_argument('--threshold', '-thresh', default=None, type=float, help="Prediction threshold")
    parser.add_argument('--input_size', default=None, type=int, help="image input resize")
    parser.add_argument("--optimize", action="store_true", default=False, help="A boolean flag")
    parser.add_argument("--subset", default="test", type=str, choices=['test', 'val', 'train'], help="Dataset subset")
    parser.add_argument('--grad-cam', action='store_true', help='Generate and save Grad-CAM images')
    parser.add_argument('--layer_name', type=str, help='Layer name for Grad-CAM')
    parser.add_argument('--output_dir', type=str, default='./cam_images', help='Output directory for Grad-CAM images')

    # Parse the arguments
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.grad_cam and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    if args.grad_cam and args.layer_name is None:
        raise ValueError("Layer name must be provided for Grad-CAM")
    
    ensemble = False
    base_model = None
    if os.path.isdir(args.model_path):
        model_paths = glob.glob(os.path.join(args.model_path, '*.pt'))
        model = [torch.load(model_path)['model'].to(device) for model_path in model_paths]
        
        base_model = copy.deepcopy(model[0])
        base_model.to('meta')
        
        params, buffers = torch.func.stack_module_state(model)        
        model = (vmap(call_single_model, (0, 0, None)), params, buffers)
        ensemble = True
        
        config = torch.load(model_paths[0])['config']
    else:
        model, config = load_model_and_config(args.model_path, inference=True, optimize=args.optimize)
        model.to(device)
    
    datamodule_args = config['datamodule']['args']
    datamodule_args['batch_size'] = args.batch_size
    datamodule_args['data_dir'] = args.data_dir
    datamodule_args['test_samples'] = args.test_samples
    
    seed = config['seed']
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    transforms_config = config['transforms']
    transforms = [getattr(A, transform['name'])(**transform['args']) for transform in transforms_config 
                  if transform['name'] in ['Resize', 'Normalize', 'Equalize']]

    if datamodule_args.get('dataset') is not None:
        datamodule_args['dataset'] = DATASETS[datamodule_args['dataset']]
    else:
        datamodule_args['dataset'] = JustRAIGSDataset
        del datamodule_args['resample_negatives']
    
    datamodule = JustRAIGSDataModule(transforms=transforms, seed=config['seed'], 
                                     input_size=args.input_size,
                                     **datamodule_args)
    
    metrics_args = bootstrap_metrics(datamodule.task, datamodule.num_classes, 
                                     datamodule.num_labels, config['metrics'], threshold=args.threshold)
    metrics_monitor = MetricMonitor(task=datamodule.task, **metrics_args)
    
    test(model, datamodule, metrics_monitor, device, args.subset, ensemble=ensemble, 
         grad_cam=args.grad_cam, layer_name=args.layer_name, output_dir=args.output_dir)
