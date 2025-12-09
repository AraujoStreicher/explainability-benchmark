import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from torchcam.utils import overlay_mask
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import shutil
from torchcam.methods import GradCAM
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader


def prepare_val(data_dir):
    val_dir = os.path.join(data_dir, 'val')
    anno_file = os.path.join(val_dir, 'val_annotations.txt')
    images_dir = os.path.join(val_dir, 'images')

    mapping = {}
    with open(anno_file, 'r') as f:
        for line in f:
            filename, cls, *rest = line.strip().split('\t')
            mapping[filename] = cls

    for fname, cls in mapping.items():
        class_folder = os.path.join(val_dir, cls)
        class_images_folder = os.path.join(class_folder, 'images')
        os.makedirs(class_images_folder, exist_ok=True)

    for fname, cls in mapping.items():
        src = os.path.join(images_dir, fname)
        dest = os.path.join(val_dir, cls, 'images', fname)
        if os.path.exists(src):
            shutil.move(src, dest)





def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    tensor = tensor.cpu()
    
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return tensor





class RISE(nn.Module):
    def __init__(self, model, input_size, gpu_batch=100):
        super(RISE, self).__init__()
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch
    
    def generate_masks(self, N, s, p1, savepath='masks.npy'):
        
        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size
        
        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')
        
        self.masks = np.empty((N, *self.input_size))
        
        for i in tqdm(range(N), desc='Gerando máscaras'):
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])

            grid_i_tensor = torch.from_numpy(grid[i]).float().unsqueeze(0).unsqueeze(0)
            
            resized_mask = F.interpolate(grid_i_tensor, size=(int(up_size[0]), int(up_size[1])), mode='bilinear', align_corners=False)
      
            self.masks[i, :, :] = resized_mask.squeeze().numpy()[x:x + self.input_size[0], y:y + self.input_size[1]]
        
        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        np.save(savepath, self.masks)
        self.N = N
        self.p1 = p1
    
    def load_masks(self, filepath):
        self.masks = np.load(filepath)
        self.N = self.masks.shape[0]
    
    def forward(self, x):
        N = self.N
        B, C, H, W = x.size()
        
        p_list = []
        
        self.model = self.model.to(x.device)
        self.model.eval()
        
        for i in tqdm(range(0, N, self.gpu_batch), desc='Processando máscaras'):
            end_idx = min(i + self.gpu_batch, N)
            
            batch_masks = torch.from_numpy(self.masks[i:end_idx]).float().to(x.device)

            stack = torch.mul(batch_masks, x.data)

            with torch.no_grad():
                batch_p = self.model(stack)
                p_list.append(F.softmax(batch_p, dim=1).cpu())  
            
            del batch_masks, stack
            torch.cuda.empty_cache()
        
        p = torch.cat(p_list).to(x.device)
        CL = p.size(1)
        
        sal = torch.zeros((CL, H * W), device=x.device)

        for i in range(0, N, self.gpu_batch):
            end_idx = min(i + self.gpu_batch, N)
            batch_masks = torch.from_numpy(self.masks[i:end_idx]).float().to(x.device)
            batch_p = p[i:end_idx]
            

            sal += torch.matmul(batch_p.data.transpose(0, 1), 
                               batch_masks.view(end_idx - i, H * W))
            
            del batch_masks
            torch.cuda.empty_cache()
        
        sal = sal.view((CL, H, W))

        sal = sal / N 
        
        return sal
    

class ScoreCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.hook = self.target_layer.register_forward_hook(self.save_activation)

    def save_activation(self, module, input, output):
        self.activations = output

    def generate(self, input_tensor, target_class=None, batch_size=32):

        with torch.no_grad():
            output = self.model(input_tensor)
            
        if target_class is None:
            target_class = output.argmax(dim=1).item()


        activations = self.activations.detach()
        b, c, h, w = activations.size()

        input_size = input_tensor.size()[2:]
        upsampled_activations = F.interpolate(activations, size=input_size, mode='bilinear', align_corners=False)

        masks = upsampled_activations.squeeze(0)
        min_vals = masks.view(c, -1).min(dim=1)[0].view(c, 1, 1)
        max_vals = masks.view(c, -1).max(dim=1)[0].view(c, 1, 1)
        masks = (masks - min_vals) / (max_vals - min_vals + 1e-7)

        scores = []
        

        input_expanded = input_tensor.repeat(batch_size, 1, 1, 1)
        
        with torch.no_grad():
            for i in tqdm(range(0, c, batch_size), desc="Score-CAM"):
                end = min(i + batch_size, c)
                current_batch_size = end - i
                

                batch_masks = masks[i:end].unsqueeze(1)

                if current_batch_size < batch_size:
                    current_input = input_tensor.repeat(current_batch_size, 1, 1, 1)
                else:
                    current_input = input_expanded

                masked_input = current_input * batch_masks
                

                preds = self.model(masked_input)

                preds = F.softmax(preds, dim=1)
                scores.append(preds[:, target_class])


        scores = torch.cat(scores)

        weights = scores.view(1, c, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=True)

        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_size, mode='bilinear', align_corners=False)
        cam = cam.squeeze()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-7)

        return cam.cpu().numpy()
    
    def remove_hooks(self):
        self.hook.remove()




class IntegratedGradients:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate(self, input_tensor, target_class, steps=50, baseline=None):

        if baseline is None:
            baseline = torch.zeros_like(input_tensor).to(input_tensor.device)

        alphas = torch.linspace(0, 1, steps + 1, device=input_tensor.device).view(-1, 1, 1, 1)

        interpolated_images = baseline + alphas * (input_tensor - baseline)


        interpolated_images.requires_grad = True

        outputs = self.model(interpolated_images)

        score_target = outputs[:, target_class].sum()


        self.model.zero_grad()
        score_target.backward()


        gradients = interpolated_images.grad


        avg_gradients = (gradients[:-1] + gradients[1:]) / 2.0
        avg_gradients = avg_gradients.mean(dim=0) # [C, H, W]

        attributions = (input_tensor - baseline) * avg_gradients

        return attributions.detach()



def force_clear_hooks(model):
    for name, module in model.named_modules():
        if hasattr(module, '_backward_hooks'): module._backward_hooks.clear()
        if hasattr(module, '_forward_hooks'): module._forward_hooks.clear()
        if hasattr(module, '_forward_pre_hooks'): module._forward_pre_hooks.clear()

# ==========================================
# Definição dos Módulos CBAM
# ==========================================

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        
        return self.sigmoid(out)

class CBAMBottleneck(models.resnet.Bottleneck):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(CBAMBottleneck, self).__init__(inplanes, planes, stride, downsample, groups,
                                             base_width, dilation, norm_layer)
        self.ca = ChannelAttention(planes * self.expansion)
        self.sa = SpatialAttention()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ca(out) * out
        out = out.contiguous() 
        out = self.sa(out) * out

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

def get_resnet50_cbam(num_classes=200, pretrained=False):
    model = models.ResNet(CBAMBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    return model

def load_trained_model(model_path, device):
    print(f"Carregando arquitetura e pesos de: {model_path}")
    
    model = get_resnet50_cbam(num_classes=200, pretrained=False)
    

    state_dict = torch.load(model_path, map_location=device)
    

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v 
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    
    model.to(device)
    model.eval()
    
    print("Modelo carregado com sucesso!")
    return model

class CBAMAnalyzer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.last_attention_map = None
        self.hook = self.model.layer4[2].sa.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.last_attention_map = output.detach()

    def get_attention_map(self, img_tensor, upsample_size=(224, 224)):
        self.model.eval()
        with torch.no_grad():
            _ = self.model(img_tensor.unsqueeze(0))
        
        att_map = F.interpolate(self.last_attention_map, size=upsample_size, mode='bilinear', align_corners=False)
        att_map = att_map.squeeze().cpu().numpy()
        att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)
        return att_map