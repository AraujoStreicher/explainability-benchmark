"""
Validation script for CUB-200-2011 explainability methods.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

def get_heatmap(model, model_name, explain_fn, img_tensor, target=None):
    if model_name == 'GradCAM' or model_name == 'RISE':
        base_heatmap = explain_fn(img_tensor)
    elif model_name == 'ScoreCAM':
        base_heatmap = explain_fn(img_tensor, target) 
    elif model_name == 'IG':
        output = model(img_tensor)
        pred_prob = F.softmax(output, dim=1)
        pred_class_idx = torch.argmax(pred_prob, dim=1).item()
        baseline = torch.zeros_like(img_tensor).to(img_tensor.device)
        base_heatmap = explain_fn(img_tensor,
                                   baselines=baseline,
                                   target=pred_class_idx,
                                   n_steps=50,
                                   internal_batch_size=10,
                                   return_convergence_delta=False)
    
    return base_heatmap

## FIDELITY
def deletion_test(model, img_tensor, heatmap, correct_class, max_steps=20, pixels_per_step = 5, deletion_value=0.0):
    model.eval()

    img = img_tensor.clone()
    C, H, W = img.shape[1], img.shape[2], img.shape[3]

    heat = heatmap.astype(np.float32)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8) # Norm

    flat = heat.flatten()
    num_pixels = len(flat)

    sorted_idx = np.argsort(-flat)  # Descending order

    confidences = []
    predictions = []

    for step in range(1, max_steps + 1):
        start = (step-1) * pixels_per_step
        end = min(step * pixels_per_step, num_pixels)
        idx_to_delete = sorted_idx[start:end]

        if len(idx_to_delete) == 0:
            break

        for idx in idx_to_delete:
            h = idx // W
            w = idx % W
            img[0, :, h, w] = deletion_value


        with torch.no_grad():
            out = model(img)
            pred_class = torch.argmax(out, dim=1).item()
            prob = F.softmax(out, dim=1)[0, correct_class].item()
            confidences.append(prob)
            predictions.append(pred_class)

    return confidences, predictions

## ROBUSTNESS
def robustness_test(model, model_name, img_tensor, explain_fn, target=None, sigma=0.1, n_perturb=5):
    model.eval()

    base_heatmap = get_heatmap(model, model_name, explain_fn, img_tensor, target)
    
    base_flat = base_heatmap.flatten()

    correlations = []

    for _ in range(n_perturb):

        # Adiciona ruído gaussiano à imagem
        noise = torch.randn_like(img_tensor) * sigma
        noisy_img = img_tensor + noise

        # Gera o mapa desta imagem ruidosa
        noisy_heatmap = get_heatmap(model, model_name, explain_fn, noisy_img, target)

        # Flatten
        noisy_flat = noisy_heatmap.flatten()

        # Correlação de Pearson
        base_flat = base_flat.to('cpu')
        noisy_flat = noisy_flat.to('cpu')
        r, _ = pearsonr(base_flat, noisy_flat)

        correlations.append(r)

    # Média das correlações
    return np.mean(correlations)
    

