"""
Validation script for CUB-200-2011 explainability methods.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from skimage.measure import label

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

    broke = None
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
            if pred_class != correct_class and broke is None:
                broke = step

    return confidences, predictions, broke

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


        if isinstance(base_flat, torch.Tensor):
            base_flat = base_flat.detach().cpu().numpy()
        else:
            base_flat = np.array(base_flat)

        if isinstance(noisy_flat, torch.Tensor):
            noisy_flat = noisy_flat.detach().cpu().numpy()
        else:
            noisy_flat = np.array(noisy_flat)
        r, _ = pearsonr(base_flat, noisy_flat)

        correlations.append(r)

    # Média das correlações
    return np.mean(correlations)



## SPARSITY AND SPATIAL COHERENCE
def calculate_gini_index(heatmap):
    h_flat = np.abs(heatmap.flatten())
    h_sorted = np.sort(h_flat)
    n = len(h_sorted)
    if n == 0 or np.sum(h_sorted) == 0: return 0.0
    cum_sum = np.cumsum(h_sorted)
    B = np.sum(cum_sum) / cum_sum[-1]
    gini = (n + 1 - 2 * B) / n
    return gini

def calculate_spatial_coherence(heatmap, threshold_percent=0.5):
    if np.max(heatmap) - np.min(heatmap) < 1e-7: return 1.0
    h_norm = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-7)
    threshold = threshold_percent * np.max(h_norm)
    binary_map = h_norm > threshold
    labeled_map, num_labels = label(binary_map, connectivity=2, background=0, return_num=True)
    if num_labels == 0: return 0.0
    total_energy = np.sum(h_norm[binary_map])
    if total_energy == 0: return 0.0
    max_cluster_energy = 0
    for i in range(1, num_labels + 1):
        cluster_mask = labeled_map == i
        cluster_energy = np.sum(h_norm[cluster_mask])
        if cluster_energy > max_cluster_energy:
            max_cluster_energy = cluster_energy
    return max_cluster_energy / total_energy


## INTRA-CLASS CONSISTENCY

def intraclass_consistency(lista_heatmaps):
    heatmaps_flat = np.array([h.flatten() for h in lista_heatmaps])
    matriz_corr = np.corrcoef(heatmaps_flat)
    indices_tri_sup = np.triu_indices_from(matriz_corr, k=1)
    correlacoes = matriz_corr[indices_tri_sup]
    score_consistencia = np.mean(correlacoes)
    return score_consistencia
