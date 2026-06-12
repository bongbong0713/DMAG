import argparse

import time

from copy import deepcopy

from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import math

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as vutils
import operator
from typing import Dict, Deque, List, Tuple
import math
import sys
from datetime import datetime

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

from clip.custom_clip import get_coop
from clip.cocoop import get_cocoop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed, accuracy1
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask

def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    return x.float().cpu().numpy()

class TSNECollector:
    def __init__(self, max_points=2000):
        self.max_points = max_points
        self.img_embeds = []
        self.img_logits = []
        self.count = 0

    @torch.no_grad()
    def add(self, emb: torch.Tensor, logits: torch.Tensor):
        """
        emb:   [B,D] or [D]
        logits:[B,C] or [C]
        """
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        remain = self.max_points - self.count
        if remain <= 0:
            return

        take = min(remain, emb.size(0))
        self.img_embeds.append(emb[:take].detach().cpu())
        self.img_logits.append(logits[:take].detach().cpu())
        self.count += take

    def dump(self):
        if self.count == 0:
            return None, None
        return torch.cat(self.img_embeds, dim=0), torch.cat(self.img_logits, dim=0)

def visualize_tsne_embeddings(
    image_embedding,        # [N, D]
    output_logits,          # [N, C]
    proto_bank_matrix,      # [C, D]
    classnames=None,
    outdir="./tsne_vis",
    topk_classes=10,        # ✅ 8~12 추천
    per_class_pool=150,     # 각 클래스에서 kNN 찾기 위한 후보 풀(너무 크면 느림)
    knn_show=30,            # ✅ 각 클래스에서 proto 주변 점 몇 개만 보여줄지
    random_state=42,
    annotate_proto=True,
):
    import os, numpy as np
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    os.makedirs(outdir, exist_ok=True)

    with torch.no_grad():
        pred = output_logits.argmax(dim=1)  # [N]

    device = image_embedding.device

    # --- 1) 자주 등장하는 클래스 top-K 선택 ---
    uniq, cnt = torch.unique(pred.cpu(), return_counts=True)
    topk = cnt.argsort(descending=True)[:topk_classes]
    keep_classes = sorted(uniq[topk].tolist())
    K = len(keep_classes)

    class_id_map = {old_c: new_i for new_i, old_c in enumerate(keep_classes)}

    # --- 2) 클래스별로 후보(pool)만 뽑고, 그 중 proto와 가까운 knn_show만 선택 ---
    kept_embeds = []
    kept_labels = []

    img_n_all = F.normalize(image_embedding.float(), dim=1)
    pb = proto_bank_matrix[keep_classes]  # [K,D]
    pb_n = F.normalize(pb.float(), dim=1)

    for old_c in keep_classes:
        idx_c = torch.nonzero(pred == old_c).squeeze(1)
        if idx_c.numel() == 0:
            continue

        # 후보 풀을 먼저 줄임 (per_class_pool)
        take_pool = min(per_class_pool, idx_c.numel())
        pool_idx = idx_c[torch.randperm(idx_c.numel(), device=device)[:take_pool]]

        # proto와 cosine sim 계산해서 top-k만 남김
        c_new = class_id_map[old_c]
        sim = (img_n_all[pool_idx] @ pb_n[c_new].unsqueeze(1)).squeeze(1)  # [pool]
        k = min(knn_show, sim.numel())
        top_local = sim.topk(k).indices
        sel_idx = pool_idx[top_local]

        kept_embeds.append(image_embedding[sel_idx])
        kept_labels.append(torch.full((k,), c_new, device=device, dtype=torch.long))

    if len(kept_embeds) == 0:
        print("[t-SNE] No samples to visualize.")
        return

    img_emb = torch.cat(kept_embeds, dim=0)  # [M,D]  (M ~ K*knn_show)
    lab     = torch.cat(kept_labels, dim=0)  # [M]

    # --- 3) t-SNE 입력: (선택된 점들) + (proto들) ---
    def _to_numpy(x):
        if hasattr(x, "detach"):
            x = x.detach()
        return x.float().cpu().numpy()

    X_img = _to_numpy(img_emb)       # [M,D]
    X_pb  = _to_numpy(pb)            # [K,D]
    X_all = np.concatenate([X_img, X_pb], axis=0)

    M = X_img.shape[0]
    perplexity = min(30, max(5, M // 3))  # 점 수 적으니 perplexity도 낮추기
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=random_state,
        n_iter=1000,
        verbose=1,
    )
    X2 = tsne.fit_transform(X_all)
    pts_img = X2[:M]
    pts_pb  = X2[M:]

    lab_np = lab.detach().cpu().numpy()

    # --- 4) plot: 점을 크고 진하게 + 테두리 ---
    plt.figure(figsize=(10, 9), dpi=180)

    cmap = plt.get_cmap("tab10" if K <= 10 else "tab20")
    colors = [cmap(i % cmap.N) for i in range(K)]

    # 이미지 점 (크게, 진하게, 검정 테두리)
    for c in range(K):
        m = (lab_np == c)
        if m.sum() == 0:
            continue
        plt.scatter(
            pts_img[m, 0], pts_img[m, 1],
            s=90, alpha=0.95, color=colors[c],
            edgecolors='k', linewidths=0.6,
        )

    # 프로토타입 (더 크게)
    for c in range(K):
        plt.scatter(
            pts_pb[c, 0], pts_pb[c, 1],
            s=520, marker='X',
            color=colors[c],
            edgecolors='k', linewidths=2.2,
            zorder=10
        )
        if annotate_proto:
            name = classnames[keep_classes[c]] if classnames is not None else f"cls{keep_classes[c]}"
            plt.text(
                pts_pb[c, 0], pts_pb[c, 1],
                f" {c}:{name}",
                fontsize=9, weight="bold", zorder=11
            )

    plt.title(f"t-SNE: proto + its nearest {knn_show} images per class (bold points)", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(outdir, "tsne_proto_neighbors_bold.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[t-SNE] Saved: {out_path}")


def save_aug_images_and_info(images, entropies, probs, classnames, targets=None,
                             save_dir="./aug_output/R", prefix="sample", step=0,
                             selected_idx=None, original_selected_idx=None, top_aug_indices=None):
    os.makedirs(save_dir, exist_ok=True)

    # 1. 전체 이미지 저장
    grid = vutils.make_grid(images.cpu(), nrow=4, normalize=True, scale_each=True)
    image_save_path = os.path.join(save_dir, f"{prefix}_step{step}.png")
    vutils.save_image(grid, image_save_path)

    # ✅ ensure targets match view count
    if targets is not None:
        if len(targets) == 1 and probs.shape[0] > 1:
            targets = targets.repeat(probs.shape[0])

    # 2. 텍스트 정보 저장
    txt_save_path = os.path.join(save_dir, f"{prefix}_step{step}.txt")
    sorted_txt_save_path = os.path.join(save_dir, f"{prefix}_step{step}_sorted.txt")

    info_list = []
    for i in range(probs.shape[0]):
        pred_idx = probs[i].argmax().item()
        conf = probs[i].max().item()
        entropy_val = entropies[i].item()
        class_name = classnames[pred_idx] if classnames else str(pred_idx)
        if targets is not None:
            target_idx = targets[i].item() if torch.is_tensor(targets[i]) else targets[i]
            gt_name = classnames[target_idx] if classnames else str(target_idx)
        else:
            gt_name = "-"
        info_list.append((i, class_name, conf, entropy_val, gt_name))

    def write_info_file(path, sorted_list):
        with open(path, "w") as f:
            f.write("View\tPred\tConf\tEntropy\tGT\n")
            for i, class_name, conf, entropy_val, gt_name in sorted_list:
                f.write(f"{i}\t{class_name}\t{conf:.4f}\t{entropy_val:.4f}\t{gt_name}\n")

    write_info_file(txt_save_path, info_list)
    write_info_file(sorted_txt_save_path, sorted(info_list, key=lambda x: x[2], reverse=True))

    # 3. 선택된 인덱스 정보 저장 및 선택된 이미지 저장
    def save_selected_images(idx_tensor, name):
        if idx_tensor is not None and len(idx_tensor) > 0:
            selected_imgs = images[idx_tensor.cpu()]
            grid_sel = vutils.make_grid(selected_imgs, nrow=4, normalize=True, scale_each=True)
            sel_img_path = os.path.join(save_dir, f"{prefix}_step{step}_{name}_selected.png")
            vutils.save_image(grid_sel, sel_img_path)
            print(f"🖼️  Saved {name} selected images to: {sel_img_path}")

    if selected_idx is not None or original_selected_idx is not None:
        sel_txt_path = os.path.join(save_dir, f"{prefix}_step{step}_selected_indices.txt")
        with open(sel_txt_path, "w") as f:
            if original_selected_idx is not None:
                f.write("Original selected indices:\n")
                f.write(", ".join(str(i.item()) for i in original_selected_idx) + "\n")
                save_selected_images(original_selected_idx, "original")

            if selected_idx is not None:
                f.write("Adapted selected indices:\n")
                f.write(", ".join(str(i.item()) for i in selected_idx) + "\n")
                save_selected_images(selected_idx, "adapted")

            if top_aug_indices is not None:
                f.write("Image-based selected indices:\n")
                f.write(", ".join(str(i.item()) for i in top_aug_indices) + "\n")
                save_selected_images(top_aug_indices, "image-based")

    print(f"✅ Saved augmented image to: {image_save_path}")
    print(f"📝 Saved prediction info to: {txt_save_path}")
    print(f"🗃️  Saved sorted info to: {sorted_txt_save_path}")
    if selected_idx is not None or original_selected_idx is not None:
        print(f"🔖 Saved selected index info to: {sel_txt_path}")

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx

def select_confident_samples_cosine(logits, selection_cosine, selection_selfentro):
    cosine_distan = [torch.nn.CosineSimilarity(dim=0)(logits[0], logits[i]) for i in range(1, logits.shape[0])]
    cosine_distan = torch.stack(cosine_distan)
    idx_cosine = torch.argsort(cosine_distan, descending=True)[:int(cosine_distan.size()[0] * selection_cosine)]
    # idx
    for i in range(idx_cosine.shape[0]):
        idx_cosine[i] +=1
    logits_cos = logits[idx_cosine]
    logits = torch.cat((logits[0, :].unsqueeze(0), logits_cos), dim=0)

    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * selection_selfentro)]

    return logits[idx], [idx_cosine, idx], cosine_distan

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

def entropy_per_sample(outputs):
    probs = F.softmax(outputs, dim=1)              # [N, C]
    log_probs = F.log_softmax(outputs, dim=1)      # [N, C]
    entropy = -(probs * log_probs).sum(dim=1)      # [N]
    return entropy

def margin_per_sample(outputs: torch.Tensor) -> torch.Tensor:
    top2 = outputs.topk(2, 1, True, True).values  # [N, 2]
    return top2[:, 0] - top2[:, 1]


def joint_confidence(
    cos: torch.Tensor,
    ent: torch.Tensor,
    mar: torch.Tensor,
    num_classes: int,
    w_cos: float = 1.0,
    w_ent: float = 0.2,
    w_mar: float = 0,
) -> torch.Tensor:
    """Higher ⇒ more reliable/usable."""
    ent_norm = 1.0 - ent / math.log(num_classes)
    mar_norm = torch.tanh(mar)
    return w_cos * cos + w_mar * mar_norm + w_ent * ent_norm


global_tuning_step = 0

def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    with torch.no_grad():
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
            elif features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1] = item
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [item]

class PrototypeBank:
    """Per‑class cumulative‑average prototype vectors."""

    def __init__(self):
        # store[class] = (prototype tensor)
        self.store: Dict[int, torch.Tensor] = {}
        self.count: Dict[int, int] = {}

    @torch.no_grad()
    def update(self, cls: int, feat: torch.Tensor):
        feat = feat.squeeze(0).detach()
        if cls not in self.store:
            # first sample becomes the prototype
            self.store[cls] = feat.clone()
            self.count[cls] = 1
        else:
            n = self.count[cls]
            # cumulative mean: new_proto = (prev * n + feat) / (n + 1)
            self.store[cls].mul_(n / (n + 1)).add_(feat / (n + 1))
            self.count[cls] = n + 1

    def gather(self, classes: List[int]) -> torch.Tensor:
        protos = [self.store[c] for c in classes if c in self.store]
        return torch.stack(protos, dim=0) if protos else torch.empty(0)
    
    def to_matrix(self, num_classes: int, feat_dim: int, device) -> torch.Tensor:
        """Return [C,D] matrix where missing prototypes are zeros."""
        mat = torch.zeros(num_classes, feat_dim, device=device)
        for c, proto in self.store.items():
            if c < num_classes:
                mat[c] = F.normalize(proto, dim=0)
        return mat

def _l2norm(x, dim=-1, eps=1e-12):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)

def l2norm(x, dim=-1, eps=1e-12):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def build_alignment_weighted_anchors(
    image_features,      # [B,D]
    all_desc_embeds,     # [C,P,D]
    tau_a=0.07,
    mask=None,           # [C,P] bool
    use_delta=True,
    eps=1e-12,
):
    import torch, torch.nn.functional as F
    I = F.normalize(image_features.float(), dim=-1)    # [B,D]
    E = F.normalize(all_desc_embeds.float(), dim=-1)   # [C,P,D]
    B, D = I.shape
    C, P, _ = E.shape
    device = I.device

    if mask is None:
        mask = torch.ones(C, P, dtype=torch.bool, device=device)
    den = mask.sum(1, keepdim=True).clamp_min(1).float()
    E_avg = F.normalize((E * mask.unsqueeze(-1).float()).sum(1) / den, dim=-1)  # [C,D]

    # cosine(i, e_{c,m})
    S = (I @ E.reshape(C*P, D).t()).reshape(B, C, P)  # [B,C,P]
    if use_delta:
        base = (I @ E_avg.t()).unsqueeze(2)           # [B,C,1]
        S = S - base                                   # Δ alignment

    # mask & softmax over prompts m
    mask_b = mask.unsqueeze(0).expand(B, -1, -1)
    S = S.masked_fill(~mask_b, float('-inf'))
    S32 = (S / tau_a).to(torch.float32)
    W = torch.softmax(S32, dim=2).to(S.dtype)             # [B,C,P]
    W = W * mask_b.float()
    W = W / W.sum(2, keepdim=True).clamp_min(eps)

    # batch-average weights -> class anchor
    b_mean = W.mean(0)                                 # [C,P]
    b_mean = b_mean * mask.float()
    b_mean = b_mean / b_mean.sum(1, keepdim=True).clamp_min(eps)
    t = F.normalize((b_mean.unsqueeze(-1) * E).sum(1), dim=-1)  # [C,D]
    return t.type(image_features.dtype)

def joint_confidence_diversity(
    cos: torch.Tensor,          # [N]
    ent: torch.Tensor,          # [N]
    embeds: torch.Tensor,       # [N,D] (raw, L2 norm 안돼도 됨; 아래서 정규화)
    num_classes: int,
    selected_embeds: torch.Tensor = None,  # [M,D] or None
    w_cos: float = 1.0,
    w_ent: float = 0.2,
    gamma: float = 0.0,         # γ=0이면 원본과 동일 스코어
    red_mode: str = "selected_max",  # or "knn_mean"
    knn_k: int = 5,
    eps: float = 1e-6,
) -> torch.Tensor:
    # 원본과 동일한 정의
    ent_norm = 1.0 - ent / math.log(num_classes)      # [0,1], 클수록 좋음
    score = w_cos * cos + w_ent * ent_norm            # γ=0이면 원본과 동일

    if gamma > 0:
        E = F.normalize(embeds, dim=1)
        if selected_embeds is not None and selected_embeds.numel() > 0 and red_mode == "selected_max":
            S = F.normalize(selected_embeds, dim=1)
            red = (E @ S.T).max(dim=1).values                      # [N], 클수록 중복↑
        else:
            sim = E @ E.T                                         # [N,N]
            k = min(knn_k, sim.size(1)-1)
            red = sim.topk(k+1, dim=1).values[:, 1:].mean(dim=1)  # 자기 자신 제외 kNN 평균
        
        red = red.float()                                  # <-- 핵심: quantile 전에 float32 보장
        if red.numel() < 2:
            red_n = torch.zeros_like(red)                  # 샘플 1개면 패널티 0
        else:
            ql, qh = torch.quantile(red, 0.05), torch.quantile(red, 0.95)
            denom = (qh - ql).clamp_min(eps)
            red_n = ((red.clamp(ql, qh) - ql) / denom)               # [0,1], 클수록 중복↑

        score = score - gamma * red_n

    return score
    
proto_bank = PrototypeBank()

global_pos_cache = {}

def test_time_tuning(model, inputs, optimizer, scaler, args, target=None, classnames=None):
    global global_tuning_step
    global global_pos_cache
    if args.cocoop:
        image_feature, pgen_ctx = inputs
        pgen_ctx.requires_grad = True
        optimizer = torch.optim.AdamW([pgen_ctx], args.lr)
    

    selected_idx = None
    for j in range(args.tta_steps):
        with torch.cuda.amp.autocast():
            if args.cocoop:
                output = model((image_feature, pgen_ctx))
            else:
                output, proto_output, image_embedding, text_embedding, prototype_bank, custom_indices, logit_scale, all_des, all_mask = model(inputs) 

            if selected_idx is not None:
                output = output[selected_idx]
            else:
                _ , original_selected_idx = select_confident_samples(output, args.selection_p)
                _, selected_idx = select_confident_samples(proto_output, args.selection_p)
                # _, cosine_selected_idx, _ = select_confident_samples_cosine(output, 0.8, 0.3)
                # 1. 전체 예측 결과
                # pred = output.argmax(dim=1)  # [B]

                # 2. 정답과 일치하는 인덱스만 선택
                # correct_mask = pred.eq(target)  # [B] - True인 곳만 정답 맞춤
                # correct_idx = correct_mask.nonzero(as_tuple=False).squeeze(1)  # 정답 맞춘 인덱스만 추출
                # 3. 해당 인덱스로 output과 proto_output 선택
                # output = output[correct_idx]
                original_output = output
                original_proto_output = proto_output
                output = output[selected_idx]
                proto_output = proto_output[selected_idx]

           

            selected_embeddings = image_embedding[selected_idx] 
            # pred = int(proto_output.mean(0).unsqueeze(0).topk(1, 1, True, True)[1].t())

            new_prototype_bank = build_alignment_weighted_anchors(
                image_features=image_embedding,             # [B, D] = image_embedding[selected_idx]
                all_desc_embeds=all_des,          # [C, P, D]
                tau_a=0.07,
                mask=all_mask
            )

            new_proto_output = logit_scale * image_embedding @ new_prototype_bank.t()
            _, selected_idx = select_confident_samples(new_proto_output, args.selection_p)
            original_new_proto_output = new_proto_output
            new_proto_output = new_proto_output[selected_idx]

            output = original_output[selected_idx]
            
            # for i in range(selected_embeddings.size(0)):
            #     single_feat = selected_embeddings[i].unsqueeze(0)  # shape: [1, D]
            #     single_loss = losses[i]               # shape: [1] or scalar\
            #     cur_pred = int(proto_output[i].argmax(dim=-1).item())
            #     update_cache(global_pos_cache, pred, [single_feat, single_loss], 3)
            # selected_entropy = entropy_per_sample(selected_embeddings)
            # ─── update prototype bank ───────────────────────────────────────
            for i in range(selected_embeddings.size(0)):
                cls_pred = int(output[i].argmax(dim=-1).item())
                proto_bank.update(cls_pred, selected_embeddings[i].unsqueeze(0))
            
            topk_cls = output.mean(0).topk(3).indices.tolist()
            proto_tensor = proto_bank.gather(topk_cls)

            if proto_tensor.numel() > 0:
                norm_proto = F.normalize(proto_tensor, dim=1)       # [K,D]
                norm_all   = F.normalize(image_embedding, dim=1)    # [N,D]
                cos_sim    = torch.matmul(norm_all, norm_proto.T).max(1).values  # [N]

                all_ent   = entropy_per_sample(original_proto_output)
                score_div = joint_confidence_diversity(
                    cos_sim, 
                    all_ent, 
                    image_embedding,
                    num_classes=original_output.size(1),
                    selected_embeds=image_embedding[selected_idx] if selected_idx.numel() > 0 else None,
                    w_cos=0.2, 
                    w_ent=1, 
                    gamma=0
                )
                thresh      = torch.quantile(score_div, 0.95).item()
                keep_idx    = torch.nonzero(score_div > thresh).squeeze(1)
                # 여기서 selected_idx 변경
                merged_idx  = torch.unique(torch.cat([selected_idx,
                                                       keep_idx.to(selected_idx.device)]))
                output      = original_output[merged_idx]
                proto_output = original_proto_output[merged_idx]
                new_proto_output = original_new_proto_output[merged_idx]
                image_embedding = image_embedding[merged_idx]
            else:
                image_embedding = image_embedding[selected_idx]


            probs = F.softmax(output, dim=-1)         # [N, C]
            avg_prob = probs.mean(dim=0)               # [C]

            optimizer = torch.optim.AdamW(list(model.prompt_learner.parameters()), args.lr)

            # ─── prototype‑bank logits --------------------------------------
            C        = output.size(1)
            feat_dim = image_embedding.size(1)
            pb_mat   = proto_bank.to_matrix(C, feat_dim, image_embedding.device)  # [C,D]
            norm_pb  = pb_mat  # already normalized inside
            norm_emb = F.normalize(image_embedding, dim=1)
            pb_logits = (logit_scale if torch.is_tensor(logit_scale) else 1.0) * (
                norm_emb @ norm_pb.T)  # [N,C]

            # confidence기반 weighted sum
            with torch.no_grad():
                # 소프트맥스를 통해 confidence 계산
                probs_output = F.softmax(output, dim=1)           # [B, C]
                # probs_proto = F.softmax(proto_output, dim=1)       # [B, C]
                probs_proto = F.softmax(new_proto_output, dim=1)
                probs_pb    = F.softmax(pb_logits, dim=1)

                # 각 sample에 대해 가장 높은 confidence 값 추출
                conf_output = probs_output.max(dim=1).values       # [B]
                conf_proto = probs_proto.max(dim=1).values         # [B]
                conf_pb    = probs_pb.max(dim=1).values
                # normalize (합이 1이 되도록)
                total = conf_output + conf_proto + conf_pb + 1e-6
                w_out, w_proto, w_pb = (conf_output/total).unsqueeze(1), (conf_proto/total).unsqueeze(1), (conf_pb/total).unsqueeze(1)
                # simple average
                # w_out, w_proto, w_pb = 1, 1, 1

                # adaptive weighted sum
                ens_logits = w_out * output + w_proto * new_proto_output + w_pb * pb_logits  # [N,C]
                # ens_logits = output
                avg_prob   = F.softmax(ens_logits, dim=1).mean(dim=0)
                # avg_prob = F.softmax(ens_logits.mean(dim=0), dim=0)
                T = 0.5 # temperature 이거도 조정하면서 해봐야할듯
                sharpened = avg_prob ** (1.0 / T)
                avg_prob = sharpened / sharpened.sum()  # normalize to 1

                # sharpen_each_view
                T = 0.01
                sharpened = F.softmax(F.softmax(ens_logits, dim=1) / T, dim=1)
                avg_prob = sharpened.mean(dim=0)

            # ens_logits = w_out.detach() * output + w_proto.detach() * proto_output + w_pb.detach() * pb_logits  # [B, C]
            # ens_logits = output + proto_output + pb_logits
            # logits = output.mean(dim=0)  # 하나의 view에서 예측 사용 (또는 avg_logits도 가능)
            logits = original_output[0]
            loss = F.kl_div(F.log_softmax(logits, dim=-1), avg_prob, reduction='batchmean') 
            # loss = t_similarity_loss
            # loss = avg_entropy(ens_logits)
            optimizer.zero_grad()
            # compute gradient and do SGD step
            scaler.scale(loss).backward()
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.step(optimizer)
            scaler.update()
        
        # optimizer.zero_grad()
        # # compute gradient and do SGD step
        # scaler.scale(loss).backward()
        # # Unscales the gradients of optimizer's assigned params in-place
        # scaler.step(optimizer)
        # scaler.update()
        # original_probs = F.softmax(original_output, dim=-1)         # [N, C]
        # original_log_probs = F.log_softmax(original_output, dim=-1) # [N, C]
        # original_entropies = -(original_probs * original_log_probs).sum(dim=-1)  # [N]

        # if j == 0 and args.tpt and global_tuning_step < 1000:
        #     # 이미지가 list 형태일 때 (TPT)
        #     if isinstance(inputs, tuple):
        #         images = torch.stack(inputs[0])
        #     elif isinstance(inputs, list):
        #         images = torch.stack(inputs)
        #     else:  # already a Tensor
        #         images = inputs
        #     save_aug_images_and_info(
        #         images=images,
        #         entropies=original_entropies,
        #         probs=original_probs,
        #         classnames=classnames,
        #         targets=target,
        #         save_dir=f"./aug_output/gpt/{args.test_sets}_all_aug",
        #         prefix="example",
        #         step=global_tuning_step,
        #         selected_idx=selected_idx,
        #         original_selected_idx=original_selected_idx,
        #         top_aug_indices = keep_idx
        #     )
    
    sel_correct = 0
    sel_total = 0
    
    with torch.no_grad():
        if target is not None and selected_idx is not None and selected_idx.numel() > 0:
            # "전체 aug 중 selected_idx"는 원래 전체 view 인덱스 기준이어야 함
            # 현재 코드에서는 output이 계속 original_output[selected_idx]로 덮일 수 있으니,
            # 'original_output'을 유지한 상태에서 pred를 뽑는 게 안전함.
            # (원본 logits 텐서가 없다면, 최소한 selected된 output으로라도 계산)
            try:
                pred_sel = original_output.argmax(dim=1)[selected_idx]  # [M]
            except NameError:
                pred_sel = output.argmax(dim=1)  # fallback (이미 selected된 output이면 [M])

            # target을 M에 맞게 확장
            if target.numel() == 1:
                tgt_sel = target.expand(pred_sel.size(0))
            else:
                # target이 view별로 있을 경우 selected_idx로 서브셋
                try:
                    tgt_sel = target.view(-1)[selected_idx]
                except:
                    tgt_sel = target.view(-1)[:pred_sel.size(0)]

            sel_correct = (pred_sel == tgt_sel).sum().item()
            sel_total = pred_sel.numel()

    # cocoop / coop 공통으로 stats 반환
    if args.cocoop:
        return pgen_ctx, sel_correct, sel_total
    return sel_correct, sel_total
        
    
    # global_tuning_step += 1
    # if args.cocoop:
    #     return pgen_ctx

    # return

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    log_file = open(log_path, "w")

    class Tee(object):
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    print(f"📄 Logging to: {log_path}")

def main():
    args = parser.parse_args()
    setup_logging(args.output_dir)
    set_random_seed(args.seed)

    # This codebase has only been tested under the single GPU setting
    assert args.gpu is not None
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu
    set_random_seed(args.seed)
    print("Use GPU: {} for training".format(args.gpu))

    # create model (zero-shot clip model (ViT-L/14@px336) with promptruning)
    if args.test_sets in fewshot_datasets:
        classnames = eval("{}_classes".format(args.test_sets.lower()))
    else:
        assert args.test_sets in ['A', 'R', 'K', 'V', 'I']
        classnames_all = imagenet_classes
        classnames = []
        if args.test_sets in ['A', 'R', 'V']:
            label_mask = eval("imagenet_{}_mask".format(args.test_sets.lower()))
            if args.test_sets == 'R':
                for i, m in enumerate(label_mask):
                    if m:
                        classnames.append(classnames_all[i])
            else:
                classnames = [classnames_all[i] for i in label_mask]
        else:
            classnames = classnames_all
    if args.cocoop:
        model = get_cocoop(args.arch, args.test_sets, 'cpu', args.n_ctx)
        assert args.load is not None
        load_model_weight(args.load, model, 'cpu', args) # to load to cuda: device="cuda:{}".format(args.gpu)
        model_state = deepcopy(model.state_dict())
    else:
        model = get_coop(args.arch, args.test_sets, args.gpu, args.n_ctx, args.ctx_init)
        if args.load is not None:
            print("Use pre-trained soft prompt (CoOp) as initialization")
            checkpoint = torch.load(args.load, map_location='cpu')
            pretrained_ctx = checkpoint['state_dict']['ctx']
            assert pretrained_ctx.size()[0] == args.n_ctx
            with torch.no_grad():
                model.prompt_learner.ctx.copy_(pretrained_ctx)
                model.prompt_learner.ctx_init_state = pretrained_ctx
        
        model_state = None

    for name, param in model.named_parameters():
        if not args.cocoop:
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        else:
            if "text_encoder" not in name:
                param.requires_grad_(False)
    
    print("=> Model created: visual backbone {}".format(args.arch))
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # define optimizer
    if args.cocoop:
        optimizer = None
        optim_state = None
    else:
        trainable_param = model.prompt_learner.parameters()
        # trainable_param = [
        #     param for name, param in model.named_parameters()
        #     if 'ln' in name
        # ]
        optimizer = torch.optim.AdamW(trainable_param, args.lr)
        optim_state = deepcopy(optimizer.state_dict())

    # setup automatic mixed-precision (Amp) loss scaling
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    print('=> Using native Torch AMP. Training in mixed precision.')

    cudnn.benchmark = True

    # norm stats from clip.load()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    
    # iterating through eval datasets
    datasets = args.test_sets.split("/")
    results = {}
    for set_id in datasets:
        if args.tpt:
            base_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution)])
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize])
            data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size-1, 
                                            augmix=len(set_id)>1)
            batchsize = 1
        else:
            data_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
                normalize,
            ])
            batchsize = args.batch_size

        print("evaluating: {}".format(set_id))
        # reset the model
        # Reset classnames of custom CLIP model
        if len(set_id) > 1: 
            # fine-grained classification datasets
            classnames = eval("{}_classes".format(set_id.lower()))
        else:
            assert set_id in ['A', 'R', 'K', 'V', 'I']
            classnames_all = imagenet_classes
            classnames = []
            if set_id in ['A', 'R', 'V']:
                label_mask = eval("imagenet_{}_mask".format(set_id.lower()))
                if set_id == 'R':
                    for i, m in enumerate(label_mask):
                        if m:
                            classnames.append(classnames_all[i])
                else:
                    classnames = [classnames_all[i] for i in label_mask]
            else:
                classnames = classnames_all
        if args.cocoop:
            model.prompt_generator.reset_classnames(classnames, args.arch)
            model = model.cpu()
            model_state = model.state_dict()
            model = model.cuda(args.gpu)
        else:
            model.reset_classnames(classnames, args.arch)

        val_dataset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode)
        print("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batchsize, shuffle=True,
                    num_workers=args.workers, pin_memory=True)
            
        results[set_id] = test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, classnames)
        del val_dataset, val_loader
        try:
            print("=> Acc. on testset [{}]: @1 {}/ @5 {}".format(set_id, results[set_id][0], results[set_id][1]))
        except:
            print("=> Acc. on testset [{}]: {}".format(set_id, results[set_id]))

    print("======== Result Summary ========")
    print("params: nstep	lr	bs")
    print("params: {}	{}	{}".format(args.tta_steps, args.lr, args.batch_size))
    print("\t\t [set_id] \t\t Top-1 acc. \t\t Top-5 acc.")
    for id in results.keys():
        print("{}".format(id), end="	")
    print("\n")
    for id in results.keys():
        print("{:.2f}".format(results[id][0]), end="	")
    print("\n")


def test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, classnames=None):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    total_sel_correct = 0
    total_sel_count = 0
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')
    tsne_collector = TSNECollector(max_points=2000)
    all_img_embeds = []
    all_logits = []
    max_collect = 2000
    collected = 0
    pb_mat_last = None
    # reset model and switch to evaluate mode
    model.eval()
    total_start_time = time.time()
    if args.reset:
        if not args.cocoop: # no need to reset cocoop because it's fixed
            with torch.no_grad():
                model.reset()
    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        assert args.gpu is not None
        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
            image = images[0]
        else:
            if len(images.size()) > 4:
                # when using ImageNet Sampler as the dataset
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(args.gpu, non_blocking=True)
            image = images
        target = target.cuda(args.gpu, non_blocking=True)
        if args.tpt:
            images = torch.cat(images, dim=0)

        # reset the tunable prompt to its initial state
        if not args.cocoop: # no need to reset cocoop because it's fixed
            if args.reset:
                if args.tta_steps > 0:
                    with torch.no_grad():
                        model.reset()
            optimizer.load_state_dict(optim_state)
            sel_correct, sel_total = test_time_tuning(model, images, optimizer, scaler, args, target, classnames)

            total_sel_correct += sel_correct
            total_sel_count   += sel_total
        else:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    image_feature, pgen_ctx = model.gen_ctx(images, args.tpt)
            optimizer = None
            pgen_ctx = test_time_tuning(model, (image_feature, pgen_ctx), optimizer, scaler, args)

        # The actual inference goes here
        if args.tpt:
            if args.cocoop:
                image_feature = image_feature[0].unsqueeze(0)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                if args.cocoop:
                    output = model((image_feature, pgen_ctx))
                else:
                    output, _, img_emb, _, _, _, _, _, _= model(image)
        
        if (not args.cocoop) and (img_emb is not None) and (collected < max_collect):
            take = min(img_emb.size(0), max_collect - collected)
            all_img_embeds.append(img_emb[:take].detach().cpu())
            all_logits.append(output[:take].detach().cpu())
            collected += take

            # ✅ 현재 proto_bank snapshot (마지막걸로 갱신)
            C = output.size(1)
            D = img_emb.size(1)
            pb_mat_last = proto_bank.to_matrix(C, D, device=output.device).detach().cpu()
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
                
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            progress.display(i)

    C = output.size(1)
    D = img_emb.size(1)
    pb_mat_last = proto_bank.to_matrix(C, D, device=output.device).detach().cpu()  # [C,D]

    progress.display_summary()
    if (not args.cocoop) and (len(all_img_embeds) > 0) and (pb_mat_last is not None):
        img_emb_all = torch.cat(all_img_embeds, dim=0)
        logits_all  = torch.cat(all_logits, dim=0)

        # new_prototype_bank가 따로 없으면 일단 동일한 pb로 넣어도 됨
        visualize_tsne_embeddings(
            image_embedding=img_emb_all.cuda(args.gpu, non_blocking=True),
            output_logits=logits_all.cuda(args.gpu, non_blocking=True),
            proto_bank_matrix=pb_mat_last.cuda(args.gpu, non_blocking=True),
            classnames=classnames,
            outdir=os.path.join(args.output_dir, "tsne_vis", f"set_{args.test_sets}"),
            topk_classes=10,
            per_class_pool=200,
            knn_show=25,
            annotate_proto=True
        )
    if total_sel_count > 0:
        pct = 100.0 * total_sel_correct / total_sel_count
        print(f"\n✅ Selected-aug accuracy (overall): {pct:.2f}%  "
              f"({total_sel_correct}/{total_sel_count})")
    else:
        print("\n✅ Selected-aug accuracy (overall): N/A (no selected views)")
    total_elapsed_time = time.time() - total_start_time  # ✅ 전체 실행 시간 계산
    print(f"\n✅ Total validation time: {total_elapsed_time:.2f} seconds")  # ✅ 실행 시간 출력

    return [top1.avg, top5.avg]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='A/R/V/K/I', help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-p', '--print-freq', default=200, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--tpt', action='store_true', default=False, help='run test-time prompt tuning')
    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default=None, type=str, help='init tunable prompts')
    parser.add_argument('--cocoop', action='store_true', default=False, help="use cocoop's output as prompt initialization")
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--reset', action='store_true', help='Enable resetting before evaluation')
    parser.add_argument('--no-reset', dest='reset', action='store_false', help='Disable resetting before evaluation')
    parser.add_argument('--output-dir', default='./logs', type=str,
                    help='directory to save log text files')
    parser.set_defaults(reset=True)

    main()