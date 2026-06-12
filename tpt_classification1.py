import argparse

import time

from copy import deepcopy

from PIL import Image
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F
import itertools


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

from clip.custom_clip import get_coop, get_coop1
from clip.cocoop import get_cocoop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed, accuracy1
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def select_confident_samples(logits, top):
    logits = logits.squeeze(0)
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx

def select_confident_samples_0(logits, top):
    logits = logits.squeeze(0)  # [N, num_classes]
    
    # 0번째는 고정
    original_logit = logits[0:1, :]  # [1, num_classes]
    
    # 나머지만 entropy 계산
    shift_logits = logits[1:, :]     # [N-1, num_classes]
    
    batch_entropy = -(shift_logits.softmax(1) * shift_logits.log_softmax(1)).sum(1)  # [N-1]
    idx_shift = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size(0) * top)]  # index for shifted logits
    
    # idx_shift는 shift_logits 기준이니까 +1 해줘야 원래 logits 기준에 맞음
    idx_shift = idx_shift + 1

    # 최종 idx는 0번(logits[0]) + top 확률 높은 것들
    idx = torch.cat([torch.zeros(1, dtype=torch.long, device=logits.device), idx_shift], dim=0)

    # 선택된 logits 리턴
    return logits[idx], idx

def avg_entropy(outputs):
    outputs = outputs.squeeze(0)
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

def avg_entropy_with_weight(outputs, entropy_threshold=None, eps=1e-3):
    outputs = outputs.squeeze(0)  # [N, C]
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)  # log-softmax
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])  # log(mean(exp(logits)))
    
    # 안전한 계산을 위한 클램핑
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    
    # 평균 예측 분포의 엔트로피
    entropy = -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)  # scalar

    # entropy_threshold를 기준으로 weight 계산
    if entropy_threshold is not None:
        weight = torch.clamp(1.0 - entropy / (entropy_threshold + eps), min=0.0)
        use_for_update = entropy <= entropy_threshold
        return entropy.item(), weight.item(), use_for_update
    else:
        return entropy.item()

def prototype_contrastive_loss(z_img, prototype_bank, pseudo_label, temperature=0.07):
    """
    z_img: [N, D] transformed embeddings for one image (N = num_transforms)
    prototype_bank: [num_classes, D]
    pseudo_label: int
    """
    if z_img.dim() == 1:
        z_img = z_img.unsqueeze(0)  # shape: [1, D]
    sim = torch.matmul(z_img, prototype_bank.T)  # [N, num_classes]
    logits = sim / temperature
    labels = torch.full((z_img.size(0),), pseudo_label, dtype=torch.long, device=z_img.device)
    
    return F.cross_entropy(logits, labels)

def consistency_loss(outputs):
    outputs = outputs.squeeze(0)

    o_0 = F.log_softmax(outputs[0], dim=-1)
    avg_output = F.softmax(outputs, dim=-1).mean(dim=0)

    loss = F.kl_div(o_0, avg_output, reduction='batchmean')

    return loss

# def test_time_generator_tuning(model, inputs, optimizer, scaler, args):
#     if args.cocoop:
#         image_feature, pgen_ctx = inputs
#         pgen_ctx.requires_grad = True
#         optimizer = torch.optim.AdamW([pgen_ctx], args.lr)
    
#     selected_idx = None
#     for j in range(args.tta_steps):
#         with torch.cuda.amp.autocast():
#             if args.cocoop:
#                 output = model((image_feature, pgen_ctx))
#             else:
#                 output, transformed_embeddings, image_embedding, text_embedding, prototype_bank, similarities = model(inputs) 
#                 # output, image_embedding, text_embedding, prototype_bank, similarities = model(inputs)
#                 model.prototype_bank.requires_grad = True
#                 # output을 affine parameter로 추가

#             if selected_idx is not None:
#                 output = output[selected_idx]
#             else:
#                 pseudo_logit = output[0]
#                 # output, selected_idx = select_confident_samples(output, args.selection_p)
#                 output, selected_idx = select_confident_samples_0(output, args.selection_p)
#                 # ouput = output    

#             original_output = output[0, :]
#             print(original_output.shape)
#             shift_outputs = output[1:, :]
#             print(shift_outputs.shape)
#             shift_outputs = shift_outputs.detach()
#             all_outputs = torch.cat([original_output, shift_outputs], dim=0)
#             # i_similarity_loss = F.mse_loss(transformed_embeddings.squeeze(0), image_embedding)
            
#             # softmax로 similarity 정규화 (soft target처럼 사용)
#             weights = F.softmax(similarities / 0.07, dim=0)  # [num_classes]

#             # weighted loss: 유사도가 높은 prototype일수록 더 많이 학습
#             proto_loss = torch.sum(weights * (1 - similarities))

#             sim = F.cosine_similarity(transformed_embeddings.mean(dim=1).unsqueeze(1), prototype_bank, dim=-1)  # [1, num_classes]

#             # Weighted similarity → weight가 큰 쪽은 유사도가 높아지도록 유도
#             generator_loss = (1 - sim) * weights.unsqueeze(0)  # [1, num_classes]
#             generator_loss = generator_loss.sum() 

#             # loss = avg_entropy(output) + args.proto_lambda * proto_loss + args.proto_contrastive_lambda * contrastive_proto_loss 
#             # loss = avg_entropy(output) + args.proto_contrastive_lambda * contrastive_proto_loss
#             # loss = args.proto_lambda * proto_loss
#             loss = generator_loss

#             #자기들끼리 멀어지게하는 로스 추가
#             # print(f"avg_entropy(output) shape: {avg_entropy(output).shape}")  # 기대값: torch.Size([])
#             # print(f"similarity_loss shape: {similarity_loss.shape}")  # 기대값: torch.Size([])
#             # print(f"loss shape: {loss.shape}")  # 기대값: torch.Size([])
#         model.prototype_bank.requires_grad = True
#         for param in model.shift_generator.parameters():
#             param.requires_grad = True
#         optimizer = torch.optim.AdamW(list(model.shift_generator.parameters()), args.lr)
#         optimizer.zero_grad()
#         # compute gradient and do SGD step
#         scaler.scale(loss).backward()
#         # Unscales the gradients of optimizer's assigned params in-place
#         scaler.step(optimizer)
#         scaler.update()
#     if args.cocoop:
#         return pgen_ctx
def confidence_reward(logits):
    """
    logits: Tensor of shape [num_classes], raw logits (e.g., cosine similarities * scale)
    Returns:
        R_conf: confidence reward score (scalar)
    """
    # softmax to get probability distribution
    probs = F.softmax(logits, dim=-1)  # [C]

    # entropy calculation
    entropy = -torch.sum(probs * torch.log(probs + 1e-8))  # scalar

    # normalize entropy
    num_classes = logits.size(-1)
    max_entropy = torch.log(torch.tensor(num_classes, dtype=probs.dtype, device=probs.device))

    r_conf = 1.0 - (entropy / max_entropy)

    return r_conf  # scalar

def test_time_tuning(model, inputs, optimizer, scaler, args):
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
                output, transformed_embeddings, image_embedding, text_embedding, prototype_bank, similarities = model(inputs) 
                model.prototype_bank.requires_grad = True
                for param in model.shift_generator.parameters():
                    param.requires_grad = True
                # output, image_embedding, text_embedding, prototype_bank = model(inputs) 
                # output을 affine parameter로 추가

            if selected_idx is not None:
                output = output[selected_idx]
            else:
                pseudo_logit = output[0]
                # output, selected_idx = select_confident_samples(output, args.selection_p)
                output, selected_idx = select_confident_samples_0(output, args.selection_p)
            
            original_output = output[0, :].unsqueeze(0)
            shift_outputs = output[1:, :]
            shift_outputs = shift_outputs.detach()
            all_outputs = torch.cat([original_output, shift_outputs], dim=0)

            t_similarity_loss = F.mse_loss(text_embedding, prototype_bank.detach()) #cosine similarity로도 실험
            # i_similarity_loss = F.mse_loss(transformed_embeddings.squeeze(0), image_embedding)
            mean_text_embedding = text_embedding.mean(dim=0, keepdim=True)
            prompt_similarity_loss = F.mse_loss(mean_text_embedding, image_embedding)
            # loss = avg_entropy(output) +  t_similarity_loss + prompt_similarity_loss
            # loss = prompt_similarity_loss

            # with torch.no_grad():
            #     similarities = F.cosine_similarity(image_embedding, prototype_bank.unsqueeze(0), dim=-1)
            original_label = pseudo_logit[0].argmax().item()
            pseudo_label = similarities.argmax().item()

            # softmax로 similarity 정규화 (soft target처럼 사용)
            weights = F.softmax(similarities / 0.07, dim=0)  # [num_classes]

            # weighted loss: 유사도가 높은 prototype일수록 더 많이 학습
            proto_loss = torch.sum(weights * (1 - similarities))
            r_conf = confidence_reward(similarities)

            contrastive_proto_loss = prototype_contrastive_loss(
                transformed_embeddings.squeeze(0).mean(dim=0), prototype_bank, pseudo_label, temperature=0.07
            )

            # print(transformed_embeddings.shape) # [1, 63, 512]
            # print(prototype_bank.shape)         # [100, 512]

            sim = F.cosine_similarity(transformed_embeddings.mean(dim=1).unsqueeze(1), prototype_bank.detach(), dim=-1)  # [1, num_classes]

            # print(sim.shape) #[1, 100]

            # Weighted similarity → weight가 큰 쪽은 유사도가 높아지도록 유도
            generator_loss = (1 - sim) * weights.unsqueeze(0)  # [1, num_classes]
            generator_loss = generator_loss.sum() 

            num_classes, D = prototype_bank.size()
            proto_logits_all = ((-1) * (args.beta - args.beta * similarities)).exp() 
            proto_logits = torch.zeros_like(proto_logits_all)
            proto_logits[:, pseudo_label] = proto_logits_all[:, pseudo_label]

            original_probs = F.softmax(original_output, dim=-1)  
            shift_probs = F.softmax(shift_outputs, dim=-1)  
            shift_probs_mean = shift_probs.mean(dim=0, keepdim=True)

            consistency_loss = F.mse_loss(shift_probs_mean, original_probs)
            pseudo_label = torch.tensor([pseudo_label], device=output.device)
            original_label = torch.tensor([original_label], device=output.device)
            cross_entropy_loss = F.cross_entropy(output[0].unsqueeze(0), pseudo_label)
            cross_entropy_loss1 = F.cross_entropy(output[0].unsqueeze(0), original_label)

            # loss = consistency_loss + generator_loss
            # loss = avg_entropy(all_outputs) + generator_loss + args.proto_lambda * proto_loss + r_conf + t_similarity_loss
            # if avg_entropy(output) < 1.1:
            #     weight = torch.clamp(1.0 - avg_entropy(output) / 2.0, min=0.0)
            #     loss = weight*avg_entropy(output) + t_similarity_loss
            # else:
            #     loss = t_similarity_loss
            # loss = t_similarity_loss
            # loss = avg_entropy(output) + generator_loss 
            loss = cross_entropy_loss
            # loss = avg_entropy(output) 

            #자기들끼리 멀어지게하는 로스 추가
            # print(f"avg_entropy(output) shape: {avg_entropy(output).shape}")  # 기대값: torch.Size([])
            # print(f"similarity_loss shape: {similarity_loss.shape}")  # 기대값: torch.Size([])
            # print(f"loss shape: {loss.shape}")  # 기대값: torch.Size([])
        model.prototype_bank.requires_grad = True
        for param in model.shift_generator.parameters():
            param.requires_grad = True
        for param in model.prompt_learner.parameters():
            param.requires_grad = True
        for name, param in model.named_parameters():
            if ('ln' in name):
                param.requires_grad_(True)
        trainable_ln = [
            param for name, param in model.named_parameters()
            if 'ln' in name
        ]
        # optimizer = torch.optim.AdamW([model.prototype_bank] + list(model.shift_generator.parameters()) + list(model.prompt_learner.parameters()), args.lr)
        # optimizer = torch.optim.AdamW([model.prompt_learner.residual_A] + [model.prompt_learner.residual_B] , lr=args.lr)
        optimizer = torch.optim.AdamW(list(model.prompt_learner.parameters()), lr=args.lr)
        # optimizer = torch.optim.AdamW(trainable_ln, args.lr)
        proto_logits = args.alpha * proto_logits
        optimizer.zero_grad()
        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.step(optimizer)
        scaler.update()
    if args.cocoop:
        return pgen_ctx

    return proto_logits


def main():
    args = parser.parse_args()
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
        classnames = imagenet_classes
    if args.cocoop:
        model = get_cocoop(args.arch, args.test_sets, 'cpu', args.n_ctx)
        assert args.load is not None
        load_model_weight(args.load, model, 'cpu', args) # to load to cuda: device="cuda:{}".format(args.gpu)
        model_state = deepcopy(model.state_dict())
    else:
        model = get_coop1(args.arch, args.test_sets, args.gpu, args.n_ctx, args.proto_ctx_init, args.ctx_init, affine_generator=False, shift_generator=True)
        if args.load is not None:
            print("Use pre-trained soft prompt (CoOp) as initialization")
            pretrained_ctx = torch.load(args.load)['state_dict']['ctx']
            assert pretrained_ctx.size()[0] == args.n_ctx
            with torch.no_grad():
                model.prompt_learner.ctx.copy_(pretrained_ctx)
                model.prompt_learner.ctx_init_state = pretrained_ctx
            
        # affine generator reset
        # if args.load1 is not None:
        #     print("Use pre-trained Affine Generator weights")
            
        #     checkpoint_affine = torch.load(args.load1)  # 🔥 Affine Generator 가중치 로드

        #     with torch.no_grad():
        #         if 'affine_generator' in checkpoint_affine['state_dict']:
        #             model.affine_generator.load_state_dict(checkpoint_affine['state_dict']['affine_generator'])
        #             model.affine_generator.initial_state_dict = model.affine_generator._get_current_state_dict()
        #             print("Affine Generator weights loaded successfully!")
        #         else:
        #             print("⚠️ Warning: 'affine_generator' not found in checkpoint.")
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
        if hasattr(model, "affine_generator"):
            print("affine_generator update")
            trainable_ln = [
                param for name, param in model.named_parameters()
                if 'ln' in name
            ]
            trainable_param = itertools.chain(
                model.prompt_learner.parameters(),
                model.affine_generator.parameters(),
                model.shift_generator.parameters(),
                [model.prototype_bank],
                trainable_ln
            )
        else:
            trainable_param = model.prompt_learner.parameters()
        model.prototype_bank.requires_grad = True
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
            # data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size-1, 
            #                                 augmix=len(set_id)>1)
            # agtpt
            data_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
                normalize,
            ])
            batchsize = 1
        elif args.agtpt:
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize])
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
            
        results[set_id] = test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args)
        del val_dataset, val_loader
        try:
            print("=> Acc. on testset [{}]: @1 {}/ @5 {}".format(set_id, results[set_id][0], results[set_id][1]))
        except:
            print("=> Acc. on testset [{}]: {}".format(set_id, results[set_id]))

    print("======== Result Summary ========")
    print("params: nstep	lr	bs   proto_reset_step")
    print("params: {}	{}	{}   {}".format(args.tta_steps, args.lr, args.batch_size, args.proto_reset_step))
    print("\t\t [set_id] \t\t Top-1 acc. \t\t Top-5 acc.")
    for id in results.keys():
        print("{}".format(id), end="	")
    print("\n")
    for id in results.keys():
        print("{:.2f}".format(results[id][0]), end="	")
    print("\n")


def test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

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
            #images = torch.cat(images, dim=0)
            images = images

        # reset the tunable prompt to its initial state
        if not args.cocoop: # no need to reset cocoop because it's fixed
            if args.reset:
                if args.tta_steps > 0:
                    with torch.no_grad():
                        model.reset()
            optimizer.load_state_dict(optim_state)
            # test_time_generator_tuning(model, images, optimizer, scaler, args)
            proto_logits = test_time_tuning(model, images, optimizer, scaler, args)
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
                    output, _, _, _, similarities = model(image, affine_generator=False, shift_generator=False)
        # measure accuracy and record loss
        updated_proto_logits = ((-1) * (args.beta - args.beta * similarities)).exp()
        updated_proto_logits = args.alpha * updated_proto_logits 

        # output += proto_logits
        # output += updated_proto_logits
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # acc1, acc5 = accuracy(proto_logits, target, topk=(1,5))
        # acc1, acc5 = accuracy(updated_proto_logits, target, topk=(1,5))
                
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            progress.display(i)

        if (i + 1) % args.proto_reset_step == 0:  # ✅ 5번째마다 prototype 리셋
            model.proto_reset()

    progress.display_summary()
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
    parser.add_argument('--proto_ctx_init', default=None, type=str, help='init proto prompts')
    parser.add_argument('--cocoop', action='store_true', default=False, help="use cocoop's output as prompt initialization")
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')
    # parser.add_argument('--load1', default=None, type=str, help='path to a pre-trained affine_generator')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--reset', action='store_true', help='Enable resetting before evaluation')
    parser.add_argument('--no-reset', dest='reset', action='store_false', help='Disable resetting before evaluation')
    parser.add_argument('--proto_lambda', default=1.0, type=float, help='lambda for prototype loss')
    parser.add_argument('--proto_contrastive_lambda', default=1.0, type=float, help='Weight for prototype contrastive loss')
    parser.add_argument('--alpha', default=1.0, type=float, help='lambda for proto logit')
    parser.add_argument('--beta', default=5.0, type=float, help='lambda for proto logit')
    parser.add_argument('--proto_reset_step', default=1.0, type=float, help='lambda for proto logit')
    parser.set_defaults(reset=True)

    main()