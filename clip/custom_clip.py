
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import clip
from clip import load, tokenize
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from data.imagnet_prompts import imagenet_classes
from data.fewshot_datasets import fewshot_datasets
from data.cls_to_names import *
import importlib
from datasets.utils import DatasetBase
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask

_tokenizer = _Tokenizer()

DOWNLOAD_ROOT='~/.cache/clip'

class ClipImageEncoder(nn.Module):
    def __init__(self, device, arch="ViT-L/14", image_resolution=224, n_class=1000):
        super(ClipImageEncoder, self).__init__()
        clip, embed_dim, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.encoder = clip.visual
        del clip.transformer
        torch.cuda.empty_cache()
        
        self.cls_head = nn.Linear(embed_dim, n_class)
    
    @property
    def dtype(self):
        return self.encoder.conv1.weight.dtype

    def forward(self, image):
        x = self.encoder(image.type(self.dtype))
        output = self.cls_head(x)
        return output


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, batch_size=None, n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False):
        super().__init__()
        n_cls = len(classnames)
        self.learned_cls = learned_cls
        dtype = clip_model.dtype
        self.dtype = dtype
        self.device = clip_model.visual.conv1.weight.device
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        self.batch_size = batch_size

        # self.ctx, prompt_prefix = self.reset_prompt(ctx_dim, ctx_init, clip_model)

        if ctx_init:
            # use given words to initialize context vectors
            print("Initializing the contect with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            if '[CLS]' in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                ctx_position = "middle"
            else:
                split_idx = None
            self.split_idx = split_idx
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        

        self.prompt_prefix = prompt_prefix

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # batch-wise prompt tuning for test-time adaptation
        if self.batch_size is not None: 
            ctx_vectors = ctx_vectors.repeat(batch_size, 1, 1)  #(N, L, D)
        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors) # to be optimized

        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
        else:
            print("Random initialization: initializing a learnable class token")
            cls_vectors = torch.empty(n_cls, 1, ctx_dim, dtype=dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [prompt_prefix + " " + cls_token + "." for _ in classnames]

            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors) # to be optimized

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if self.learned_cls:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + 1:, :])  # ..., EOS
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.ctx_init = ctx_init
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = ctx_position
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.classnames = classnames
        # self.residual = nn.Parameter(torch.zeros_like(self.ctx))
        # r = 16
        # self.residual_A = nn.Parameter(torch.empty(self.n_ctx, r))     # r << ctx_dim
        # self.residual_B = nn.Parameter(torch.empty(r, self.ctx_dim))   # low-rank
        # nn.init.normal_(self.residual_A, std=0.02)
        # nn.init.normal_(self.residual_B, std=0.02)

    def reset(self):
        ctx_vectors = self.ctx_init_state
        self.ctx.copy_(ctx_vectors) # to be optimized
        if self.learned_cls:
            cls_vectors = self.cls_init_state
            self.cls.copy_(cls_vectors)

    def reset_classnames(self, classnames, arch):
        self.n_cls = len(classnames)
        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        else:
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim, dtype=self.dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [self.prompt_prefix + " " + cls_token + "." for _ in classnames]
            # TODO: re-init the cls parameters
            # self.cls = nn.Parameter(cls_vectors) # to be optimized
            self.cls_init_state = cls_vectors.detach().clone()
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)

        clip, _, _ = load(arch, device=self.device, download_root=DOWNLOAD_ROOT)

        with torch.no_grad():
            embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)

        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1 + self.n_ctx :, :]  # CLS, EOS

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.classnames = classnames

    def forward(self, init=None):
        # the init will be used when computing CLIP directional loss
        if init is not None:
            ctx = init
        else:
            # residual = self.residual_A @ self.residual_B  # (n_ctx, ctx_dim)
            # ctx = self.ctx + residual
            # ctx = self.ctx + self.residual
            ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        elif not ctx.size()[0] == self.n_cls:
            ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        if self.batch_size is not None: 
            # This way only works for single-gpu setting (could pass batch size as an argument for forward())
            prefix = prefix.repeat(self.batch_size, 1, 1, 1)
            suffix = suffix.repeat(self.batch_size, 1, 1, 1)

        if self.learned_cls:
            assert self.class_token_position == "end"
        if self.class_token_position == "end":
            if self.learned_cls:
                cls = self.cls
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        cls,     # (n_cls, 1, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
            else:
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
        elif self.class_token_position == "middle":
            # TODO: to work with a batch of prompts
            if self.split_idx is not None:
                half_n_ctx = self.split_idx # split the ctx at the position of [CLS] in `ctx_init`
            else:
                half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

class AffineGenerator(nn.Module):
    def __init__(self, embed_dim, num_classes, hidden_dim=256, num_transforms=64):
        super(AffineGenerator, self).__init__()
        self.num_transforms = num_transforms
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.fc1 = nn.Linear(embed_dim + num_classes, hidden_dim)
        self.fc1_n = nn.Linear(embed_dim, hidden_dim)
        self.fc_a = nn.Linear(hidden_dim, num_transforms)  # Generate scalar a for each transform
        self.fc_b = nn.Linear(hidden_dim, num_transforms)  # Generate scalar b for each transform
        
        # 🔥 Xavier 초기화 적용
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc_a.weight)
        nn.init.xavier_uniform_(self.fc_b.weight)

        # Bias 초기화
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc_a.bias)
        nn.init.zeros_(self.fc_b.bias)

    def forward(self, x, logits=None):
        x = x.float()  # [B, D]

        if logits is not None:
            logits = logits.float().view(1, -1)  # [C, 1] → [1, C]
            logits = logits.expand(x.size(0), -1)  # → [B, C]
            x = torch.cat([x, logits], dim=-1)  # → [B, D+C]
            x = F.relu(self.fc1(x))
        else:
            x = F.relu(self.fc1_n(x))
        a = self.fc_a(x).view(-1, self.num_transforms, 1)  # Scalar scaling factors
        b = self.fc_b(x).view(-1, self.num_transforms, 1)  # Scalar shifting factors
        return a, b

    def reset(self):
        """ 기존의 가중치로 다시 초기화하는 함수 """
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc_a.weight)
        nn.init.xavier_uniform_(self.fc_b.weight)

        # Bias 초기화
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc_a.bias)
        nn.init.zeros_(self.fc_b.bias)

class ShiftGenerator(nn.Module):
    def __init__(self, embed_dim, num_classes, hidden_dim=256, num_transforms=64):
        super(ShiftGenerator, self).__init__()
        self.num_transforms = num_transforms
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.fc1 = nn.Linear(embed_dim * 2, hidden_dim)
        self.fc1_n = nn.Linear(embed_dim, hidden_dim)
        self.fc_s = nn.Linear(hidden_dim, embed_dim*num_transforms)  # Generate shifter a for each transform
        
        # 🔥 Xavier 초기화 적용
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc_s.weight)

        # Bias 초기화
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc_s.bias)

    def forward(self, x, logits=None):
        x = x.float()  # [B, D]

        # if logits is not None:
        #     logits = logits.float().view(1, -1)  # [C, 1] → [1, C]
        #     logits = logits.expand(x.size(0), -1)  # → [B, C]
        #     x = torch.cat([x, logits], dim=-1)  # → [B, D+C]
        #     x = F.relu(self.fc1(x))
        # else:
        #     x = F.relu(self.fc1_n(x))

        if logits is not None:
            # Normalize
            x_norm = F.normalize(x, dim=-1)
            text_norm = F.normalize(logits, dim=-1)
            similarity = torch.matmul(x_norm, text_norm.T)       # [B, C]
            weights = F.softmax(similarity, dim=-1)              # [B, C]
            text_summary = torch.matmul(weights, logits)  # [B, D]
            x = torch.cat([x, text_summary], dim=-1)             # [B, 2D]
            x = F.relu(self.fc1(x))
        else:
            x = F.relu(self.fc1_n(x))
        s = self.fc_s(x).view(1, self.num_transforms, -1)  # Scalar scaling factors

        return s

    def reset(self):
         # 🔥 Xavier 초기화 적용
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc_s.weight)

        # Bias 초기화
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc_s.bias)

def clip_classifier(classnames, template, cupl_path, clip_model, coop=False, backbone='RN50'):
    # 디바이스 자동 감지
    device = next(clip_model.parameters()).device

    # CoOp 프롬프트 사용 시
    if coop:
        n_ctx = 4
        if backbone == 'RN50':
            print('Using CoOp weights (RN50) for initialization.')
            coop_path = '/home/ce/DiffTPT/coop_weights/rn50_ep50_16shots/nctx4_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50'
        elif backbone == 'ViT-B/16':
            print('Using CoOp weights (ViT-B/16) for initialization.')
            coop_path = '/home/ce/DiffTPT/coop_weights/vit_b16_ep50_16shots/nctx4_cscFalse_ctpend/seed2/prompt_learner/model.pth.tar-50'
        ctx = torch.load(coop_path, map_location=device)['state_dict']['ctx'].unsqueeze(0).to(device)

    # CuPL 프롬프트 로딩
    with open(cupl_path, 'r') as f:
        cupl = json.load(f)

    # OpenCLIP tokenizer 지정
    if backbone == 'OpenCLIP':
        tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')

    with torch.no_grad():
        clip_weights = []
        per_class_embeddings = [] 
        lengths =[]

        for classname in classnames:
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts += cupl[classname]

            if coop:
                prompts = [f'a photo of a {classname}.']
                tokenized_prompts = clip.tokenize(prompts).to(device)
                embedding = clip_model.token_embedding(tokenized_prompts).type(clip_model.visual.conv1.weight.dtype)

                prefix = embedding[:, :1, :]
                suffix = embedding[:, 1 + n_ctx :, :]

                prompts = torch.cat([prefix, ctx, suffix], dim=-2)

                text_encoder_w_prompt = TextEncoderWithPrompt(clip_model)
                class_embedding = text_encoder_w_prompt(prompts, tokenized_prompts).squeeze()
            else:
                if backbone == 'RN50' or backbone == 'ViT-B/16':
                    texts = clip.tokenize(texts).to(device)
                elif backbone == 'OpenCLIP':
                    texts = tokenizer(texts).to(device)

                class_embeddings = clip_model.encode_text(texts)  # [num_prompts, D]
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)  # normalize
                class_embedding = class_embeddings.mean(dim=0)  # [D]
                class_embedding /= class_embedding.norm()

            per_class_embeddings.append(class_embeddings)
            lengths.append(class_embeddings.size(0))
            clip_weights.append(class_embedding)
        
        # --- 여기서부터 패딩 + 마스크 구성 ---
        C = len(per_class_embeddings)
        D = per_class_embeddings[0].size(1)
        P_max = max(lengths)

        # [C, P_max, D] 로 0-패딩 텐서 준비
        all_weights = torch.zeros(C, P_max, D, device=device)
        # 유효 프롬프트 위치를 표시하는 마스크
        all_mask = torch.zeros(C, P_max, dtype=torch.bool, device=device)

        for i, e in enumerate(per_class_embeddings):
            P_i = e.size(0)
            all_weights[i, :P_i] = e
            all_mask[i, :P_i] = True

        # 최종 스택 및 정규화
        # all_weights = torch.stack(per_class_embeddings, dim=0)
        clip_weights = torch.stack(clip_weights, dim=0).to(device)  # [num_classes, D]

    return clip_weights, all_weights, all_mask

def get_templates(test_set: str):
    """
    test_set 이름에 해당하는 datasets 모듈을 불러와 template과 cupl_path를 반환.

    Args:
        test_set (str): 예: "caltech101", "food101", "oxford_pets"

    Returns:
        template (list of str), cupl_path (str)
    """
    # 파일명이 소문자인 경우를 가정
    test_set = test_set.lower()
    dataset_module = importlib.import_module(f"datasets.{test_set}")

    # datasets.{test_set} 안의 첫 번째 클래스 찾기 (예: Caltech101)
    for attr_name in dir(dataset_module):
        attr_val = getattr(dataset_module, attr_name)
        if isinstance(attr_val, type) and attr_val.__module__ == dataset_module.__name__:
            dataset_class = attr_val
            break
    else:
        raise RuntimeError(f"No class found in datasets.{test_set}")
    
    dataset_instance = dataset_class(root='../multimodal-prompt-learning/data')  # 루트 경로 맞게 지정
    return dataset_instance.template, dataset_instance.cupl_path

class ClipTestTimeTuning1(nn.Module):
    def __init__(self, device, classnames, batch_size, criterion='cosine', arch="ViT-L/14",
                        n_ctx=16, proto_ctx_init=None, ctx_init=None, ctx_position='end', num_transforms=63, learned_cls=False, affine_generator=False, shift_generator=False, test_set=None):
        super(ClipTestTimeTuning1, self).__init__()
        clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.image_encoder = clip.visual
        self.text_encoder = TextEncoder(clip)
        self.logit_scale = clip.logit_scale.data
        # prompt tuning
        self.prompt_learner = PromptLearner(clip, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls)
        self.prompt_learner = self.prompt_learner.to(device)
        # self.proto_prompt_learner = PromptLearner(clip, classnames, batch_size, n_ctx, proto_ctx_init, ctx_position, learned_cls)
        self.criterion = criterion
        self.affine_generator = AffineGenerator(clip.visual.output_dim, num_classes=len(classnames), num_transforms=num_transforms)
        self.affine_generator = self.affine_generator.to(device)
        self.shift_generator = ShiftGenerator(clip.visual.output_dim, num_classes=len(classnames), num_transforms=num_transforms)
        self.test_set = test_set
        self.classnames = classnames
        self.arch = arch
        self.clip = clip
        self.template, self.cupl_path = get_templates(test_set)
        self.init_proto = clip_classifier(classnames, self.template, self.cupl_path, clip, coop=False, backbone=arch)
        self.prototype_bank = nn.Parameter(self.init_proto.clone().to(device).requires_grad_())
        self.device = device
        self.use_affine_generator = affine_generator
        self.use_shift_generator = shift_generator

        # with torch.no_grad():
        #     init_proto = self.get_text_features().to(device)  # [num_classes, D]
        #     # init_proto = F.normalize(init_proto, dim=-1)

        # self.prototype_bank = nn.Parameter(init_proto.clone().detach().to(device), requires_grad=True)

    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()
        # self.affine_generator.reset()
        # self.shift_generator.reset()

    def proto_reset(self):
        self.prototype_bank = nn.Parameter(self.init_proto.clone().to(self.device).requires_grad_())

    def reset_classnames(self, classnames, arch):
        self.prompt_learner.reset_classnames(classnames, arch)
        self.affine_generator = AffineGenerator(
            embed_dim=self.image_encoder.output_dim,  # 또는 clip.visual.output_dim
            num_classes=len(classnames),
            num_transforms=self.affine_generator.num_transforms  # 기존 값 유지
        ).to(next(self.image_encoder.parameters()).device)
        self.shift_generator = ShiftGenerator(
            embed_dim=self.image_encoder.output_dim,  # 또는 clip.visual.output_dim
            num_classes=len(classnames),
            num_transforms=self.shift_generator.num_transforms  # 기존 값 유지
        ).to(next(self.image_encoder.parameters()).device)  
        self.init_proto = clip_classifier(classnames, self.template, self.cupl_path, self.clip, coop=False, backbone=arch)
        self.prototype_bank = nn.Parameter(self.init_proto.clone().to(next(self.image_encoder.parameters()).device).requires_grad_())

    def get_text_features(self):
        text_features = []
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        t_features = self.text_encoder(prompts, tokenized_prompts)
        text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        text_features = torch.stack(text_features, dim=0)

        return torch.mean(text_features, dim=0)

    def get_proto_features(self):
        text_features = []
        prompts = self.proto_prompt_learner()
        tokenized_prompts = self.proto_prompt_learner.tokenized_prompts
        t_features = self.text_encoder(prompts, tokenized_prompts)
        text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        text_features = torch.stack(text_features, dim=0)

        return torch.mean(text_features, dim=0)

    def inference(self, image, affine_generator=False, shift_generator=False):
        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))

        text_features = self.get_text_features()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # if not hasattr(self, "prototype_bank"):
        #     print("🚀 Initializing prototype_bank...")
        #     # init_proto = self.get_proto_features().to(image.device)
        #     template, cupl_path = get_templates(self.test_set)
        #     init_proto = clip_classifier(self.classnames, template, cupl_path, self.clip, coop=False, backbone=self.arch)
        #     print(init_proto.shape)
        #     self.prototype_bank = nn.ParameterList([
        #         nn.Parameter(init_proto.clone().detach().to(device), requires_grad=True)
        #     ])

        logit_scale = self.logit_scale.exp()
        proto_logits = logit_scale * image_features @ self.prototype_bank.t()
        logits = logit_scale * image_features @ text_features.t()

        if affine_generator==True:
            a, b = self.affine_generator(image_features, logits)
            transformed_embeddings = a * image_features.unsqueeze(1) + b
            transformed_embeddings = F.normalize(transformed_embeddings, dim=-1)  # shape: [B, N, D]
            text_features = F.normalize(text_features, dim=-1)
            logits_transformed = logit_scale * (transformed_embeddings @ text_features.t().unsqueeze(0))  # (B, N, T)\
            # 최종 logits 결합
            logits = torch.cat([logits.unsqueeze(1), logits_transformed], dim=1)  # (B, N+1, T)

            import os
            import numpy as np

            os.makedirs("outputs", exist_ok=True)
            np.savetxt("outputs/a.txt", a.detach().cpu().numpy().reshape(-1, a.size(-1)), fmt="%.6f")
            np.savetxt("outputs/b.txt", b.detach().cpu().numpy().reshape(-1, b.size(-1)), fmt="%.6f")
            logits_np = logits.detach().cpu().numpy().squeeze(0)  # shape: [transforms, classes]

            with open("outputs/logits.txt", "w") as f:
                for row in logits_np:
                    line = " ".join([f"{x:.6f}" for x in row])
                    f.write(line + "\n")

        if shift_generator==True:
            s = self.shift_generator(image_features, text_features)
            transformed_embeddings = image_features.expand(s.shape[0], -1) + s
            transformed_embeddings = F.normalize(transformed_embeddings, dim=-1)  # shape: [B, N, D]
            text_features = F.normalize(text_features, dim=-1)
            logits_transformed = logit_scale * (transformed_embeddings @ text_features.t().unsqueeze(0))  # (B, N, T)\
            logits = torch.cat([logits.unsqueeze(1), logits_transformed], dim=1)  # (B, N+1, T)
        
        similarities = F.cosine_similarity(image_features, self.prototype_bank.unsqueeze(0), dim=-1)

        alpha = 0.5  # blending coefficient
        weights = F.softmax(similarities / 0.07, dim=0)
        proto_blend = weights @ self.prototype_bank  # [B, D]
        transformed_features = (1 - alpha) * image_features + alpha * proto_blend

        if affine_generator==True or shift_generator==True:
            return logits, transformed_embeddings, image_features, text_features, self.prototype_bank, similarities
        else:
            return logits, image_features, text_features, self.prototype_bank, similarities

    def forward(self, input, affine_generator=None, shift_generator=None):
        if affine_generator is None:
            affine_generator = self.use_affine_generator
        if shift_generator is None:
            shift_generator = self.use_shift_generator

        if isinstance(input, Tuple):
            view_0, view_1, view_2 = input
            return self.contrast_prompt_tuning(view_0, view_1, view_2)
        elif len(input.size()) == 2:
            return self.directional_prompt_tuning(input)
        else:
            return self.inference(input, affine_generator=affine_generator, shift_generator=shift_generator)

def get_custom_text_feature(clip_model, text: str, normalize: bool = True):
        """
        한 문장에서 CLIP text embedding 추출
        Args:
            clip_model : openai.clip.load(...) 로 받은 모델 객체
            text (str) : 예) "This image contains an airplane"
            normalize  : L2 정규화 여부
        Returns:
            Tensor shape [1, D]
        """
        device = next(clip_model.parameters()).device
        dtype  = clip_model.visual.conv1.weight.dtype

        # 기본 tokenizer(_Tokenizer) 그대로 사용
        tokenized = _Tokenizer().encode(text)
        token_tensor = torch.zeros(1, 77, dtype=torch.long, device=device)
        token_tensor[0, :len(tokenized)] = torch.tensor(tokenized, device=device)

        with torch.no_grad():
            x = clip_model.token_embedding(token_tensor).type(dtype)
            x = x + clip_model.positional_embedding.type(dtype)
            x = x.permute(1, 0, 2)
            x = clip_model.transformer(x)
            x = x.permute(1, 0, 2)
            x = clip_model.ln_final(x).type(dtype)
            x = x[torch.arange(x.shape[0]), token_tensor.argmax(dim=-1)] \
                @ clip_model.text_projection        # [1, D]

            if normalize:
                x = x / x.norm(dim=-1, keepdim=True)
        return x

class ClipTestTimeTuning(nn.Module):
    def __init__(self, device, classnames, batch_size, criterion='cosine', arch="ViT-L/14",
                        n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False, test_set=None):
        super(ClipTestTimeTuning, self).__init__()
        clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.image_encoder = clip.visual
        self.text_encoder = TextEncoder(clip)
        self.logit_scale = clip.logit_scale.data
        # prompt tuning
        self.prompt_learner = PromptLearner(clip, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls)
        self.criterion = criterion
        self.test_set = test_set
        self.classnames = classnames
        self.arch = arch
        self.clip = clip
        self.template, self.cupl_path = get_templates(test_set)
        self.init_proto, self.all_des, self.all_mask = clip_classifier(classnames, self.template, self.cupl_path, clip, coop=False, backbone=arch)
        # print(self.all_des.shape) # [C, P, D]
        self.prototype_bank = nn.Parameter(self.init_proto.clone().to(device).requires_grad_())
        # texts = [
        #     "This is a clear and informative view of the object in the image.",
        #     "This image highlights the key features of the object in focus.",
        #     "This crop is semantically rich and offers an in-depth look at the object.",
        #     "The object’s shape and structure are prominently displayed in this image.",
        #     "The object is captured from an interesting and useful angle.",
        #     "This image provides clear details of the object's unique characteristics.",
        #     "The object is clearly defined, and its features are well-exposed in this view.",
        #     "This image focuses on the main elements of the object and minimizes distractions.",
        #     "The crop in this image gives a detailed perspective of the object's surface.",
        #     "This view allows a full and clear look at the object's composition.",
        #     "The image provides a distinct and focused view of the object in question.",
        #     "The object’s features and geometry are clearly emphasized in this photo.",
        #     "This crop preserves the essential details of the object without background interference.",
        #     "This view brings out the key visual traits of the object.",
        #     "The object is centered and well-lit, making its shape easily identifiable.",
        #     "This image offers a clean, uncluttered view of the object from a useful angle.",
        #     "This crop focuses on the object, highlighting important visual features.",
        #     "The object stands out clearly, with minimal occlusions or distractions.",
        #     "The object’s outline and key features are prominent and clear in this view.",
        #     "This image presents the object in a manner that aligns with its most important characteristics."
        # ]

        # features = [get_custom_text_feature(self.clip, text) for text in texts]  # list of [1, D] tensors
        # ensemble_feature = torch.mean(torch.stack(features), dim=0)         # shape: [1, D]
        # self.custom_text_feature = F.normalize(ensemble_feature, dim=-1) 
        
    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()

    def reset_classnames(self, classnames, arch):
        self.prompt_learner.reset_classnames(classnames, arch)
        self.init_proto, self.all_des, self.all_mask = clip_classifier(classnames, self.template, self.cupl_path, self.clip, coop=False, backbone=arch)
        self.prototype_bank = nn.Parameter(self.init_proto.clone().to(next(self.image_encoder.parameters()).device).requires_grad_())

    def get_text_features(self):
        text_features = []
        prompts = self.prompt_learner()
        # print(prompts.shape) #[100, 77, 512]
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        # print(tokenized_prompts.shape) #[100, 77]
        t_features = self.text_encoder(prompts, tokenized_prompts)
        text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        text_features = torch.stack(text_features, dim=0)

        return torch.mean(text_features, dim=0)

    def inference(self, image):
        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))

        text_features = self.get_text_features()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        proto_logits = logit_scale * image_features @ self.prototype_bank.t()
        # print(logits.shape)
        # custom text랑 similarity 계산
        # similarities = torch.matmul(image_features, self.custom_text_feature.T).squeeze(1)  # shape: [B]

        # sorted_sim_values, sorted_indices = similarities.sort(descending=True)

        # top5_indices = sorted_indices[:1]
        top5_indices = None

        return logits, proto_logits, image_features, text_features, self.prototype_bank, top5_indices, logit_scale, self.all_des, self.all_mask

    def forward(self, input):
        if isinstance(input, Tuple):
            view_0, view_1, view_2 = input
            return self.contrast_prompt_tuning(view_0, view_1, view_2)
        elif len(input.size()) == 2:
            return self.directional_prompt_tuning(input)
        else:
            return self.inference(input)

def get_coop1(clip_arch, test_set, device, n_ctx, proto_ctx_init, ctx_init, learned_cls=False, affine_generator=False, shift_generator=False):
    if test_set in fewshot_datasets:
        classnames = eval("{}_classes".format(test_set.lower()))
    elif test_set == 'bongard':
        if learned_cls:
            classnames = ['X', 'X']
        else:
            classnames = ['True', 'False']
    else:
        classnames = imagenet_classes

    model = ClipTestTimeTuning1(device, classnames, None, arch=clip_arch,
                            n_ctx=n_ctx, proto_ctx_init=proto_ctx_init, ctx_init=ctx_init, learned_cls=learned_cls, affine_generator=affine_generator, shift_generator=shift_generator, test_set=test_set)

    return model

def get_coop(clip_arch, test_set, device, n_ctx, ctx_init, learned_cls=False):
    if test_set in fewshot_datasets:
        classnames = eval("{}_classes".format(test_set.lower()))
    elif test_set == 'bongard':
        if learned_cls:
            classnames = ['X', 'X']
        else:
            classnames = ['True', 'False']
    else:
        assert test_set in ['A', 'R', 'K', 'V', 'I']
        classnames_all = imagenet_classes
        classnames = []
        if test_set in ['A', 'R', 'V']:
            label_mask = eval("imagenet_{}_mask".format(test_set.lower()))
            if test_set == 'R':
                for i, m in enumerate(label_mask):
                    if m:
                        classnames.append(classnames_all[i])
            else:
                classnames = [classnames_all[i] for i in label_mask]
        else:
            classnames = classnames_all

    model = ClipTestTimeTuning(device, classnames, None, arch=clip_arch,
                            n_ctx=n_ctx, ctx_init=ctx_init, learned_cls=learned_cls, test_set=test_set)

    return model

