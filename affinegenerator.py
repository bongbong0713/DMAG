import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from torchvision import transforms, datasets
from PIL import Image
import random
import os

# Load ImageNet class labels (extract only class names)
imagenet_classes = {line.strip().split(" ", 1)[0]: line.strip().split(" ", 1)[1] for line in open("classnames.txt")}

imagenet_templates = [
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}."
]

class AffineGenerator(nn.Module):
    def __init__(self, embed_dim, hidden_dim=256, num_transforms=5):
        super(AffineGenerator, self).__init__()
        self.num_transforms = num_transforms
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
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

    def forward(self, x):
        x = F.relu(self.fc1(x.float()))  # Ensure input is float32
        a = self.fc_a(x).view(-1, self.num_transforms, 1)  # Scalar scaling factors
        b = self.fc_b(x).view(-1, self.num_transforms, 1)  # Scalar shifting factors
        return a, b