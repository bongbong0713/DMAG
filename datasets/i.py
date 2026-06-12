# datasets/imagenet.py

imagenet_templates = [
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}."
]

class I:
    def __init__(self, root):
        self.template = imagenet_templates
        self.cupl_path = './gpt3_prompts/CuPL_prompts_imagenet.json'
