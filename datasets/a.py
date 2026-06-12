# datasets/imageneta.py

template = [
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
]

class A:
    def __init__(self, root):
        self.template = template
        self.cupl_path = './gpt3_prompts/CuPL_prompts_imagenet.json'
        # self.cupl_path = './gpt4_prompts/CuPL_full_gpt4_merged.json'
        # self.cupl_path = './AWT_prompts/imagenet.json'
        # self.cupl_path = './RA-TTA_prompts/CuPL_prompts_imagenet_adversarial.json'