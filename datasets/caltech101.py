# datasets/caltech101.py

template = [
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}."
]

class Caltech101:
    def __init__(self, root):
        self.template = template
        self.cupl_path = './gpt3_prompts/CuPL_prompts_caltech101.json'
        # self.cupl_path = './AWT_prompts/caltech101.json'
        