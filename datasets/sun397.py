# datasets/sun397.py

template = ["itap of a {}.",
            "a bad photo of the {}.",
            "a origami {}.",
            "a photo of the large {}.",
            "a {} in a video game.",
            "art of the {}.",
            "a photo of the small {}."]

class SUN397:
    def __init__(self, root):
        self.cupl_path = './gpt3_prompts/CuPL_prompts_sun397.json'
        # self.cupl_path = './AWT_prompts/sun397.json'
        self.template = template