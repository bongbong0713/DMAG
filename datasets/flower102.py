# datasets/oxfordflowers.py

template = ['a photo of a {}, a type of flower.']

class Flower102:
    def __init__(self, root):
        self.template = template
        self.cupl_path = './gpt3_prompts/CuPL_prompts_flowers102.json'
        # self.cupl_path = './AWT_prompts/oxford_flowers.json'