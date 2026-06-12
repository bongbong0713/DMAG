# datasets/food101.py

template = ['a photo of {}, a type of food.']

class Food101:
    def __init__(self, root):
        self.template = template
        self.cupl_path = './gpt3_prompts/CuPL_prompts_food101.json'
        # self.cupl_path = './AWT_prompts/food101.json'