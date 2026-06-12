# datasets/stanfordcars.py

template = [
    'a photo of a {}.',
    'A {} featuring a wide range of color options for easy selection.'
]

class cars:
    def __init__(self, root):
        self.template = template
        self.cupl_path = './gpt3_prompts/CuPL_prompts_stanfordcars.json'
        # self.cupl_path = './AWT_prompts/stanford_cars.json'