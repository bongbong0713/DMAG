# datasets/dtd.py


template = ['{} texture.']

class DescribableTextures:
    def __init__(self, root):
        self.cupl_path = './gpt3_prompts/CuPL_prompts_dtd.json'
        # self.cupl_path = './AWT_prompts/dtd.json'
        # self.cupl_path = './general_prompts/dtd1.json'
        self.template = template
