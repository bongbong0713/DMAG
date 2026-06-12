# datasets/ucf101.py


template = ['a photo of a person doing {}.']

class UCF101:
    def __init__(self, root):
        self.cupl_path = './gpt3_prompts/CuPL_prompts_ucf101.json'
        # self.cupl_path = './general_prompts/ucf.json'
        # self.cupl_path = './AWT_prompts/ucf101.json'

        self.template = template

