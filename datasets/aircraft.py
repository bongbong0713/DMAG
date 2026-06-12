# datasets/fgvcaircraft.py

template = ['a photo of a {}, a type of aircraft.']

class Aircraft:
    def __init__(self, root):
        self.template = template
        self.cupl_path = './gpt3_prompts/CuPL_prompts_fgvcaircraft.json'
        # self.cupl_path = './Attr_prompts/fgvc_dist.json'
        # self.cupl_path = './AWT_prompts/fgvc_aircraft.json'
        # self.cupl_path = './general_prompts/aircraft1.json'