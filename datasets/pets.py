# datasets/oxfordpets.py

template = ['a photo of a {}, a type of pet.']

class pets:
    def __init__(self, root):
        self.template = template
        self.cupl_path = './gpt3_prompts/CuPL_prompts_oxfordpets.json'
        # self.cupl_path = './AWT_prompts/oxford_pets.json'