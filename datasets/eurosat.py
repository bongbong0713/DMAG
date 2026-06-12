# datasets/eurosat.py

template = ['a centered satellite photo of {}.']

NEW_CLASSNAMES = {
    'AnnualCrop': 'Annual Crop Land',
    'Forest': 'Forest',
    'HerbaceousVegetation': 'Herbaceous Vegetation Land',
    'Highway': 'Highway or Road',
    'Industrial': 'Industrial Buildings',
    'Pasture': 'Pasture Land',
    'PermanentCrop': 'Permanent Crop Land',
    'Residential': 'Residential Buildings',
    'River': 'River',
    'SeaLake': 'Sea or Lake'
}

class EuroSAT:
    def __init__(self, root):
        self.template = template
        self.cupl_path = './gpt3_prompts/CuPL_prompts_eurosat.json'
        # self.cupl_path = './gpt3_prompts/CuPL_prompts_eurosat_noisy.json'
        # self.cupl_path = './general_prompts/eurosat1.json'
        # self.cupl_path = './AWT_prompts/eurosat.json'