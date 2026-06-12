from .pets import pets
from .eurosat import EuroSAT
from .ucf101 import UCF101
from .sun397 import SUN397
from .caltech101 import Caltech101
from .dtd import DescribableTextures
from .aircraft import Aircraft
from .food101 import Food101
from .flower102 import Flower102
from .cars import cars
from .v import V
from .a import A
from .r import R
from .s import S
from .i import I


dataset_list = {
                "pets": pets,
                "eurosat": EuroSAT,
                "ucf101": UCF101,
                "sun397": SUN397,
                "caltech101": Caltech101,
                "dtd": DescribableTextures,
                "aircraft": Aircraft,
                "food101": Food101,
                "flower102": Flower102,
                "stanford_cars": cars,
                "imagenet-a": A,
                "imagenet-v": V,
                "imagenet-r": R,
                "imagenet-s": S,
                }


def build_dataset(dataset, root_path):
    return dataset_list[dataset](root_path)