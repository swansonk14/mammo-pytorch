import torch.utils.data as data
from PIL import Image
import os

def is_valid_image(path):
    if not os.path.exists(path):
        return False

    try:
        Image.open(path)
    except Exception:
        return False

    return True

def load_image(filepath):
    img = Image.open(filepath).convert('L')

    return img

class DatasetFromMetadata(data.Dataset):
    def __init__(self, metadata, input_transform=None, target_transform=None, target_string=''):
        super(DatasetFromMetadata, self).__init__()
        print 'start'
        import time
        start = time.time()
        self.metadata = filter(lambda x: is_valid_image(x['image_path']), metadata)
        print time.time() - start
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.target_string = target_string

    def __getitem__(self, index):
        input = load_image(self.metadata[index]['image_path'])
        target = self.metadata[index][self.target_string]
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.metadata)
