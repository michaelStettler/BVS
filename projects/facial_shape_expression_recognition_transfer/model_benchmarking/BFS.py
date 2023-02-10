"""
Module to load the BFS dataset into Torch format.
"""
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

class BFS():
    def __init__(self, path, batch_size=300):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([transforms.Resize((224, 224)),
                                                            transforms.ToTensor(),
                                                            normalize])

        self.test_set = datasets.ImageFolder(
            root=path,
            transform=transform
        )
        self.test_loader = DataLoader(
            self.test_set, batch_size=batch_size, shuffle=False
        )
