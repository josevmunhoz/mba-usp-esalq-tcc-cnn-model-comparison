from torchvision import transforms

def train_transform(size=(128,128), grayscale=True) -> transforms:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if grayscale:
        t = transforms.Compose([
            transforms.Resize(size=size),
            transforms.Grayscale(),
            transforms.RandomHorizontalFlip(p=0.1),
            transforms.ToTensor(),
            normalize,
            ])
    else:
        t = transforms.Compose([
            transforms.Resize(size=size),
            transforms.RandomHorizontalFlip(p=0.1),
            transforms.ToTensor(),
            normalize,
            ])
    
    return t

def test_transform(size=(128,128), grayscale=True) -> transforms:
    if grayscale:
        t = transforms.Compose([
            transforms.Resize(size=size),
            transforms.Grayscale(),
            transforms.ToTensor(),
            ])
    else:
        t = transforms.Compose([
            transforms.Resize(size=size),
            transforms.ToTensor(),
            ])
    
    return t