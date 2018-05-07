import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((384, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.486, 0.459, 0.408), (0.229, 0.224, 0.225))
])