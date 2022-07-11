import os
import time
import pandas as pd

import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.utils.data import DataLoader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_labels = pd.DataFrame(os.listdir(img_dir))
        self.img_labels["class"] = 0
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label


class EmbedModel(nn.Module):
    def __init__(self, layer_output_size=512) -> None:
        super().__init__()
        self.model, self.extraction_layer = self._get_model_and_layer()
        self.layer_output_size = layer_output_size
        self.model.eval()

    def _get_model_and_layer(self):
        model = models.resnet18(pretrained=True)
        layer = model._modules.get("avgpool")
        return model, layer

    def forward(self, img):
        my_embedding = torch.zeros(len(img), self.layer_output_size, 1, 1)

        def copy_data(m, i, o):
            my_embedding.copy_(o.data)

        h = self.extraction_layer.register_forward_hook(copy_data)
        with torch.no_grad():
            h_x = self.model(img)
        h.remove()
        embeddings = my_embedding.numpy()[:, :, 0, 0]

        pd.DataFrame(embeddings).to_csv(
            "./embeddings.csv", index=False, mode="a", header=None
        )
        return None


transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

image_dir = "./sample_images/0"
dataset = CustomImageDataset(img_dir=image_dir, transform=transform)
train_dataloader = DataLoader(dataset, batch_size=200)

model = EmbedModel()
if torch.cuda.device_count() > 1:
    print(torch.cuda.device_count(), "GPUs in use!")
    model = nn.DataParallel(model)

model.to(device)

start = time.time()
for inputs, _ in train_dataloader:
    inputs = inputs.to(device)
    embeddings = model(inputs)

print('Time: ', time.time() - start)
