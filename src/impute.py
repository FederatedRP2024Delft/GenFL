import random
from torchvision import transforms

import torch.utils.data
from src.vae.mnist_vae import ConditionalVae


def impute_cvae_naive(k, trained_cvae: ConditionalVae, initial_dataset):
    # Return a dataset imputed k images from trained_vae
    apply_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.round(x))  # Binarize the images
    ])
    # Return a dataset imputed k images from trained_vae
    generated_dataset = []
    uniform_digits = [random.randint(0, 9) for _ in range(k)]
    for i in uniform_digits:
        generated_image = trained_cvae.generate_data(n_samples=1, target_label=i).squeeze(1)
        multiplier = 1.0/generated_image.max().item()
        transformed_image = torch.round(generated_image*multiplier)
        generated_dataset.append((transformed_image, i))

    to_be_zipped = []
    for image_ind in range(k):
        to_be_zipped.append(
            (generated_dataset[image_ind][0], generated_dataset[image_ind][1]))
    return torch.utils.data.ConcatDataset([initial_dataset, to_be_zipped])