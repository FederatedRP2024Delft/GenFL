import random

import torch.utils.data
from vae.mnist_vae import ConditionalVae


def impute_cvae_naive(k, trained_cvae: ConditionalVae, initial_dataset):
    # Return a dataset imputed k images from trained_vae
    generated_dataset = []
    uniform_digits = [random.randint(0, 9) for _ in range(k)]
    digit_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for i in uniform_digits:
        digit_count[i] += 1
    print(digit_count)

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