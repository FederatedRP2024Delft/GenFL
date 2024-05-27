from src.utils import kl_loss, vae_loss_fn, vae_classifier_loss_fn, reg_loss_fn

import torch
from torch import Tensor, tensor, device, cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal


INPUT_SIZE = 784
HIDDEN_LAYER_SIZE_1 = 512
HIDDEN_LAYER_SIZE_2 = 256
DEFAULT_DIMENSION_ENCODING = 2

device = device('cuda' if cuda.is_available() else 'cpu')


class VaeEncoder(nn.Module):
    def __init__(self, dim_encoding):
        """
        dim_encoding - dimensionality of the latent space
        encoder outputs two parameters per dimension of the latent space, which is typical for VAEs
        """
        super(VaeEncoder, self).__init__()

        # linear layer that takes in MNIST input size
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_LAYER_SIZE_1)

        self.fc2 = nn.Linear(HIDDEN_LAYER_SIZE_1, HIDDEN_LAYER_SIZE_2)

        # linear layer that takes output of fc1 and transforms it to dim_encoding
        # dim_encoding * 2, otherwise sigma is null
        self.fc3 = nn.Linear(HIDDEN_LAYER_SIZE_2, dim_encoding * 2)

    def forward(self, x: Tensor) -> Tensor:
        """
        takes in input of tensor x and outputs tensor of the latent space vectors.

        For example, given that x has 6 data points and VAE has latent space of 2 dimensions:

        x: torch.Size([6, 1, 28, 28])
        output: torch.Size([6, 2])
        """
        # reshapes each data point tensor from its original shape to a 1D tensor from 2nd dimension onwards
        # e.g. torch.Size([6, 1, 28, 28]) -> torch.Size([6, 784])
        x = torch.flatten(x, start_dim=1)

        # pass through fc1 followed by ReLU activation
        x = F.relu(self.fc1(x))

        # pass through fc1 followed by ReLU activation
        x = F.relu(self.fc2(x))

        # absence of an activation function means that the output can be any real-valued number
        return self.fc3(x)

class ConditionalVaeDecoder(nn.Module):
    """
    Classifier decoder that outputs both images and its corresponding vector of label probabilities
    """
    def __init__(self, dim_encoding):
        super(ConditionalVaeDecoder, self).__init__()
        self.fc1 = nn.Linear(dim_encoding + 10, HIDDEN_LAYER_SIZE_2)
        self.fc2 = nn.Linear(HIDDEN_LAYER_SIZE_2, HIDDEN_LAYER_SIZE_1)
        self.fc3 = nn.Linear(HIDDEN_LAYER_SIZE_1, INPUT_SIZE)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        # match input shape back to 28x28 pixels
        return x.reshape(-1, 1, 28, 28)


class ConditionalVae(nn.Module):
    """
    Classifier decoder that returns both images and its corresponding vector of label probabilities
    """
    def __init__(self, dim_encoding):
        super(ConditionalVae, self).__init__()
        self.z_dist = None
        self.encodings = None
        self.latent_space_vector = None
        self.dim_encoding = dim_encoding
        self.encoder = VaeEncoder(dim_encoding)
        self.decoder = ConditionalVaeDecoder(dim_encoding)

    def reparameterize(self, encodings: Tensor) -> Tensor:
        mu = encodings[:, :self.dim_encoding]

        # must do exponential, otherwise get value error that not all positive
        sigma = 1e-6 + F.softplus(encodings[:, self.dim_encoding:])

        z_dist = Normal(mu, sigma)
        self.z_dist = z_dist
        z = z_dist.rsample()
        return z

    def forward(self, x: Tensor, y: Tensor) -> tensor:
        """
        After encoder compresses input data into encodings, this method performs re-parameterization to convert
        them to a latent space vector (has normal distribution).

        Decoder then returns a tensor of images and label probabilities
        """
        device = next(self.parameters()).device
        x = x.to(device)
        y = y.to(device)
        encodings = self.encoder(x)
        self.encodings = encodings
        z = self.reparameterize(encodings)

        assert z.shape[1] == self.dim_encoding
        self.latent_space_vector = z

        # Create one-hot vector from labels
        y = y.view(x.shape[0], 1)
        onehot_y = torch.zeros((x.shape[0], 10), device=device, requires_grad=False)
        onehot_y.scatter_(1, y, 1)

        latent = torch.cat((self.latent_space_vector, onehot_y), dim=1)
        return self.decoder(latent)

    def train_model(
            self,
            training_data,
            batch_size=64,
            beta=1.0,
            epochs=5,
            learning_rate=0.01
    ) -> tuple[nn.Module, list, list, list]:
        vl_fn = vae_loss_fn(beta)
        kl_div_fn = kl_loss()
        reg = reg_loss_fn()

        cvae = self.to(device)
        optimizer = torch.optim.Adam(params=cvae.parameters(), lr=learning_rate)
        training_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

        vae_loss_li = []
        kl_loss_li = []
        reg_loss_li = []

        for epoch in range(epochs):
            i = 0
            for input, label in training_dataloader:
                input = input.to(device)
                label = label.to(device)
                output = cvae(input, label)

                # loss function to back-propagate on
                loss = vl_fn(input, output, cvae.z_dist)

                # back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                i += 1
                if i % 100 == 0:
                    # append vae loss
                    vae_loss_li.append(loss.item())

                    # calculate KL divergence loss
                    kl_loss_li.append(
                        kl_div_fn(cvae.z_dist)
                    )

                    # print(f"vl_loss: {reg(input, output)}")
                    reg_loss_li.append(reg(input, output))

            print(f"vl_loss: {sum(reg_loss_li)}")
            print("Finished epoch: ", epoch + 1)
        return (
            cvae.to('cpu'),
            vae_loss_li,
            kl_loss_li,
            reg_loss_li
        )

    def generate_data(self, n_samples=5, target_label=0) -> tensor:
        """
        Generates random data samples (of size n) from the latent space
        """
        device = next(self.parameters()).device
        input_sample = torch.randn(n_samples, self.dim_encoding).to(device)

        assert input_sample.shape[0] == n_samples

        with torch.no_grad():
            label = torch.zeros((n_samples, 10), device=device)
            label[:, target_label] = 1
            latent = torch.cat((input_sample, label), dim=1)
            generated_images = self.decoder(latent)
            return generated_images / 255.0
