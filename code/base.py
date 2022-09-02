# Generative Adversarial Networks
import numpy as np
import matplotlib.pyplot as plt
import argparse
from util import plot_image_grid, plot_curve_error, plot_curve_error2
from models import Discriminator, Generator
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

# Path
parser.add_argument('--result_path', default='/nas/users/jaeho/online-meta-gan/result', type=str, help='save results')

# Load Data
train_dataset = MNIST('/nas/dataset/MNIST', train=True, download=True, 
                                            transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Resize((32, 32)),
                                            ]))

# Hyper-parameters
n_epoch = 100
batch_size = 128
learning_rate_discriminator = 0.001
learning_rate_generator = 0.001
dim_latent = 32
dim_channel = 1

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare for Training

# Networks
discriminator = Discriminator(in_channel=dim_channel, out_channel=1, dim_feature=32).to(device)
generator     = Generator(in_channel=dim_latent, out_channel=1, dim_feature=32).to(device)

print()
print('<Architecture of Generator Network>')
summary(generator, input_size=(dim_latent, 1, 1))
print()
print('<Architecture of Discriminator Network>')
summary(discriminator, input_size=(dim_channel, 32, 32))
print()

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Loss function
criterion = nn.BCELoss()

# Optimizers
optimizer_generator = torch.optim.SGD(generator.parameters(), lr=learning_rate_generator)
optimizer_discriminator = torch.optim.SGD(discriminator.parameters(), lr=learning_rate_discriminator)

# lists for loss and prediction values to plot curves
loss_discriminator_mean = np.zeros(n_epoch)
loss_discriminator_std = np.zeros(n_epoch)
loss_generator_mean = np.zeros(n_epoch)
loss_generator_std = np.zeros(n_epoch)

prediction_real_mean = np.zeros(n_epoch) # D(x)
prediction_real_std = np.zeros(n_epoch) # D(x)
prediction_fake_mean = np.zeros(n_epoch) # D(G(z))
prediction_fake_std = np.zeros(n_epoch) # D(G(z))

# Train
discriminator.train()
generator.train()

for epoch in tqdm(range(n_epoch)):

    loss_discriminator_batch = []
    loss_generator_batch = []
    prediction_real_batch = []
    prediction_fake_batch = []

    for x, y in tqdm(train_loader):

        # update discriminator: maximize log(D(x)) + log(1 - D(G(z)))
        optimizer_discriminator.zero_grad()
        # real images
        real = x.to(device)
        prediction_real = discriminator(real)
        label_real = torch.ones_like(prediction_real, device=device)
        real_loss = criterion(prediction_real, label_real)
        # fake image
        noise = torch.randn(batch_size, dim_latent, 1, 1, device=device)
        fake = generator(noise)
        prediction_fake = discriminator(fake)
        label_fake = torch.zeros_like(prediction_fake)
        fake_loss = criterion(prediction_fake, label_fake)

        loss_discriminator = (real_loss + fake_loss) / 2.0
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # update generator: maximize log(D(G(z)))
        optimizer_generator.zero_grad()
        fake = generator(noise)
        prediction_fake = discriminator(fake)
        label_real = torch.ones_like(prediction_fake, label_real)
        loss_generator = criterion(prediction_fake, label_real)
        loss_generator.backward()
        optimizer_discriminator.step()

        # update generator: maximize log(D(G(z)))
        optimizer_generator.zero_grad()
        fake = generator(noise)
        prediction_fake = discriminator(fake)
        label_real = torch.ones_like(prediction_fake)
        loss_generator = criterion(prediction_fake, label_real)
        loss_generator.backward()
        optimizer_generator.step()

        # save losses & prediction values for minibatch
        loss_discriminator_batch.append(loss_discriminator.item())
        loss_generator_batch.append(loss_generator.item())
        prediction_real_batch.append(prediction_real.mean().item())
        prediction_fake_batch.append(prediction_fake.mean().item())

    # save losses & predicition values
    loss_discriminator_mean[epoch] = np.mean(loss_discriminator_batch)
    loss_discriminator_std[epoch] = np.std(loss_discriminator_batch)
    loss_generator_mean[epoch] = np.mean(loss_generator_batch)
    loss_generator_std[epoch] = np.std(loss_generator_batch)
    prediction_real_mean[epoch] = np.mean(prediction_real_batch)
    prediction_real_std[epoch] = np.std(prediction_real_batch)
    prediction_fake_mean[epoch] = np.mean(prediciton_fake_batch)
    prediction_fake_std[epoch] = np.std(prediciton_fake_batch)

    print('epoch: {}/{} loss_discriminator: {:.6f} loss_generator: {:.6f} prediction_real: {:.6f} prediction_fake: {:.6f}' .format(epoch + 1, n_epoch, loss_discriminator_mean[epoch], loss_generator_mean[epoch], prediction_real_mean[epoch], prediction_fake_mean[epoch]))

# Plot Result
generator.eval()
discriminator.eval()

plot_curve_error2(loss_discriminator_mean, loss_discriminator_std, 'Discriminator', loss_generator_mean, loss_generator_std, 'Generator', 'epoch', 'loss', 'Loss Curve', ars.result_path)
