# Generative Adversarial Networks
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from util import plot_image_grid, plot_curve_error, plot_curve_error2, calculate_fid_given_batches, make_hidden, get_data_subsampler
from models import Discriminator, Generator
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('--n_epoch', default=500, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--batch_size_fid', default=16, type=int)
parser.add_argument('--learning_rate_discriminator', default=0.001, type=float)
parser.add_argument('--learning_rate_generator', default=0.002, type=float)
parser.add_argument('--dim_latent', default=32, type=int)
parser.add_argument('--dim_channel', default=1, type=int)
parser.add_argument('--eval_freq', default=5, type=int)
parser.add_argument('--result_path', default='/nas/users/jaeho/online-meta-gan/result', type=str, help='save results')
parser.add_argument('--sample_folder', default='sample_meta', type=str, help='save results')
parser.add_argument('--Loss_Curve', default='Loss_Curve_meta', type=str, help='Loss Curve image file&folder name')
parser.add_argument('--Prediction_Curve', default='Prediction_Curve_meta', type=str, help='Prediction Curve image file&folder name')
parser.add_argument('--FID_score_Curve', default='FID_score_Curve_meta', type=str, help='FID score Curve image file&folder name')

###################### FOR META-TRAINING ######################
# MNIST : 6000 imgs per digits classes
parser.add_argument('--data_per_class', default=100, type=int)
parser.add_argument('--lambda_', default=0.1, type=float)
parser.add_argument('--PATH_discriminator_theta', default='./discriminator_theta/PATH_discriminator_theta.pt', type=str)


args = parser.parse_args()

n_epoch = args.n_epoch
batch_size = args.batch_size
batch_size_fid = args.batch_size_fid
learning_rate_discriminator = args.learning_rate_discriminator
learning_rate_generator = args.learning_rate_generator
dim_latent = args.dim_latent
dim_channel = args.dim_channel
eval_freq = args.eval_freq
result_path = args.result_path
Loss_Curve  = args.Loss_Curve
Prediction_Curve = args.Prediction_Curve
FID_score_Curve = args.FID_score_Curve
sample_folder = args.sample_folder
data_per_class = args.data_per_class
lambda_ = args.lambda_
PATH_discriminator_theta = args.PATH_discriminator_theta

# Load Data
train_dataset = MNIST('/nas/dataset/MNIST', train=True, download=True, 
                                            transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Resize((32, 32)),
                                            ]))

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
subsampler = get_data_subsampler(train_dataset, data_per_class)
train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, sampler=subsampler)
fid_loader = DataLoader(train_dataset, batch_size=batch_size_fid, shuffle=True, drop_last=True)

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
fids = np.zeros(n_epoch)

for epoch in tqdm(range(n_epoch)):

    loss_discriminator_batch = []
    loss_generator_batch = []
    prediction_real_batch = []
    prediction_fake_batch = []

    for x, y in tqdm(train_loader):

        # update discriminator: maximize log(D(x)) + log(1 - D(G(z)))
        optimizer_discriminator.zero_grad()
        
        temp_weights_theta = [w.clone() for w in list(discriminator.parameters())]
        
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

        temp_weights_phi = [w.clone() for w in list(discriminator.parameters())]

        grad = [theta - phi for theta, phi in zip(temp_weights_theta, temp_weights_phi)]
        params_discriminator = discriminator.state_dict()

        temp_weights_theta = [w - lambda_ * grad if grad is not None else w for w, grad in zip(temp_weights_theta, grad)]

        for n, key in enumerate(list(params_discriminator.keys())):
            params_discriminator[key] = temp_weights_theta[n]
        
        torch.save(params_discriminator, PATH_discriminator_theta)
        discriminator.load_state_dict(torch.load(PATH_discriminator_theta))

        # update generator: maximize log(D(G(z)))
        optimizer_generator.zero_grad()
        fake = generator(noise)
        prediction_fake = discriminator(fake)
        label_real = torch.ones_like(prediction_fake)
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

    if (epoch+1) % eval_freq == 0:
        # Plot Result
        print("------------------Save Samples & fid----------------------")
        generator.eval()
        discriminator.eval()
        
        noise = torch.randn(100, dim_latent, 1, 1, device=device)
        generated_images = generator(noise)

        out_grid = plot_image_grid(generated_images, 32, 10, epoch+1, sample_folder,result_path)

        real_batch = []
        fake_batch = []

        for i_fid, (real, _) in enumerate(fid_loader):
            
            fid_latent = make_hidden(batch_size, dim_latent)
            fake = generator(fid_latent)
            real_batch.append(real.type(torch.FloatTensor))
            fake_batch.append(fake.type(torch.FloatTensor))

            if i_fid * batch_size > 2500:
                break

        real_batch = torch.cat(real_batch, dim=0)
        fake_batch = torch.cat(fake_batch, dim=0)

        score = calculate_fid_given_batches(real_batch, fake_batch, batch_size=batch_size_fid)
        for k in range(eval_freq):
            
            fids[epoch-k] = score
        
        print("fid: %.5f" % score)

        # save fid scores via iteration
        plt.figure()
        plt.plot(fids, '-', color='red', label='FID')
        plt.xlabel('iteration')
        plt.ylabel('FID')
        plt.legend()
        plt.tight_layout()

        fid_dir = os.path.join(result_path, 'fid_iteration')
        if not os.path.exists(fid_dir):
            os.mkdir(fid_dir)
        plt.savefig(fid_dir + '/' + FID_score_Curve)
        plt.close('all')

        discriminator.train()
        generator.train()
    
    # save losses & predicition values
    loss_discriminator_mean[epoch] = np.mean(loss_discriminator_batch)
    loss_discriminator_std[epoch] = np.std(loss_discriminator_batch)
    loss_generator_mean[epoch] = np.mean(loss_generator_batch)
    loss_generator_std[epoch] = np.std(loss_generator_batch)
    prediction_real_mean[epoch] = np.mean(prediction_real_batch)
    prediction_real_std[epoch] = np.std(prediction_real_batch)
    prediction_fake_mean[epoch] = np.mean(prediction_fake_batch)
    prediction_fake_std[epoch] = np.std(prediction_fake_batch)

    print('epoch: {}/{} loss_discriminator: {:.6f} loss_generator: {:.6f} prediction_real: {:.6f} prediction_fake: {:.6f}' .format(epoch + 1, n_epoch, loss_discriminator_mean[epoch], loss_generator_mean[epoch], prediction_real_mean[epoch], prediction_fake_mean[epoch]))
    plot_curve_error2(loss_discriminator_mean, loss_discriminator_std, 'Discriminator', loss_generator_mean, loss_generator_std, 'Generator', 'epoch', 'loss', 'Loss Curve', Loss_Curve, result_path)
    plot_curve_error2(prediction_real_mean, prediction_real_std, 'D(x)', prediction_fake_mean, prediction_fake_std, 'D(G(z))', 'epoch', 'Output', 'Prediction Curve', Prediction_Curve, result_path)