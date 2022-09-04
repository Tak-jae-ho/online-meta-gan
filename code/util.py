import matplotlib.pyplot as plt
import os
from torchvision.utils import make_grid, save_image
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from torch.utils.data import SubsetRandomSampler

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

try:
	from torchvision.models.utils import load_state_dict_from_url
except ImportError:
	from torch.utils.model_zoo import load_url as load_state_dict_from_url

from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

from PIL import Image

try:
	from tqdm import tqdm
except ImportError:
	# If not tqdm is not available, provide a mock version of it
	def tqdm(x): return x


def plot_image_grid(output, img_size, n_row, epoch, sample_folder, result_dir=None):

    out_grid = make_grid(output, normalize=True, nrow=n_row, scale_each=True, padding=int(0.125*img_size)).permute(1,2,0)

    fig, ax = plt.subplots(1,1 ,figsize=(100, 100))
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    plt.imshow(out_grid.cpu())
    plt.show()

    if result_dir is not None:
        result_dir = os.path.join(result_dir, sample_folder)
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        fig.savefig(result_dir+'/epoch_%d' %(epoch))
        pass

def plot_curve_error(data_mean, data_std, x_label, y_label, title, file_name,result_dir=None):

    fig = plt.figure(figsize=(8, 6))
    plt.title(title)

    alpha = 0.4

    plt.plot(range(len(data_mean)), data_mean, '-', color = 'red')
    plt.fill_between(range(len(data_mean)), data_mean - data_std, data_mean + data_std, facecolor = 'blue', alpha = alpha)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.tight_layout()
    plt.show()

    if result_dir is not None:
        result_dir = os.path.join(result_dir, title)
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        fig.savefig(result_dir + '/' + file_name)
        pass

def plot_curve_error2(data1_mean, data1_std, data1_label, data2_mean, data2_std, data2_label, x_label, y_label, title, file_name, result_dir=None):
    
    fig = plt.figure(figsize=(8, 6))
    plt.title(title)

    alpha = 0.3

    plt.plot(range(len(data1_mean)), data1_mean, '-', color = 'blue', label = data1_label)
    plt.fill_between(range(len(data1_mean)), data1_mean - data1_std, data1_mean + data1_std, facecolor = 'blue', alpha = alpha)

    plt.plot(range(len(data2_mean)), data2_mean, '-', color = 'red', label = data2_label)
    plt.fill_between(range(len(data2_mean)), data2_mean - data2_std, data2_mean + data2_std, facecolor = 'red', alpha = alpha)

    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.tight_layout()
    plt.show()

    if result_dir is not None:
        result_dir = os.path.join(result_dir, title)
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        fig.savefig(result_dir + '/' + file_name)
		
        pass

def get_data_subsampler(dataset, data_per_class):
    assert data_per_class < len(dataset) / len(dataset.classes)
    subset_idx = []
    
    for label in dataset.targets.unique():
        class_idx = (dataset.targets == label).nonzero(as_tuple=True)
        rand_select = torch.randint(0, len(class_idx[0]), (data_per_class, ))
        class_idx = class_idx[0][rand_select]
        subset_idx.append(class_idx)
    
    subset_idx = torch.cat(subset_idx)
    
    return SubsetRandomSampler(indices=subset_idx)


##################################################### FOR FID_SCORE #####################################################

FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'


class InceptionV3(nn.Module):
	"""Pretrained InceptionV3 network returning feature maps"""

	# Index of default block of inception to return,
	# corresponds to output of final average pooling
	DEFAULT_BLOCK_INDEX = 3

	# Maps feature dimensionality to their output blocks indices
	BLOCK_INDEX_BY_DIM = {
		64: 0,   # First max pooling features
		192: 1,  # Second max pooling featurs
		768: 2,  # Pre-aux classifier features
		2048: 3  # Final average pooling features
	}

	def __init__(self,
				 output_blocks=[DEFAULT_BLOCK_INDEX],
				 resize_input=True,
				 normalize_input=True,
				 requires_grad=False,
				 use_fid_inception=True):
		"""Build pretrained InceptionV3

		Parameters
		----------
		output_blocks : list of int
			Indices of blocks to return features of. Possible values are:
				- 0: corresponds to output of first max pooling
				- 1: corresponds to output of second max pooling
				- 2: corresponds to output which is fed to aux classifier
				- 3: corresponds to output of final average pooling
		resize_input : bool
			If true, bilinearly resizes input to width and height 299 before
			feeding input to model. As the network without fully connected
			layers is fully convolutional, it should be able to handle inputs
			of arbitrary size, so resizing might not be strictly needed
		normalize_input : bool
			If true, scales the input from range (0, 1) to the range the
			pretrained Inception network expects, namely (-1, 1)
		requires_grad : bool
			If true, parameters of the model require gradients. Possibly useful
			for finetuning the network
		use_fid_inception : bool
			If true, uses the pretrained Inception model used in Tensorflow's
			FID implementation. If false, uses the pretrained Inception model
			available in torchvision. The FID Inception model has different
			weights and a slightly different structure from torchvision's
			Inception model. If you want to compute FID scores, you are
			strongly advised to set this parameter to true to get comparable
			results.
		"""
		super(InceptionV3, self).__init__()

		self.resize_input = resize_input
		self.normalize_input = normalize_input
		self.output_blocks = sorted(output_blocks)
		self.last_needed_block = max(output_blocks)

		assert self.last_needed_block <= 3, \
			'Last possible output block index is 3'

		self.blocks = nn.ModuleList()

		if use_fid_inception:
			inception = fid_inception_v3()
		else:
			inception = _inception_v3(pretrained=True)

		# Block 0: input to maxpool1
		block0 = [
			inception.Conv2d_1a_3x3,
			inception.Conv2d_2a_3x3,
			inception.Conv2d_2b_3x3,
			nn.MaxPool2d(kernel_size=3, stride=2)
		]
		self.blocks.append(nn.Sequential(*block0))

		# Block 1: maxpool1 to maxpool2
		if self.last_needed_block >= 1:
			block1 = [
				inception.Conv2d_3b_1x1,
				inception.Conv2d_4a_3x3,
				nn.MaxPool2d(kernel_size=3, stride=2)
			]
			self.blocks.append(nn.Sequential(*block1))

		# Block 2: maxpool2 to aux classifier
		if self.last_needed_block >= 2:
			block2 = [
				inception.Mixed_5b,
				inception.Mixed_5c,
				inception.Mixed_5d,
				inception.Mixed_6a,
				inception.Mixed_6b,
				inception.Mixed_6c,
				inception.Mixed_6d,
				inception.Mixed_6e,
			]
			self.blocks.append(nn.Sequential(*block2))

		# Block 3: aux classifier to final avgpool
		if self.last_needed_block >= 3:
			block3 = [
				inception.Mixed_7a,
				inception.Mixed_7b,
				inception.Mixed_7c,
				nn.AdaptiveAvgPool2d(output_size=(1, 1))
			]
			self.blocks.append(nn.Sequential(*block3))

		for param in self.parameters():
			param.requires_grad = requires_grad

	def forward(self, inp):
		"""Get Inception feature maps

		Parameters
		----------
		inp : torch.autograd.Variable
			Input tensor of shape Bx3xHxW. Values are expected to be in
			range (0, 1)

		Returns
		-------
		List of torch.autograd.Variable, corresponding to the selected output
		block, sorted ascending by index
		"""
		outp = []
		x = inp

		if self.resize_input:
			x = F.interpolate(x,
							  size=(299, 299),
							  mode='bilinear',
							  align_corners=False)

		if self.normalize_input:
			x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

		for idx, block in enumerate(self.blocks):
			x = block(x)
			if idx in self.output_blocks:
				outp.append(x)

			if idx == self.last_needed_block:
				break

		return outp


def _inception_v3(*args, **kwargs):
	"""Wraps `torchvision.models.inception_v3`

	Skips default weight inititialization if supported by torchvision version.
	See https://github.com/mseitzer/pytorch-fid/issues/28.
	"""
	try:
		version = tuple(map(int, torchvision.__version__.split('.')[:2]))
	except ValueError:
		# Just a caution against weird version strings
		version = (0,)

	if version >= (0, 6):
		kwargs['init_weights'] = False

	return torchvision.models.inception_v3(*args, **kwargs)


def fid_inception_v3():
	"""Build pretrained Inception model for FID computation

	The Inception model for FID computation uses a different set of weights
	and has a slightly different structure than torchvision's Inception.

	This method first constructs torchvision's Inception and then patches the
	necessary parts that are different in the FID Inception model.
	"""
	inception = _inception_v3(num_classes=1008,
							  aux_logits=False,
							  pretrained=False)
	inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
	inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
	inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
	inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
	inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
	inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
	inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
	inception.Mixed_7b = FIDInceptionE_1(1280)
	inception.Mixed_7c = FIDInceptionE_2(2048)

	state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
	inception.load_state_dict(state_dict)
	return inception


class FIDInceptionA(torchvision.models.inception.InceptionA):
	"""InceptionA block patched for FID computation"""
	def __init__(self, in_channels, pool_features):
		super(FIDInceptionA, self).__init__(in_channels, pool_features)

	def forward(self, x):
		branch1x1 = self.branch1x1(x)

		branch5x5 = self.branch5x5_1(x)
		branch5x5 = self.branch5x5_2(branch5x5)

		branch3x3dbl = self.branch3x3dbl_1(x)
		branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
		branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

		# Patch: Tensorflow's average pool does not use the padded zero's in
		# its average calculation
		branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
								   count_include_pad=False)
		branch_pool = self.branch_pool(branch_pool)

		outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
		return torch.cat(outputs, 1)


class FIDInceptionC(torchvision.models.inception.InceptionC):
	"""InceptionC block patched for FID computation"""
	def __init__(self, in_channels, channels_7x7):
		super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

	def forward(self, x):
		branch1x1 = self.branch1x1(x)

		branch7x7 = self.branch7x7_1(x)
		branch7x7 = self.branch7x7_2(branch7x7)
		branch7x7 = self.branch7x7_3(branch7x7)

		branch7x7dbl = self.branch7x7dbl_1(x)
		branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
		branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
		branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
		branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

		# Patch: Tensorflow's average pool does not use the padded zero's in
		# its average calculation
		branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
								   count_include_pad=False)
		branch_pool = self.branch_pool(branch_pool)

		outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
		return torch.cat(outputs, 1)


class FIDInceptionE_1(torchvision.models.inception.InceptionE):
	"""First InceptionE block patched for FID computation"""
	def __init__(self, in_channels):
		super(FIDInceptionE_1, self).__init__(in_channels)

	def forward(self, x):
		branch1x1 = self.branch1x1(x)

		branch3x3 = self.branch3x3_1(x)
		branch3x3 = [
			self.branch3x3_2a(branch3x3),
			self.branch3x3_2b(branch3x3),
		]
		branch3x3 = torch.cat(branch3x3, 1)

		branch3x3dbl = self.branch3x3dbl_1(x)
		branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
		branch3x3dbl = [
			self.branch3x3dbl_3a(branch3x3dbl),
			self.branch3x3dbl_3b(branch3x3dbl),
		]
		branch3x3dbl = torch.cat(branch3x3dbl, 1)

		# Patch: Tensorflow's average pool does not use the padded zero's in
		# its average calculation
		branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
								   count_include_pad=False)
		branch_pool = self.branch_pool(branch_pool)

		outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
		return torch.cat(outputs, 1)


class FIDInceptionE_2(torchvision.models.inception.InceptionE):
	"""Second InceptionE block patched for FID computation"""
	def __init__(self, in_channels):
		super(FIDInceptionE_2, self).__init__(in_channels)

	def forward(self, x):
		branch1x1 = self.branch1x1(x)

		branch3x3 = self.branch3x3_1(x)
		branch3x3 = [
			self.branch3x3_2a(branch3x3),
			self.branch3x3_2b(branch3x3),
		]
		branch3x3 = torch.cat(branch3x3, 1)

		branch3x3dbl = self.branch3x3dbl_1(x)
		branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
		branch3x3dbl = [
			self.branch3x3dbl_3a(branch3x3dbl),
			self.branch3x3dbl_3b(branch3x3dbl),
		]
		branch3x3dbl = torch.cat(branch3x3dbl, 1)

		# Patch: The FID Inception model uses max pooling instead of average
		# pooling. This is likely an error in this specific Inception
		# implementation, as other Inception models use average pooling here
		# (which matches the description in the paper).
		branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
		branch_pool = self.branch_pool(branch_pool)

		outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
		return torch.cat(outputs, 1)

cuda = True if torch.cuda.is_available() else False



parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('path', type=str, nargs=2,
					help=('Path to the generated images or '
						  'to .npz statistic files'))
parser.add_argument('--batch-size', type=int, default=50,
					help='Batch size to use')
parser.add_argument('--dims', type=int, default=2048,
					choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
					help=('Dimensionality of Inception features to use. '
						  'By default, uses pool3 features'))
parser.add_argument('-c', '--gpu', default='', type=str,
					help='GPU to use (leave blank for CPU only)')




def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
	"""Numpy implementation of the Frechet Distance.
	The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
	and X_2 ~ N(mu_2, C_2) is
			d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

	Stable version by Dougal J. Sutherland.

	Params:
	-- mu1   : Numpy array containing the activations of a layer of the
			   inception net (like returned by the function 'get_predictions')
			   for generated samples.
	-- mu2   : The sample mean over activations, precalculated on an
			   representative data set.
	-- sigma1: The covariance matrix over activations for generated samples.
	-- sigma2: The covariance matrix over activations, precalculated on an
			   representative data set.

	Returns:
	--   : The Frechet Distance.
	"""

	mu1 = np.atleast_1d(mu1)
	mu2 = np.atleast_1d(mu2)

	sigma1 = np.atleast_2d(sigma1)
	sigma2 = np.atleast_2d(sigma2)

	assert mu1.shape == mu2.shape, \
		'Training and test mean vectors have different lengths'
	assert sigma1.shape == sigma2.shape, \
		'Training and test covariances have different dimensions'

	diff = mu1 - mu2

	# Product might be almost singular
	covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
	if not np.isfinite(covmean).all():
		msg = ('fid calculation produces singular product; '
			   'adding %s to diagonal of cov estimates') % eps
		print(msg)
		offset = np.eye(sigma1.shape[0]) * eps
		covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

	# Numerical error might give slight imaginary component
	if np.iscomplexobj(covmean):
		if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
			m = np.max(np.abs(covmean.imag))
			raise ValueError('Imaginary component {}'.format(m))
		covmean = covmean.real

	tr_covmean = np.trace(covmean)

	return (diff.dot(diff) + np.trace(sigma1) +
			np.trace(sigma2) - 2 * tr_covmean)






def get_activations(batch, model, batch_size=50, dims=2048):

	model.eval()

	if batch_size > batch.shape[0]:
		print(('Warning: batch size is bigger than the data size. '
			   'Setting batch size to data size'))
		batch_size = batch.shape[0]

	pred_arr = np.empty((batch.shape[0], dims))

	for i in tqdm(range(0, batch.shape[0], batch_size)): #tqdm: progressive bar

		start = i
		end = i + batch_size

		data = batch[start:end]

		if cuda:
			data=data.type(torch.cuda.FloatTensor)

		#pred = model(batch)[0]
		pred = model(data)[0]

		# If model output is not scalar, apply global spatial average pooling.
		# This happens if you choose a dimensionality not equal 2048.
		if pred.size(2) != 1 or pred.size(3) != 1:
			pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

		pred_arr[start:end] = pred.cpu().data.numpy().reshape(pred.size(0), -1)

	return pred_arr

def calculate_activation_statistics(batch, model, batch_size=50, dims=2048):

	act = get_activations(batch, model, batch_size, dims)
	mu = np.mean(act, axis=0)
	sigma = np.cov(act, rowvar=False)
	return mu, sigma


def _compute_statistics_of_batch(batch, model, batch_size, dims):
	
	m, s = calculate_activation_statistics(batch, model, batch_size, dims)

	return m, s


#def calculate_fid_given_paths(paths, batch_size, cuda, dims):
def calculate_fid_given_batches(batch1, batch2, batch_size, dims=2048):


	# gray image
	if batch1.shape[1] < 3:  batch1 = batch1.expand(-1,3,-1,-1) 
	if batch2.shape[1] < 3:  batch2 = batch2.expand(-1,3,-1,-1) 

	block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

	model = InceptionV3([block_idx])

	if cuda:
		model.cuda()


	m1, s1 = _compute_statistics_of_batch(batch1, model, batch_size, dims)
	m2, s2 = _compute_statistics_of_batch(batch2, model, batch_size, dims)
	fid_value = calculate_frechet_distance(m1, s1, m2, s2)

	return fid_value

def make_hidden(batch_size, latent_size):
    device = torch.device('cuda:0')
    z = torch.normal(0, 1, size=(batch_size, latent_size, 1, 1))
    z /= torch.sqrt(torch.sum(z * z, dim=1, keepdims=True) / latent_size + 1e-8)
    
    return z.to(device)


if __name__ == '__main__':
	args = parser.parse_args()
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

	fid_value = calculate_fid_given_paths(args.path, args.batch_size, args.gpu != '', args.dims)
	print('FID: ', fid_value)