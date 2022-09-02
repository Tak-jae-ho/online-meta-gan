import matplotlib.pyplot as plt
import os
from torchvision.utils import make_grid, save_image

def plot_image_grid(output, img_size, n_row, epoch, result_dir=None):

    out_grid = make_grid(output, normalize=True, nrow=n_row, scale_each=True, padding=int(0.125*img_size)).permute(1,2,0)

    fig, ax = plt.subplots(1,1 ,figsize=(100, 100))
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    plt.imshow(out_grid.cpu())
    plt.show()

    if result_dir is not None:
        result_dir = os.path.join(result_dir, 'samples')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        fig.savefig(result_dir+'/epoch_%d' %(epoch))
        pass

def plot_curve_error(data_mean, data_std, x_label, y_label, title, result_dir=None):

    fig = plt.figure(figsize=(8, 6))
    plt.title(title)

    alpha = 0.3

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
        fig.savefig(result_dir)
        pass

def plot_curve_error2(data1_mean, data1_std, data1_label, data2_mean, data2_std, data2_label, x_label, y_label, title, result_dir=None):
    
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
        fig.savefig(result_dir)
        pass
