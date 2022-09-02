import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

def plot_image_grid(output, img_size, n_row, steps=None, sample_path=None, save=False):

    out_grid = make_grid(output, normalize=True, nrow=n_row, scale_each=True, padding=int(0.125*img_size)).permute(1,2,0)

    fig, ax = plt.subplots(1,1 ,figsize=(100, 100))
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    plt.imshow(out_grid.cpu())
    plt.show()

    if save:
        plt.savefig(sample_path + '/steps_%d' %(steps))

def plot_curve_error(data_mean, data_std, x_label, y_label, title, filename=None):

    fig = plt.figure(figsize=(8, 6))
    plt.title(title)

    alpha = 0.3

    plt.plot(range(len(data_mean)), data_mean, '-', color = 'red')
    plt.fill_between(range(len(data_mean)), data_mean - data_std, data_mean + data_std, facecolor = 'blue', alpha = alpha)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.tight_layout()
    plt.show()

    if filename is not None:

        fig.savefig(filename)
        pass