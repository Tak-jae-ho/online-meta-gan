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