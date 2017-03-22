from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from scipy import stats
from datasets import read_fishyfish

def imscatter(x, y, image, ax=None, zoom=1):
    artists = []
    n = x.shape[0]
    for i in np.arange(n):
        im = OffsetImage(image[i].reshape(60,80), zoom=zoom)
        ab = AnnotationBbox(im, (x[i], y[i]), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

def kernel_density_estimate(x,y, ax=None):
    X, Y = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)

    ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
          extent=[x.min(), x.max(), y.min(), y.max()])
    ax.plot(x, y, 'k.', markersize=2)
    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([y.min(), y.max()])
    plt.show()

def plot_tsne(images, vae = None):
    print("Plotting, and maybe t-sning image array of dims ", images.shape)
    
    model = TSNE(n_components=2, random_state=0)

    if vae == None:
        XYtsne = np.array(model.fit_transform(images))
    else:
        mean, log_sigma = vae.encode(images)
        if mean.shape[1] > 2:
            XYtsne = np.array(model.fit_transform(mean))
        else:
            print ("warning, dim 2")
            XYtsne = mean

    x = XYtsne[:,0]
    y = XYtsne[:,1]

    fig, ax = plt.subplots()
    imscatter(x, y, images, zoom=1.0, ax=ax)
    
    fig2, ax = plt.subplots()
    kernel_density_estimate(x,y,ax=ax) 
    plt.show()


if __name__ == "__main__":
    fishyfish = read_fishyfish()
    X = fishyfish.validation._images #next_batch(10)
 
#    try:
#        to_reload = sys.argv[1]
#        plot_mnist_latent(X,to_reload=to_reload)
#    except(IndexError):
#        print("No file given")
    plot_tsne(X)


#def plot_mnist_latent():
#    v = vae.VAE(ARCHITECTURE, HYPERPARAMS, meta_graph=to_reload)
#    print("Loaded!")


