from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

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


if __name__ == "__main__":

    fishyfish = read_fishyfish()

    X = fishyfish.validation._images #next_batch(10)
    model = TSNE(n_components=2, random_state=0)
    Y = np.array(model.fit_transform(X))

    fig, ax = plt.subplots()
    imscatter(Y[:,0], Y[:,1], X, zoom=1, ax=ax)
    plt.show()


