import os
import sys

import numpy as np
import tensorflow as tf

import plot
import vae
from datasets import read_fishyfish 

IMG_WIDTH = 80
IMG_HEIGHT = 60

ARCHITECTURE = [IMG_WIDTH*IMG_HEIGHT, # 
                800, 500,# intermediate encoding
                #2] # latent space dims
                50]
# (and symmetrically back out again)

HYPERPARAMS = {
    "batch_size": 128,
    "learning_rate": 5E-4,
    "dropout": 0.9,
    "lambda_l2_reg": 1E-5,
    "nonlinearity": tf.nn.relu,
    "squashing": tf.nn.sigmoid
}

MAX_ITER = 20000#2**16
MAX_EPOCHS = np.inf

LOG_DIR = "./log"
METAGRAPH_DIR = "./out"
PLOTS_DIR = "./png"


def load_fishyfish():
    return read_fishyfish()

def all_plots(model, dataset):
    if model.architecture[-1] == 2: # only works for 2-D latent
        print("Plotting in latent space...")
        plot_all_in_latent(model, dataset)

        print("Exploring latent...")
        plot.exploreLatent(model, nx=20, ny=20, range_=(-4, 4), outdir=PLOTS_DIR)
        for n in (24, 30, 60, 100):
            plot.exploreLatent(model, nx=n, ny=n, ppf=True, outdir=PLOTS_DIR,
                               name="explore_ppf{}".format(n))

    #print("Interpolating...")
    #interpolate_digits(model, mnist)

    print("Plotting end-to-end reconstructions...")
    plot_all_end_to_end(model, dataset)

    #print("Morphing...")
    #morph_numbers(model, mnist, ns=[9,8,7,6,5,4,3,2,1,0])

    #print("Plotting 10 MNIST digits...")
    #for i in range(10):
    #    plot.justMNIST(get_mnist(i, mnist), name=str(i), outdir=PLOTS_DIR)

def plot_all_in_latent(model, mnist):
    names = ("train", "validation", "test")
    datasets = (mnist.train, mnist.validation, mnist.test)
    for name, dataset in zip(names, datasets):
        plot.plotInLatent(model, dataset.images, dataset.labels, name=name,
                          outdir=PLOTS_DIR)

#def interpolate_digits(model, mnist):
#    imgs, labels = mnist.train.next_batch(100)
#    idxs = np.random.randint(0, imgs.shape[0] - 1, 2)
#    mus, _ = model.encode(np.vstack(imgs[i] for i in idxs))
#    plot.interpolate(model, *mus, name="interpolate_{}->{}".format(
#        *(labels[i] for i in idxs)), outdir=PLOTS_DIR)

def plot_all_end_to_end(model, dataset):
    names = ("train", "validation", "test")
    datasets = (dataset.train, dataset.validation, dataset.test)
    for name, dataset in zip(names, datasets):
        x = dataset.next_batch(10)
        x_reconstructed = model.vae(x)
        plot.plotSubset(model, x, x_reconstructed, n=10, name=name,
                        outdir=PLOTS_DIR)

#def morph_numbers(model, mnist, ns=None, n_per_morph=10):
#    if not ns:
#        import random
#        ns = random.sample(range(10), 10) # non-in-place shuffle
#
#    xs = np.squeeze([get_mnist(n, mnist) for n in ns])
#    mus, _ = model.encode(xs)
#    plot.morph(model, mus, n_per_morph=n_per_morph, outdir=PLOTS_DIR,
#               name="morph_{}".format("".join(str(n) for n in ns)))

def main(to_reload=None):
    fishyfish = load_fishyfish()

    if to_reload: # restore
        v = vae.VAE(ARCHITECTURE, HYPERPARAMS, meta_graph=to_reload)
        print("Loaded!")

    else: # train
        v = vae.VAE(ARCHITECTURE, HYPERPARAMS, log_dir=LOG_DIR)
        v.train(fishyfish, max_iter=MAX_ITER, max_epochs=MAX_EPOCHS, cross_validate=False,
                verbose=True, save=True, outdir=METAGRAPH_DIR, plots_outdir=PLOTS_DIR,
                plot_latent_over_time=False)
        print("Trained!")

    all_plots(v, fishyfish)


if __name__ == "__main__":
    tf.reset_default_graph()

    for DIR in (LOG_DIR, METAGRAPH_DIR, PLOTS_DIR):
        try:
            os.mkdir(DIR)
        except(FileExistsError):
            pass

    try:
        to_reload = sys.argv[1]
        main(to_reload=to_reload)
    except(IndexError):
        main()
