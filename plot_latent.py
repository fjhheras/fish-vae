import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


import plot
from tsne import plot_tsne
import vae
from datasets import read_fishyfish as load_fishyfish 

IMG_WIDTH = 80
IMG_HEIGHT = 60


ARCHITECTURE = [IMG_WIDTH*IMG_HEIGHT, # 
                500, 500, 500,# intermediate encoding
                #2] # latent space dims
                20]
# (and symmetrically back out again)

HYPERPARAMS = {
    "batch_size": 128,
    "learning_rate": 4E-4,
    "dropout": .95,
    "lambda_l2_reg": 0,#1E-5,
    "nonlinearity": tf.nn.relu,
    "squashing": tf.nn.sigmoid,
    "beta": 10.0
    }


def latent_plots(model):
    plot.latent_one_d(model,10) 
    if model.architecture[-1] == 2:
        plot.exploreLatent(model,nx=15,ny=11,range_=(-3,3))

def main(to_reload):
    fishyfish = load_fishyfish()
    v = vae.VAE(ARCHITECTURE, HYPERPARAMS, meta_graph=to_reload)
    print("Loaded!")
    latent_plots(v)
    #plot_tsne(fishyfish.train._images, vae = v) 
    plt.show()

if __name__ == "__main__":
    tf.reset_default_graph()

    try:
        to_reload = sys.argv[1]
        main(to_reload=to_reload)
    except(IndexError):
        print("No file given")
