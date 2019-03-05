# Script to move a subset of files to another directory
# Choose test set data from a large set of train Data

import numpy as np
import os


def get_imlist(path):
  """  Returns a list of filenames for
    all jpg images in a directory. """

  return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

def select_testImgs(folder):
    im_1s = get_imlist(folder+'train/1')
    numtest = 0
    for im in im_1s:
        myrand = np.random.rand()
        if (myrand < 0.2):
            os.rename(im, folder+'test/1/'+os.path.basename(im))
            numtest = numtest + 1
        else:
            continue


    return numtest


trainfolder = '/Users/nkolli/Documents/Python/Cats_vs_Dogs/'
numtest = select_testImgs(trainfolder)
print(numtest)
