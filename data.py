import tensorflow as tf
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.set_printoptions(precision=4)

def main():
    dataset = tf.data.Dataset.from_tensor_slices([5, 4, 6, 23, 67, 7, 8])

    print("---- Dataset iteration ----")
    for elem in dataset:
        print(elem.numpy())

    it = iter(dataset)
    print(next(it).numpy())
    print("---------------------------")

    # print(dataset.reduce(0, lamba.state, value=state + value).numpy())

    print("---- Element spec ----")
    dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10]))
    print(dataset1.element_spec)
    print("----------------------")
main()
