import os
import re

import numpy as np
from matplotlib import pyplot as plt

IMAGE_EXTENSIONS = [".png", ".jpeg"]

def find_files_by_pattern_recursive(find_root_dir, pattern):
    all_files = []
    for root, dirs, files in os.walk(find_root_dir):
        for file in files:
            if re.match(pattern, file):
                all_files.append(os.path.join(root, file))
    return all_files

def get_random_color(salt):
    colors = ['#ff80ed', '#00ffff', '#ffd700', '#bada55', '#ff0000', '#f08080', '#6897bb', '#088da5']
    return colors[salt % len(colors)]

def draw_knn_accuracy_plots(k_range, scores_EUCL_list, scores_BHAT_list, scores_KVDIV_list):
    labels = [f"k={k}" for k in k_range]

    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots()

    ax.bar(x - 3 * width / 2, scores_EUCL_list, width, label='EUCLIDEAN')
    ax.bar(x - width / 2, scores_BHAT_list, width, label='BHATTACHARYYA')
    ax.bar(x + width / 2, scores_KVDIV_list, width, label='KV_DIVERGENCE')
    # ax.bar(x + 3 * width / 2, scores_COREL_list, width, label='CORRELATION')

    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by metric and k value')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 1)
