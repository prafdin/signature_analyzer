import os
import re

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
