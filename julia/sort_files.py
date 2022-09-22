#!/usr/bin/python
import os
import shutil
import re
import glob

mtx = glob.glob("mtx-*.mtx")
mtx.sort()

for i, m in enumerate(mtx):
    root, ext = os.path.splitext(m)
    root = os.path.basename(root)
    match = re.search(r'mtx-(\d+)-\d+-decoupled-\d+-\d+-(\d+)-(\d\.\d+e[\+-]\d+)-\d+', root)

    N_fine = match.group(1)
    n_holes = match.group(2)
    eps = match.group(3)
    root_dest = os.path.join("decoupled_eps_" + eps, N_fine)

    try:
        os.mkdir(root_dest)
    except FileExistsError:
        pass

    root_dest_n = os.path.join(root_dest, n_holes)
    try:
        os.mkdir(root_dest_n)
    except FileExistsError:
        pass

    paths = [
        root + ext,
        root + '-rhs1.json',
        root + '-rhs2.json',
        root + '-rhs3.json',
    ]
    for p in paths:
        shutil.move(p, root_dest_n)
