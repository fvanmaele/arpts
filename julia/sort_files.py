#!/usr/bin/python3
import os
import shutil
import re
import glob

mtx = glob.glob("mtx-*.mtx")
mtx.sort()
mtx_json = glob.glob("mtx-*.json")
mtx_json.sort()

for i, m in enumerate(mtx + mtx_json):
    root, ext = os.path.splitext(m)
    root = os.path.basename(root)
    match = re.search(r'mtx-(\d+)-\d+-decoupled-\d+-\d+-(\d+)-(\d\.\d+e[\+-]\d+)-\d+(-rhs\d)?', root)

    mtx_id = match.group(1)
    n_holes = match.group(2)
    eps = match.group(3)

    try:
        os.mkdir("decoupled_eps_" + eps)
    except FileExistsError:
        pass

    root_dest = os.path.join("decoupled_eps_" + eps, mtx_id)
    try:
        os.mkdir(root_dest)
    except FileExistsError:
        pass

    root_dest_n = os.path.join(root_dest, n_holes)
    try:
        os.mkdir(root_dest_n)
    except FileExistsError:
        pass

    shutil.move(root + ext, os.path.join(root_dest_n, root + ext))
    # try:
    #     shutil.move(root + ext, root_dest_n)
    # except shutil.Error:
    #     pass  # file exists
