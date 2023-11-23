'''
Created on 20.11.2023

@author: Michael Diedenhofen
Max Planck Institute for Metabolism Research, Cologne

Read NIfTI files in input folder, rotate data and save in output folder.
Divide voxel dimensions by 10 (to get scale factor 1.0).
Output filename extension is .1.1.nii.gz
'''

from __future__ import print_function

import os
import sys

import numpy as np
import nibabel as nib

def read_nifti(path_in):
    image = nib.load(path_in)
    print("Input file:", image.get_filename())

    return (np.asanyarray(image.dataobj), image.header, image.affine)

def save_nifti(path_out, data, header, affine=None):
    image = nib.Nifti1Image(data, affine, header=header)
    image.set_data_dtype(data.dtype)
    image.to_filename(path_out)
    print("Output file:", image.get_filename())

def get_list(dir_path, ext='_merge'):
    res_list = []
    for item in os.listdir(dir_path):
        path_item = os.path.join(dir_path, item)
        if os.path.isfile(path_item) and item.endswith((''.join([ext, '.nii']), ''.join([ext, '.nii.gz']))):
            res_list.append(item)

    return res_list

def rotate_data(path_in, path_out, scalefactor=0.1, tr=3.5):
    # Read NIfTI input file
    data, header, _ = read_nifti(path_in)

    # scale voxel dimensions
    pixdim = header['pixdim']
    pixdim[1:4] = [(scalefactor * x) for x in pixdim[1:4]]
    pixdim[4] = tr
    header['pixdim'] = pixdim

    # Rotate around y-axis (flip x and z)
    data_f32 = data.astype(np.float32)
    data_rot = np.flip(data_f32, axis=(0, 2))

    # Save NIfTI output file
    save_nifti(path_out, data_rot, header, affine=None)

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Read NIfTI files in input folder, rotate data and save in output folder')
    parser.add_argument('dir_path_in', help='Input directory path')
    parser.add_argument('dir_path_out', help='Output directory path')
    parser.add_argument('-x', '--ext', default='_merge', help='File extension (without .nii or .nii.gz)')
    args = parser.parse_args()

    dir_path_in = os.path.abspath(args.dir_path_in)
    if not os.path.isdir(dir_path_in):
        sys.exit("Error: '%s' is not an existing directory." % (dir_path_in,))

    dir_path_out = os.path.abspath(args.dir_path_out)
    if not os.path.isdir(dir_path_out):
        sys.exit("Error: '%s' is not an existing directory." % (dir_path_out,))

    try:
        is_samefile = os.path.samefile(dir_path_in, dir_path_out)
    except OSError:
        is_samefile = (os.path.normcase(dir_path_in) == os.path.normcase(dir_path_out))

    if is_samefile:
        sys.exit("Error: '%s' is input and output directory." % (dir_path_in,))

    ext_out = '.1.1.nii.gz'
    res_list = get_list(dir_path_in, ext=args.ext)
    for index, item in enumerate(res_list):
        print("%d. Rotate:" % (index + 1,), item)
        path_in = os.path.join(dir_path_in, item)
        if item.endswith('.nii'):
            path_out = os.path.join(dir_path_out, ''.join([item[:-4], ext_out]))
        elif item.endswith('.nii.gz'):
            path_out = os.path.join(dir_path_out, ''.join([item[:-7], ext_out]))
        else:
            path_out = os.path.join(dir_path_out, item)
        if path_out.endswith(ext_out):
            rotate_data(path_in, path_out, scalefactor=0.1)

if __name__ == '__main__':
    main()
