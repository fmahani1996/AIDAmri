'''
Created on 04.10.2023

Author:
Michael Diedenhofen
Max Planck Institute for Metabolism Research, Cologne
'''

from __future__ import print_function

VERSION = 'pv_example.py v 1.0.0 20231004'

import os
import sys

import numpy as np

import pv_conv2Nifti as pvc

def main():
    # processed data folder
    #proc_folder = r'/Volumes/ivnmr_scratch/michaeld/pv_conv/processed_data'
    proc_folder = r'/Volumes/Expansion/2023_Mahani_SFC/inputs/mri/065_Claudia_Stemcells_nu/proc_data/w0'

    # raw data folder
    #raw_folder = r'/Volumes/ivnmr_scratch/michaeld/065_Claudia_Stemcells_nu/raw_data'
    raw_folder = r'/Volumes/Expansion/2023_Mahani_SFC/inputs/mri/065_Claudia_Stemcells_nu/raw_data'

    # study name
    study = 'CH_30232_w0.yD1'
    #study = 'CH_30240_w0.yA1'

    # processed images number
    procno = '1'

    input_folder = os.path.join(raw_folder, study)
    #proc_folder = input_folder

    if not os.path.isdir(input_folder):
        sys.exit("Error: '%s' is not an existing directory." % (input_folder,))
    else:
        print("input_folder:", input_folder)

    if not os.path.isdir(proc_folder):
        sys.exit("Error: '%s' is not an existing directory." % (proc_folder,))
    else:
        print("proc_folder:", proc_folder)

    list_input = os.listdir(input_folder)
    listOfScans = [s for s in list_input if s.isdigit()]

    if len(listOfScans) == 0:
        sys.exit("Error: '%s' contains no numbered scans." % (raw_folder,))

    print(study)
    print("Start to process %d scans..." % (len(listOfScans),))

    img = None
    for expno in np.sort(listOfScans):
        path = os.path.join(input_folder, expno, 'pdata', procno)
        if not os.path.isdir(path):
            sys.exit("Error: '%s' is not an existing directory." % (path,))

        if os.path.exists(os.path.join(path, '2dseq')):
            img = pvc.Bruker2Nifti(study, expno, procno, raw_folder, proc_folder, ftype='NIFTI_GZ')
            img.read_2dseq(map_raw=True, pv6=False)
            resPath = img.save_nifti()
            if resPath is None:
                continue
            if 'VisuAcqEchoTime' in img.visu_pars:
                echoTime = img.visu_pars['VisuAcqEchoTime']
                echoTime = np.fromstring(echoTime, dtype=float, sep=' ')
                #if len(echoTime) > 3:
                #    mapT2.getT2mapping(resPath,args.model,args.upLim,args.snrLim,args.snrMethod,echoTime)
        else:
            print("The following file does not exist, it will be skipped:")
            print(os.path.join(path, '2dseq'))
            continue

    if resPath is not None:
        pathlog = os.path.dirname(os.path.dirname(resPath))
        pathlog = os.path.join(pathlog, 'data.log')
        logfile = open(pathlog, 'w')
        if img.subject.get('coilname') is not None:
            logfile.write(img.subject['coilname'])
        logfile.close()

if __name__ == '__main__':
    main()
