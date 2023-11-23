"""
Created on 10/08/2017

@author: Niklas Pallast
Neuroimaging & Neuroengineering
Department of Neurology
University Hospital Cologne

"""

from __future__ import print_function

import os
import re
import sys

import numpy as np
import nibabel as nib
import nibabel.nifti1 as nii

import pv_parseBruker_md_np as pB
#import P2_IDLt2_mapping as mapT2 # michaeld 20231004
import pv_parser as par

class Bruker2Nifti:
    def __init__(self, study, expno, procno, rawfolder, procfolder, ftype='NIFTI_GZ'):
        self.study = study
        self.expno = str(expno)
        self.procno = str(procno)
        self.rawfolder = rawfolder
        self.procfolder = procfolder
        self.ftype = ftype

    def __check_params_md(self, params_name, labels):
        misses = [label for label in labels if label not in getattr(self, params_name)]
        if len(misses) > 0:
            sys.exit("Missing labels in %s: %s" % (params_name, str(misses),))

    def __extract_jcamp_strings_md(self, string, get_all=True):
        if string is None:
            result = None
        elif get_all:
            result = re.findall(r'<(.*?)>', string)
        else:
            result = re.search(r'<(.*?)>', string)
            if result is not None:
                result = result.group(1)
    
        return result

    def __get_data_dims_md(self):
        labels_visu_pars = ['VisuCoreDim', 'VisuCoreSize', 'VisuCoreWordType']
        self.__check_params_md('pv_visu_pars', labels_visu_pars)

        #VisuCoreFrameCount = self.pv_visu_pars.get('VisuCoreFrameCount') # Number of frames
        VisuCoreDim = self.pv_visu_pars.get('VisuCoreDim')
        VisuCoreSize = self.pv_visu_pars.get('VisuCoreSize')
        VisuCoreDimDesc = self.pv_visu_pars.get('VisuCoreDimDesc')
        VisuCoreWordType = self.pv_visu_pars.get('VisuCoreWordType')
        #VisuCoreByteOrder = self.pv_visu_pars.get('VisuCoreByteOrder')
        #VisuFGOrderDescDim = self.pv_visu_pars.get('VisuFGOrderDescDim')
        VisuFGOrderDesc = self.pv_visu_pars.get('VisuFGOrderDesc')

        dim_desc = None if VisuCoreDimDesc is None else VisuCoreDimDesc[0]

        # FrameGroup dimensions and names
        if (VisuFGOrderDesc is not None) and len(VisuFGOrderDesc) > 0:
            #fg_dims = list(map(lambda item: int(item[0]), VisuFGOrderDesc))
            #fg_names = list(map(lambda item: str(item[1]), VisuFGOrderDesc))
            fg_dims = [int(item[0]) for item in VisuFGOrderDesc]
            fg_names = [str(item[1]) for item in VisuFGOrderDesc]
            fg_names = [self.__extract_jcamp_strings_md(item, get_all=False) for item in fg_names]
        else:
            fg_dims = []
            fg_names = []

        # Data dimensions
        data_dims = list(map(int, VisuCoreSize)) + fg_dims

        # FrameGroup FG_SLICE index
        fg_index, fg_slice = (None, None)
        if VisuCoreDim == 2:
            fg_slices = ('FG_SLICE', 'FG_IRMODE')
            fg_indices = [fg_names.index(x) for x in fg_slices if x in fg_names]
            if len(fg_indices) > 0:
                fg_index = fg_indices[0]
                fg_slice = fg_slices[fg_index]
                fg_index += VisuCoreDim

        # ParaVision to NumPy data-type conversion
        if VisuCoreWordType == '_8BIT_UNSGN_INT':
            data_type = 'uint8'
        elif VisuCoreWordType == '_16BIT_SGN_INT':
            data_type = 'int16'
        elif VisuCoreWordType == '_32BIT_SGN_INT':
            data_type = 'int32'
        elif VisuCoreWordType == '_32BIT_FLOAT':
            data_type = 'float32'
        else:
            sys.exit("The data format is not correct specified.")

        return (data_dims, data_type, dim_desc, fg_index, fg_slice)

    def __map_data_md(self, data, map_pv6):
        VisuCoreExtent = self.pv_visu_pars.get('VisuCoreExtent')
        VisuCoreDataOffs = np.array(self.pv_visu_pars.get('VisuCoreDataOffs'), dtype=np.float32)
        VisuCoreDataSlope = np.array(self.pv_visu_pars.get('VisuCoreDataSlope'), dtype=np.float32)

        n = min(len(VisuCoreExtent), 3)
        dims = data.shape[n:]

        if VisuCoreDataOffs.size > 1:
            VisuCoreDataOffs = VisuCoreDataOffs.reshape(dims, order='F').astype(np.float32)
        else:
            VisuCoreDataOffs = np.float32(VisuCoreDataOffs[0])

        if VisuCoreDataSlope.size > 1:
            VisuCoreDataSlope = VisuCoreDataSlope.reshape(dims, order='F').astype(np.float32)
        else:
            VisuCoreDataSlope = np.float32(VisuCoreDataSlope[0])

        if map_pv6:
            data = data.astype(np.float32) / VisuCoreDataSlope
            data = data + VisuCoreDataOffs
        else:
            data = data.astype(np.float32) * VisuCoreDataSlope
            data = data + VisuCoreDataOffs

        return (data, 'float32')

    def getNiftiHeader_md(self, scale=10.0):
        labels_visu_pars = ['VisuCoreExtent']
        self.__check_params_md('pv_visu_pars', labels_visu_pars)

        #ACQ_fov = self.pv_acqp.get('ACQ_fov')
        ACQ_slice_sepn = self.pv_acqp.get('ACQ_slice_sepn')
        #ACQ_slice_thick = self.pv_acqp.get('ACQ_slice_thick')
        #PVM_Fov = self.pv_method.get('PVM_Fov')
        #PVM_SpatResol = self.pv_method.get('PVM_SpatResol')
        #PVM_SPackArrSliceGap = self.pv_method.get('PVM_SPackArrSliceGap')
        PVM_SPackArrSliceDistance = self.pv_method.get('PVM_SPackArrSliceDistance')
        VisuCoreExtent = self.pv_visu_pars.get('VisuCoreExtent')
        VisuCoreFrameThickness = self.pv_visu_pars.get('VisuCoreFrameThickness')
        #VisuCoreUnits = self.pv_visu_pars.get('VisuCoreUnits')
        VisuCoreSlicePacksSliceDist = self.pv_visu_pars.get('VisuCoreSlicePacksSliceDist')
        VisuAcqRepetitionTime = self.pv_visu_pars.get('VisuAcqRepetitionTime')

        data_dims, data_type, _, _, _ = self.__get_data_dims_md()

        nd = min(len(data_dims), 4)
        dims = [1] * 4
        dims[:nd] = data_dims
        nx, ny, nz, nt = dims

        # Voxel dimensions
        if len(VisuCoreExtent) > 1:
            dx = scale * float(VisuCoreExtent[0]) / nx
            dy = scale * float(VisuCoreExtent[1]) / ny
        else:
            dx = 1.0
            dy = 0.0
        if len(VisuCoreExtent) > 2:
            dz = scale * float(VisuCoreExtent[2]) / nz
        elif ACQ_slice_sepn is not None: # Slice thickness inclusive gap
            dz = scale * float(ACQ_slice_sepn[0])
        elif PVM_SPackArrSliceDistance is not None: # Slice thickness inclusive gap
            dz = scale * float(PVM_SPackArrSliceDistance[0])
        elif VisuCoreSlicePacksSliceDist is not None: # Slice thickness inclusive gap (PV6)
            dz = scale * float(VisuCoreSlicePacksSliceDist[0])
        elif VisuCoreFrameThickness is not None: # Slice thickness
            dz = scale * float(VisuCoreFrameThickness[0])
        else:
            dz = 0.0
        if (VisuAcqRepetitionTime is not None) and (nt > 1):
            dt = float(VisuAcqRepetitionTime[0]) / 1000.0
        else:
            dt = 0.0

        tmp = '%s.%s.%s' % (self.study, self.expno, self.procno)
        return (tmp, nx, ny, nz, nt, dx, dy, dz, dt, 0, 0, 0, data_type)

    def read_2dseq(self, map_raw=False, pv6=False, sc=1.0):
        study = self.study
        expno = self.expno
        procno = self.procno
        rawfolder = self.rawfolder

        datadir = os.path.join(rawfolder, study, expno, 'pdata', procno)

        self.acqp = pB.parsePV(os.path.join(rawfolder, study, expno, 'acqp'))
        self.method = pB.parsePV(os.path.join(rawfolder, study, expno, 'method'))
        self.subject = pB.parsePV(os.path.join(rawfolder, study, 'subject'))
        #self.d3proc = pB.parsePV(os.path.join(datadir, 'd3proc')) # removed for PV6
        self.visu_pars = pB.parsePV(os.path.join(datadir, 'visu_pars'))
        # get header information
        #hdr = pB.getNiftiHeader(self.visu_pars, sc=sc) # michaeld 20231004
        #print("hdr:", hdr)

        # michaeld 20231004
        _, self.pv_acqp = par.read_param_file(os.path.join(rawfolder, study, expno, 'acqp'))
        _, self.pv_method = par.read_param_file(os.path.join(rawfolder, study, expno, 'method'))
        #_, self.pv_subject = par.read_param_file(os.path.join(rawfolder, study, 'subject'))
        #_, self.pv_d3proc = par.read_param_file(os.path.join(datadir, 'd3proc')) # removed for PV6
        _, self.pv_visu_pars = par.read_param_file(os.path.join(datadir, 'visu_pars'))
        # get header information
        hdr = self.getNiftiHeader_md(scale=sc)
        #print("hdr:", hdr)

        if hdr is None or not isinstance(hdr[12], str):
            return

        # read '2dseq' file
        f_id = open(os.path.join(datadir, '2dseq'), 'rb')
        data = np.fromfile(f_id, dtype=np.dtype(hdr[12])).reshape(hdr[1], hdr[2], hdr[3], hdr[4], order='F')
        f_id.close()

        # map to raw data range (PV6)
        if map_raw:
            #visu_core_data_slope = np.array(map(float, self.visu_pars['VisuCoreDataSlope'].split()), dtype=np.float32)
            #visu_core_data_offs = np.array(map(float, self.visu_pars['VisuCoreDataOffs'].split()), dtype=np.float32)
            #visu_core_data_shape = list(data.shape)
            #visu_core_data_shape[:2] = (1, 1)
            #if pv6:
            #    data = data / visu_core_data_slope.reshape(visu_core_data_shape)
            #else:
            #    data = data * visu_core_data_slope.reshape(visu_core_data_shape)
            #data = data + visu_core_data_offs.reshape(visu_core_data_shape)
            data, _ = self.__map_data_md(data, pv6) # michaeld 20231004

        # NIfTI image
        nim = nii.Nifti1Image(data, None)

        # NIfTI header
        header = nim.header # michaeld 20230927
        #header = nim.get_header() # michaeld 20230927
        #print("header:"); print(header)
        header['pixdim'] = [0.0, hdr[5], hdr[6], hdr[7], hdr[8], 0.0, 0.0, 0.0]
        #nim.setXYZUnit('mm')
        header.set_xyzt_units(xyz='mm', t=None)
        #nim.header = header
        #header = nim.get_header()
        #print("header:"); print(header)

        # write header in xml structure
        #xml = pB.getXML(datadir + "/")
        xml = pB.getXML(os.path.join(datadir, 'visu_pars'))
        #print("xml:"); print(xml)

        # add protocol information (method, acqp, visu_pars, d3proc) to Nifti's header extensions
        #nim.extensions += ('comment', xml)
        #extension = nii.Nifti1Extension('comment', xml)

        self.hdr = hdr
        self.nim = nim
        self.xml = xml

    def save_nifti(self, subfolder=''):
        procfolder = os.path.join(self.procfolder, self.study)
        if not os.path.isdir(procfolder):
            os.mkdir(procfolder)

        protocol_name = self.acqp.get('ACQ_protocol_name').upper() # michaeld 20231009
        nz = self.hdr[3] # michaeld 20231009
        rotate = False # michaeld 20231009

        if any((sub in protocol_name) for sub in ('LOCALIZER', 'TRIPILOT')): # michaeld 20231018
            folder = 'Localizer'
        elif any((sub in protocol_name) for sub in ('DTI', 'DIFFUSION', 'QBALL')): # michaeld 20231009
            folder = 'DTI'
            rotate = True
        elif 'FMRI' in protocol_name:
            folder = 'fMRI'
            rotate = True
        elif ('TURBO' in protocol_name) and (nz > 1): # michaeld 20231009
            folder = 'T2w'
            rotate = True
        elif 'MSME' in protocol_name:
            folder = 'T2map'
        else:
            folder = 'Others'

        procfolder = os.path.join(self.procfolder, self.study, folder)
        if not os.path.isdir(procfolder):
            os.mkdir(procfolder)

        if self.ftype   == 'NIFTI_GZ': ext = 'nii.gz'
        elif self.ftype == 'NIFTI':    ext = 'nii'
        elif self.ftype == 'ANALYZE':  ext = 'img'
        else: ext = 'nii.gz'

        fname = '.'.join([self.study, self.expno, self.procno, ext])

        # Rotate around y-axis (flip x and z)
        if rotate: # michaeld 20231009
            self.nim = nii.Nifti1Image(np.flip(self.nim.get_fdata(), axis=(0, 2)), None, header=self.nim.header)
            print("The following file is rotated:")

        # write Nifti file
        print(os.path.join(procfolder, fname))
        if not hasattr(self, 'nim'):
            #return # michaeld 20230928
            return None # michaeld 20230928

        nib.save(self.nim, os.path.join(procfolder, fname))

        return os.path.join(procfolder, fname)

    def save_table(self, subfolder=''):
        procfolder = os.path.join(self.procfolder, self.study)
        if not os.path.isdir(procfolder):
            os.mkdir(procfolder)

        procfolder = os.path.join(self.procfolder, self.study, subfolder)
        if not os.path.isdir(procfolder):
            os.mkdir(procfolder)

        #dw_bval_each = float(self.method['PVM_DwBvalEach'])
        if 'PVM_DwEffBval' in self.method:
            dw_eff_bval = np.array(list(map(float, self.method['PVM_DwEffBval'].split())), dtype=np.float32)
        #print("dw_bval_each:", dw_bval_each)
        #print("dw_eff_bval:"); print(dw_eff_bval)

        if 'PVM_DwAoImages' in self.method:
            dw_ao_images = int(self.method['PVM_DwAoImages'])

        if 'PVM_DwNDiffDir' in self.method:
            dw_n_diff_dir = int(self.method['PVM_DwNDiffDir'])
        #print("dw_ao_images:", dw_ao_images)
        #print("dw_n_diff_dir:", dw_n_diff_dir)

            if 'PVM_DwDir' in self.method:
                dw_dir = np.array(list(map(float, self.method['PVM_DwDir'].split())), dtype=np.float32)
                dw_dir = dw_dir.reshape((dw_n_diff_dir, 3))

                nd = dw_ao_images + dw_n_diff_dir
                bvals = np.zeros(nd, dtype=np.float32)
                dwdir = np.zeros((nd, 3), dtype=np.float32)

                bvals[dw_ao_images:] = dw_eff_bval[dw_ao_images:]
                dwdir[dw_ao_images:] = dw_dir

                fname = '.'.join([self.study, self.expno, self.procno, 'btable', 'txt'])
                print(os.path.join(procfolder, fname))

                # Open btable file to write binary (windows format)

                #fid = open(os.path.join(procfolder, fname), 'wb') - py 2.6
                fid = open(os.path.join(procfolder, fname),mode='w',buffering=-1)

                for i in range(nd):
                    fid.write("%.4f" % (bvals[i],) + " %.8f %.8f %.8f" % tuple(dwdir[i]))
                    #print("%.4f" % (bvals[i],) + " %.8f %.8f %.8f" % tuple(dwdir[i]), end="\r\n", file=fid) - py 2.6

                # Close file
                fid.close()

                fname = '.'.join([self.study, self.expno, self.procno, 'bvals', 'txt'])
                print(os.path.join(procfolder, fname))

                # Open bvals file to write binary (unix format)
                fid = open(os.path.join(procfolder, fname), mode='w', buffering=-1)
                #fid = open(os.path.join(procfolder, fname), 'wb') - py 2.6


                fid.write(" ".join("%.4f" % (bvals[i],) for i in range(nd)))
                #print(" ".join("%.4f" % (bvals[i],) for i in range(nd)), end=chr(10), file=fid) - py 2.6

                # Close bvals file
                fid.close()

                fname = '.'.join([self.study, self.expno, self.procno, 'bvecs', 'txt'])
                print(os.path.join(procfolder, fname))

                # Open bvecs file to write binary (unix format)
                fid = open(os.path.join(procfolder, fname), mode='w', buffering=-1)
                #fid = open(os.path.join(procfolder, fname), 'wb') - py 2.6

                for k in range(3):
                    fid.write(" ".join("%.8f" % (dwdir[i,k],) for i in range(nd)))
                    #print(" ".join("%.8f" % (dwdir[i,k],) for i in range(nd)), end=chr(10), file=fid)- py 2.6

                # Close bvecs file
                fid.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert ParaVision to NIfTI')

    requiredNamed = parser.add_argument_group('Required named arguments')
    requiredNamed.add_argument('-i','--input_folder', help='raw data folder')
    # parser.add_argument('-o','--output_folder', help='output data folder')
    # parser.add_argument('study', help='study name')
    # parser.add_argument('expno', help='experiment number')
    # parser.add_argument('procno', help='processed (reconstructed) images number')
    parser.add_argument('-f','--model',
                        help='T2_2p  (default)  : Two   parameter T2 decay S(t) = S0 * exp(-t/T2)\n'
                             'T2_3p             : Three parameter T2 decay S(t) = S0 * exp(-t/T2) + C'
                        , nargs='?', const='T2_2p', type=str, default='T2_2p')
    parser.add_argument('-u','--upLim', help='upper limit of TE - default: 100', nargs='?', const=100, type=int, default=100)
    parser.add_argument('-s','--snrLim', help='upper limit of SNR - default: 1.5', nargs='?', const=1.5, type=float,
                        default=1.5)
    parser.add_argument('-k','--snrMethod', help='Brummer ,Chang, Sijbers', nargs='?', const='Brummer', type=str,
                        default='Brummer')
    parser.add_argument('-m', '--map_raw', action='store_true', help='get the real values')
    parser.add_argument('-p', '--pv6', action='store_true', help='ParaVision 6')
    parser.add_argument('-t', '--table', action='store_true', help='save b-values and diffusion directions')
    args = parser.parse_args()

    input_folder = None
    # raw data folder
    if args.input_folder is not None:
        input_folder = args.input_folder
    if not os.path.isdir(input_folder):
        sys.exit("Error: '%s' is not an existing directory." % (input_folder,))

    list_input = os.listdir(input_folder)
    listOfScans = [s for s in list_input if s.isdigit()]

    #if len(listOfScans) is 0: # michaeld 20230927
    if len(listOfScans) == 0: # michaeld 20230927
        sys.exit("Error: '%s' contains no numbered scans." % (input_folder,))

    print('Start to process '+str(len(listOfScans))+' scans...')
    procno ='1'
    study=input_folder.split('/')[len(input_folder.split('/'))-1]
    print(study)

    img = []
    for expno in np.sort(listOfScans):
        path = os.path.join(input_folder, expno, 'pdata', procno)
        if not os.path.isdir(path):
            sys.exit("Error: '%s' is not an existing directory." % (path,))

        if os.path.exists(os.path.join(path,'2dseq')):
            img = Bruker2Nifti(study, expno, procno, os.path.split(input_folder)[0], input_folder, ftype='NIFTI_GZ')
            img.read_2dseq(map_raw=True, pv6=False) # args.map_raw    args.pv6
            resPath = img.save_nifti()
            if resPath is None:
                continue
            if 'VisuAcqEchoTime' in img.visu_pars:
                echoTime = img.visu_pars['VisuAcqEchoTime']
                echoTime = np.fromstring(echoTime, dtype=float, sep=' ')
                #if len(echoTime) > 3: # michaeld 20231004
                #    mapT2.getT2mapping(resPath,args.model,args.upLim,args.snrLim,args.snrMethod,echoTime)
        else: 
            print("The following file does not exist, it will be skipped:")
            print(os.path.join(path,'2dseq'))
            continue

    if resPath is not None:
        pathlog = os.path.dirname(os.path.dirname(resPath))
        pathlog = os.path.join(pathlog, 'data.log')
        logfile = open(pathlog, 'w')
        if img.subject.get('coilname') is not None: # michaeld 20230927
            logfile.write(img.subject['coilname'])
        logfile.close()
