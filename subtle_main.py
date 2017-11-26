#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 16:50:27 2017

@author: sorenc, 
updated: enhao, Nov 21 2017
"""

import sys
import dicomutils
import numpy as np
import subprocess
from subtle_gad import enhance_gad

inargs=sys.argv[1:]
#example input arguments
#inargs=["lowcon","/home/sorenc/DEIDENT/DL_output/lowcon","precon","/home/sorenc/DEIDENT/DL_output/precon","highcon","/home/sorenc/DEIDENT/DL_output/highcon"]
# lowcon "../data/data_lowcon/lowcon_0006/011/" precon "../data/data_lowcon/lowcon_0006/007/" highcon "../data/data_lowcon/lowcon_0006/014/" "../data/data_lowcon/lowcon_0006/011_enhanced/"
# =============================================================================
# inargs=['precon', '/home/sorenc/SUBTLEDATA/JOBS/1.2.826.0.1.3680043.8.498.17603826117779956134153312/DEIDENT/precon',
#         'lowcon', '/home/sorenc/SUBTLEDATA/JOBS/1.2.826.0.1.3680043.8.498.17603826117779956134153312/DEIDENT/lowcon', 
#         'highcon', '/home/sorenc/SUBTLEDATA/JOBS/1.2.826.0.1.3680043.8.498.17603826117779956134153312/DEIDENT/highcon',
#         '/home/sorenc/SUBTLEDATA/JOBS/1.2.826.0.1.3680043.8.498.17603826117779956134153312/PROCESSED']
# =============================================================================
assert(len(inargs)==5)
#assert(len(inargs)=7)

#put args in dict for ease of use
arg_dict={}
for indx in range(0,len(inargs)-1,2):
    arg_dict[inargs[indx]]=inargs[indx+1]
#assert( set(arg_dict.keys())==set(["lowcon","highcon","precon"]))  #we do require these to be input
assert( set(arg_dict.keys())==set(["lowcon","precon"]))  #we do require these to be input

# get output folder
outputfolder = inargs[-1]

matexport=enhance_gad(dir_precon = arg_dict['precon'], 
            dir_lowcon = arg_dict['lowcon'])
  
# =============================================================================
# matexport=enhance_gad(dir_precon = arg_dict['precon'], 
#             dir_lowcon = arg_dict['lowcon'],
#             dir_highcon = arg_dict['highcon'],
#             dir_output = outputfolder)
# 
# =============================================================================
''' notify job is done '''

cseries=dicomutils.dicomscan(arg_dict['precon'])
preconseries= cseries.values()[0]

    #now get pixel data and write out results. Then call the dispatcher with some output

mat4d=np.flipud(np.transpose(np.expand_dims(matexport,3),(1,0,2,3)))
preconseries.write_pixeldata_to_file_based_on_series(outputfolder, mat4d,'Contrast Enhanced MRI DL','1120')


cmd="/home/sorenc/dicom_dispatcher/notify_dispatcher.py " + outputfolder  
print cmd    
print('calling command:', cmd)          
op=subprocess.Popen( cmd, shell = True)   #just let this guy run independently

sys.exit(0)

                           #the dispatcher will scan DICOMs in the outputfolder and reidentify+send to PACS (if configured)


#now notify the dispatcher that we have a complated job

