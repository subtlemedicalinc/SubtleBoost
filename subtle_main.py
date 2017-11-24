#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 16:50:27 2017

@author: sorenc, 
updated: enhao, Nov 21 2017
"""

import sys,os
import dicomutils
import numpy as np
import subprocess
from subtle_gad import enhance_gad

inargs=sys.argv[1:]
#example input arguments
#inargs=["lowcon","/home/sorenc/DEIDENT/DL_output/lowcon","precon","/home/sorenc/DEIDENT/DL_output/precon","highcon","/home/sorenc/DEIDENT/DL_output/highcon"]
# lowcon "../data/data_lowcon/lowcon_0006/011/" precon "../data/data_lowcon/lowcon_0006/007/" highcon "../data/data_lowcon/lowcon_0006/014/" "../data/data_lowcon/lowcon_0006/011_enhanced/"

assert(len(inargs)==7)

#put args in dict for ease of use
arg_dict={}
for indx in range(0,len(inargs)-1,2):
    arg_dict[inargs[indx]]=inargs[indx+1]
assert( set(arg_dict.keys())==set(["lowcon","highcon","precon"]))  #we do require these to be input

# get output folder
outputfolder = inargs[-1]
if not os.path.exists(outputfolder):
	os.mkdir(outputfolder)
    
enhance_gad(dir_precon = arg_dict['precon'], 
			dir_lowcon = arg_dict['lowcon'],
			dir_highcon = arg_dict['highcon'],
			dir_output = outputfolder,
			usage_gpu = '0',
			process_highcon = True,
			skip_coreg = False,
			# batch_size = 4,
			# gpu_memory = 0.8,
			)
			

''' notify job is done '''
cmd = "/usr/local/bin/notify_dispatcher.py " + outputfolder                
print('calling command:', cmd)
try:
	subprocess.Popen( cmd, shell = True)                                #the dispatcher will scan DICOMs in the outputfolder and reidentify+send to PACS (if configured)
except:
	print('error calling command:', cmd)


#now notify the dispatcher that we have a complated job
