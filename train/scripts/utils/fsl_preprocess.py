import sys
import numpy as np
import os
from os.path import abspath

# import subtle.subtle_preprocess as sup
import nibabel as nib
import nipype
from nipype.interfaces import fsl
import nipype.interfaces.io


print(fsl.Info().version())


working_dir = sys.argv[1]

nib_files = []
for filename in os.listdir(working_dir):
    if ".nii" in filename.lower():
            nib_files.append(os.path.join(working_dir,filename))
order = np.argsort([int(os.path.basename(l).split('_')[0]) for l in nib_files])
nib_files = list(np.array(nib_files)[order])
print(nib_files)

if len(nib_files) != 3:
    print('ERROR: incorrect number of files', len(nib_files))
    sys.exit(-1)

frac = .5
robust = False
reduce_bias = True

bet_zero_node = nipype.Node(fsl.BET(frac=frac, reduce_bias=reduce_bias), name='bet_zero')
bet_low_node = nipype.Node(fsl.BET(frac=frac, reduce_bias=reduce_bias), name='bet_low')
bet_full_node = nipype.Node(fsl.BET(frac=frac, reduce_bias=reduce_bias), name='bet_full')

cost = 'corratio'
dof = 6
searchr_x = [-20, 20]
searchr_y = [-20, 20]
searchr_z = [-20, 20]
bins = 256
interp = 'trilinear'

coreg_low_node = nipype.Node(fsl.FLIRT(cost=cost, dof=dof, interp=interp, bins=bins, searchr_x=searchr_x, searchr_y=searchr_y, searchr_z=searchr_z), name='coreg_low')
coreg_full_node = nipype.Node(fsl.FLIRT(cost=cost, dof=dof, interp=interp, bins=bins, searchr_x=searchr_x, searchr_y=searchr_y, searchr_z=searchr_z), name='coreg_full')

wf = nipype.Workflow(name='registration', base_dir=working_dir)
wf.connect([
    (bet_zero_node, coreg_low_node, [('out_file', 'reference')]),
    (bet_low_node, coreg_low_node, [('out_file', 'in_file')]),
    (bet_zero_node, coreg_full_node, [('out_file', 'reference')]),
    (bet_full_node, coreg_full_node, [('out_file', 'in_file')]),
    ])
wf.write_graph(graph2use='flat')

bet_zero_node.inputs.in_file = nib_files[0]
bet_low_node.inputs.in_file = nib_files[1]
bet_full_node.inputs.in_file = nib_files[2]

# wf.run()
wf.run(plugin='MultiProc', plugin_args={'n_procs' : 10})
