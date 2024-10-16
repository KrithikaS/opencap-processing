import os
import sys
baseDir = os.path.join(os.getcwd())
sys.path.append(baseDir)

import json
import numpy as np
import opensim
import pandas as pd

from utils import numpy_to_storage
from utilsProcessing import get_filt_frequency, lowPassFilter
from utilsOpenSim import runIKTool

pathScaledModel = r"G:/Shared drives/HPL_Drive/ACL OpenCap Study/MOCAP OpenSim Pipeline/S1/Scaling/S1_scaled.osim"
pathGenericSetupFile = os.path.join(
                    baseDir, 'OpenSimPipeline', 
                    'InverseKinematics', 'Setup_IK_Generic.xml')
pathTRCFile_out = r"G:/Shared drives/HPL_Drive/ACL OpenCap Study/MOCAP OpenSim Pipeline/S1/Motion/Filtered/filtered_rotated_DLS0001.trc"
pathKinematicsFolder = r"C:\Users\Krithika-PC\repos\ks-opencap-processing\opencap-processing\Data\dc95f941-b377-4c0f-929f-c18fc3a202f1\OpenSimData_Mocap\Kinematics"
trialName = 'DLS0001'
lowpass_cutoff_frequency = get_filt_frequency(trialName)

sys.path.append(pathScaledModel)
sys.path.append(pathGenericSetupFile)
sys.path.append(pathTRCFile_out)
sys.path.append(pathKinematicsFolder)

runIKTool(pathGenericSetupFile, pathScaledModel, pathTRCFile_out, pathKinematicsFolder, IKFileName=trialName)

# Read in data to run lowPassFilter
motionPath = os.path.join(pathKinematicsFolder, '{}.mot'.format(trialName))
table = opensim.TimeSeriesTable(motionPath)        
tableProcessor = opensim.TableProcessor(table)
columnLabels = list(table.getColumnLabels())
tableProcessor.append(opensim.TabOpUseAbsoluteStateNames())
time = np.asarray(table.getIndependentColumn())
data = table.getMatrix().to_numpy()
data_filt = lowPassFilter(time, data, lowpass_cutoff_frequency)

# Add time back in and save to .sto file
data_filt = np.concatenate((np.expand_dims(time, axis=1), data_filt), axis=1)
columns = ['time'] + columnLabels
numpy_to_storage(columns, data_filt,  os.path.join(pathKinematicsFolder, 'ik_filtered_{}.sto'.format(trialName)), datatype='IK')