'''
    ---------------------------------------------------------------------------
    OpenCap processing: batch_processing_aclsimulations.py
    ---------------------------------------------------------------------------
    Copyright 2023 Stanford University and the Authors
    
    Author(s): Scott Uhlrich & Antoine Falisse
    
    Licensed under the Apache License, Version 2.0 (the "License"); you may not
    use this file except in compliance with the License. You may obtain a copy
    of the License at http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.    

    MODIFIED BY KRITHIKA SWAMINATHAN FOR ACL BATCH PROCESSING -- AUG 2024
'''

import os
import sys
sys.path.append("ActivityAnalyses")
sys.path.append("UtilsDynamicSimulations/OpenSimAD")
baseDir = os.path.join(os.getcwd())
sys.path.append(baseDir)
activityAnalysesDir = os.path.join(baseDir, 'ActivityAnalyses')
sys.path.append(activityAnalysesDir)
opensimADDir = os.path.join(baseDir, 'UtilsDynamicSimulations', 'OpenSimAD')
sys.path.append(opensimADDir)
import json
import numpy as np
import pandas as pd

# from gait_analysis import gait_analysis
# from squat_analysis import squat_analysis
from utils import get_trial_id, download_trial, numpy_to_storage, TRC2numpy
from utilsPlotting import plot_dataframe
import utilsTRC as dm

from utilsProcessing import align_markers_with_ground_3, get_filt_frequency, crop_to_roi
from utilsOpenSim import runIKTool

from data_info import get_data_info, get_data_info_by_session, get_data_info_problems, get_data_select_previous_cycle, get_data_manual_alignment, get_data_select_window

from utilsOpenSimAD import processInputsOpenSimAD, plotResultsOpenSimAD
from mainOpenSimAD import run_tracking

from utilsKineticsOpenSimAD import kineticsOpenSimAD
from utilsKinematics import kinematics

# %% Paths.
driveDir_synced = r"G:\Shared drives\HPL_Drive\ACL OpenCap Study\OpenCap Subject Data\InLabStationary" #HARD-CODED
driveDir = r"C:\Users\Krithika-PC\repos\ks-opencap-processing\opencap-processing" #HARD-CODED
dataFolder = os.path.join(driveDir, 'Data')

basemocapDir = r"G:\Shared drives\HPL_Drive\ACL OpenCap Study\MOCAP OpenSim Pipeline" # MoCap location on GDrive HARD-CODED
baseTrialMappingFile = r"G:\Shared drives\HPL_Drive\ACL OpenCap Study\HPL OpenCap_ACL_MarkerEditingSheet_Excel.xlsx" # MoCap location on GDrive HARD-CODED

# Session List
sessionList = [
                # In-Lab Stationary
                # {'S1':'dc95f941-b377-4c0f-929f-c18fc3a202f1'},
                # {'S2':'332f5657-efa0-454b-8f82-e63488096006'},
                # {'S3':'d5ff6e67-c237-4000-86a4-47b10b31e33b'},
                {'S4':'53a6aaea-8059-414b-84ca-f4a0f4a777d5'},
                {'S5':'78e028a0-a2c2-4a1a-b1c1-1013b13963ea'},
                {'S6':'2adc93da-0853-40ef-8a7a-6aa15e4b2e2a'},
                {'S7':'94cfc971-1167-419d-9474-bed0b96e5206'},
                {'S8':'71ca0104-9a13-44a9-a12e-e0865b4cc70f'},
                {'S9':'8794230e-fa8e-4f44-ba62-5868ab88fde7'},
                {'S10':'a119d7e6-1628-495e-b576-8eb892c8193d'},
                {'S11':'44788707-43ee-4031-9e95-8724852da152'},
                {'S12':'1ec22e49-cbe0-4b60-a4bd-5d2ad54459a6'},
                {'S13':'5a3a1e42-4744-49a3-a6f6-2cc0ed53bcfe'},
                {'S14':'2f4d561f-c4bf-469d-9a42-bf12ef2271c8'},
                {'S15':'9fc09046-4247-4dbb-8bf4-98758c02211e'},
                {'S16':'d8f5cdca-34de-4e23-9007-55c886616a28'},
                {'S17':'031dbb98-9f5d-4ab4-bdd3-d4b3c52e9bdf'},
                {'S18':'ff39bfb3-8293-4d27-b9ca-41972af28692'},
                {'S21':'6e7901ec-23d7-4ed7-939c-a143e6d03f9f'},
                {'S22':'34a2a803-3ffb-4718-a2d6-040d02f5baa7'},
                {'S23':'a9ec7429-fd69-4922-b4b4-4ce41b4570c9'},
                {'S24':'3409b96e-90cb-4ef8-a67e-b72d7407d0f4'},
                {'S25':'373c45d0-dc2d-4eb4-bb0e-4cc6dce6301f'},
                {'S26':'b011b3e2-203a-4e98-87fd-c6ea4d95acbf'},
                {'S28':'af17abef-7507-48f6-941b-25d152d317ed'},
                {'S29':'d10751a5-7e94-495a-94d0-2dd229ca39e0'},
                {'S30':'e742eb1c-efbc-4c17-befc-a772150ca84d'},

                # Traversing
                # '1443ed48-3cbe-4226-a2cc-bc34e21a0fb3', # S1 - Change the neutral number of frames to 5 from 10
                # '5249a0b7-282d-4339-bef2-8e3b3dc88372', # S2
                # '9c1b5be7-cf95-4f0f-abc2-5f117b134123', # S3
                # '26c51f1d-6b16-4a80-af35-de61818a2c85', # S4
                # '81121c4c-f197-4323-b4c5-a7649a5c5f93', # S5
                # '1b365060-f439-406d-a98d-cbd1be79966b', # S6
                # 'b09da98f-df14-4285-8b24-a3b3e4df05d6', # S7
                # 'd353fa2a-1c02-40f7-9a1c-f4dbe5d02a83', # S8
                # 'aa735c89-e43f-4a0f-8a6b-117c5d854d79', # S9
                # '09a91af3-de83-488c-ae98-9201af866e95', # S10
                # 'b07c0de4-d081-4dda-a941-ceca43522c7b', # S11
                # '263c4a4e-2c18-4157-a276-a8a8d1842b35', # S12
                # '064ff969-08ab-4aad-9cfc-973b70b19c37', # S13
                # '89214e8a-b2f3-45b5-b3ed-91115cf5e69a', # S14
                # '1e5aae82-9380-4d7a-9d9f-89ff240fc987', # S15
                # '76d371ba-3c89-4383-aa19-7971e82fe718', # S16
                # '7b3f02d6-cecc-4a5f-9314-a32d81d695c3', # S17
                # '2b80cdd6-1f4c-4f08-91e9-7670694a91d3', # S18
                # 'a4007d87-5cef-446a-b2d7-d6581e1cd63e', # S21
                # '6fdffc2c-eade-4ddc-b8d5-dc047f5dd488', # S22
                # '379fb610-90ed-459a-ae9d-d0b7070ec4a8', # S23
                # '238f41ce-0e35-4594-b4cd-df5042628a2f', # S24
                # '1fc766fc-c103-4116-a1ee-9fc7a1c62e5d', # S25
                # 'c8ec5277-0adc-4afb-a7c0-a767dc9fa5d6', # S26
                # 'f09528a1-0992-409f-9f8f-e7e852b8e4e8', # S28
                # 'eb517e30-17c8-4b00-a46b-8d564a53b5f8', # S29
                # '9fd8370e-4fd3-4caa-a085-42c29eb497b5', # S30
                ]

# %% User-defined variables.
# Select how many gait cycles you'd like to analyze. Select -1 for all gait
# cycles detected in the trial.
n_gait_cycles = -1 # DEL AT END

# Settings for dynamic simulation.
motion_type = 'squats_ka'
case = '2'
legs = ['r','l']
runProblem = True
overwrite_aligned_data = False
overwrite_results = False
overwrite_tracked_motion_file = False
processInputs = False
runSimulation = False
solveProblem = False
analyzeResults = True

# Buffers
if case == '2': #Q: What did the cases refer to in the example code?
    buffer_start = 0.3
    buffer_end = 0.3
# elif ...
    
# %% Kinematic analysis.
trials_info = get_data_info_by_session(sessionList)
print(trials_info)

trials_info_problems = get_data_info_problems()
trials_select_previous_cycle = get_data_select_previous_cycle()
trials_manual_alignment = get_data_manual_alignment()
trials_select_window = get_data_select_window() # HARD-CODED -- NEED TO FIX to automate segmentation

for trial in trials_info:
    # Get trial info.
    pid = trials_info[trial]['pid']
    session_id = trials_info[trial]['sid']
    trial_name = trials_info[trial]['trial']
    print(trials_info[trial])
    print('Processing session {} - trial {}...'.format(session_id, trial_name))
    # Get trial id from name.
    trial_id = get_trial_id(session_id, trial_name)
    # Set session path.
    # sessionDir = os.path.join(dataFolder, "{}_{}".format(trial, session_id)) #Q: Why do it this way? anyway trials is just going to be a number?
    sessionDir = os.path.join(dataFolder, session_id)
    # sessionDir = os.path.join(dataFolder)
    # Set kinematic folder path.
    pathKinematicsFolder = os.path.join(sessionDir, 'OpenSimData', 'Kinematics')

    # Get corresponding Mocap directories and files
    mocapDir = os.path.join(basemocapDir, pid, 'IK', 'Results_06_2024') #!!! UPDATE THIS BASED ON FINAL ORGANIZATIONAL STRUCTURE
    forceDir = os.path.join(basemocapDir, pid, 'Forces')#!!! UPDATE THIS BASED ON FINAL ORGANIZATIONAL STRUCTURE
    trialMappingFile = pd.read_excel(baseTrialMappingFile, sheet_name = pid + "_inlab")
    try:
        mocapStofile = trialMappingFile['MOCAP Trial Name'][trialMappingFile['OpenCap Trial Name'] == trial_name].values[0][:-4] + "_results_filtered.sto"
    except:
        print("No good mocap data for this trial - SKIPPING")
        continue
    mocapStoPath = os.path.join(mocapDir, mocapStofile)
    print(mocapStoPath)

    # Get filter frequency
    filter_frequency = get_filt_frequency(trial_name)
    print("Using filter frequency: ", filter_frequency, "Hz")

    # Sync data
    subj_folder = 'OpenCapData_' + pid
    trial_file = trial_name + '_video.trc'
    dataFolder_synced = os.path.join(driveDir_synced, subj_folder, 'MarkerData', 'videoAndMocap', trial_file)
    print(dataFolder_synced)
    try:
        dataTRC = dm.TRCFile(dataFolder_synced)
    except:
        print("Error in reading data trc file -- SKIPPING")
        continue
    dataMkrNames = dataTRC.marker_names
    data_synced = TRC2numpy(dataFolder_synced, dataMkrNames)
    t_delay = data_synced[0][0]
    # print(t_delay)

    # Crop to region of interest based on the MOCAP motion data (squats) or FP data (all else)
    # print(pathKinematicsFolder, trial_name)
    t_start, t_end = crop_to_roi(mocapStoPath, trial_name) - t_delay
    print(t_start, t_end)

    if runProblem:
        # Download data.
        try:
            trialName, modelName = download_trial(trial_id, sessionDir, session_id=session_id)
        except Exception as e:
            print(f"Error downloading trial {trial_id}: {e}")
            continue
        
        # We align all trials with ground.
        suffixOutputFileName = 'aligned'
        trialName_aligned = trialName + '_' + suffixOutputFileName
        
        angle = 0 # HARD-CODED -- NEED TO FIX!
        if trial in trials_manual_alignment:
            angle = trials_manual_alignment[trial]['angle']
            
        select_window = []
        if trial in trials_select_window:
            select_window = trials_select_window[trial]
        
        # Do if not already done or if overwrite_aligned_data is True.
        # Inverse kinematics
        if not os.path.exists(os.path.join(sessionDir, 'OpenSimData', 'Kinematics', trialName_aligned + '.mot')) or overwrite_aligned_data:
            print('Aligning markers with ground...')
            try:       
                pathTRCFile_out = align_markers_with_ground_3(
                    sessionDir, trialName,
                    suffixOutputFileName=suffixOutputFileName,
                    lowpass_cutoff_frequency_for_marker_values=filter_frequency,
                    angle=angle, select_window=select_window)
                # Run inverse kinematics.
                print('Running inverse kinematics...')
                # pathGenericSetupFile = os.path.join(
                #     baseDir, 'OpenSimPipeline', 
                #     'InverseKinematics', 'Setup_InverseKinematics.xml')
                pathGenericSetupFile = os.path.join(
                    baseDir, 'OpenSimPipeline', 
                    'InverseKinematics', 'Setup_IK_KA.xml') # HARD-CODED - NEED TO FIX ONCE THIS IS FINALIZED; ALSO IS THIS BEING RE-RUN SINCE MARKERS ARE RE-ALIGNED?
                pathScaledModel = os.path.join(sessionDir, 'OpenSimData', 'Model', modelName)
                runIKTool(pathGenericSetupFile, pathScaledModel, pathTRCFile_out, pathKinematicsFolder)
            except Exception as e:
                print(f"Error alignment trial {trial_id}: {e}")
                continue
                    
        # Dynamic simulations
        print('Processing data...')        
        for leg in legs:
            if trial in trials_info_problems.keys():
                if leg in trials_info_problems[trial]['leg']:
                    print('Skipping leg {} for trial {} because it is a problem trial.'.format(leg, trial))
                    continue
            
            # Squat segmentation and analysis.
            # Purposefuly save with trialName and not trialName_aligned, since trialName is the trial actually being analysed.
            pathOutputJsonFile = os.path.join(pathKinematicsFolder, '{}_kinematic_features_{}.json'.format(trialName, leg))

            # Setup dynamic optimization problem for each time interval found of interest
            for i in range(len(t_start)):
                print(i)
                time_window = [t_start[i], t_end[i]]

                # Adjust time window to add buffers
                time_window[0] = time_window[0] - buffer_start
                time_window[1] = time_window[1] + buffer_end
                
                # Creating mot files for visualization.
                pathResults = os.path.join(sessionDir, 'OpenSimData', 'Dynamics', trialName_aligned)
                case_leg = '{}_{}'.format(case, leg)
                pathTrackedMotionFile = os.path.join(pathResults, 'kinematics_to_track_{}.mot'.format(case_leg))            
                # if not os.path.exists(pathTrackedMotionFile) or overwrite_tracked_motion_file:
                #     gait = gait_analysis(
                #         sessionDir, trialName_aligned, leg=leg,
                #         lowpass_cutoff_frequency_for_coordinate_values=filter_frequency,
                #         n_gait_cycles=n_gait_cycles)
                #     coordinateValues = gait.coordinateValues
                #     time = coordinateValues['time'].to_numpy()
                #     labels = coordinateValues.columns
                #     idx_start = (np.abs(time - time_window[0])).argmin()
                #     idx_end = (np.abs(time - time_window[1])).argmin() + 1
                #     data = coordinateValues.to_numpy()[idx_start:idx_end, :]
                #     os.makedirs(pathResults, exist_ok=True)
                #     numpy_to_storage(labels, data, pathTrackedMotionFile, datatype='IK')

                print('Processing data for dynamic simulation...')
                if processInputs:
                    try:
                        settings = processInputsOpenSimAD(
                            baseDir, sessionDir, session_id, trialName_aligned, 
                            motion_type, time_window=time_window)
                    except Exception as e:
                        print(f"Error setting up dynamic optimization for trial {trial_id}: {e}")
                        continue
            
                # Simulation.
                if runSimulation:
                    try:
                        run_tracking(baseDir, sessionDir, settings, case=case_leg, 
                                    solveProblem=solveProblem, analyzeResults=analyzeResults)
                        test=1
                    except Exception as e:
                        print(f"Error during dynamic optimization for trial {trial_id}: {e}")
                        continue
        
    else:
        suffixOutputFileName = 'aligned'
        trialName_aligned = trial_name + '_' + suffixOutputFileName
        # plotResultsOpenSimAD(sessionDir, session_id, trialName_aligned, cases=['2_r','2_l'], mainPlots=True)
        plotResultsOpenSimAD(dataFolder, session_id, trial_name, cases=['2_r','2_l'], mainPlots=True)

