# -*- coding: utf-8 -*-
"""
Created on Jun 7 2024

@author: Krithika (modified from danie's v2 code)
"""

"""
    This script:
        
        Syncs mocap and video marker data.
        Computes MPJEs between mocap and video marker data and outputs the
        result in a report.
"""

# import
import sys
import os
import glob
import csv

sys.path.append('./..')
import utils as ut
import utilsTRC as dm
import numpy as np
import scipy
import copy
from scipy.spatial.transform import Rotation as R
import pandas as pd

# %% User inputs
markerdataFolder = 'MarkerData'
MPJEName = 'MPJEs'
saveProcessedMocapData = False  # Save processed and non-trimmed mocap data.
overwriteMarkerDataProcessed = False  # Overwrite non-trimmed mocap data.
overwriteForceDataProcessed = False  # Overwrite non-trimmed force data.
overwritevideoAndMocap = True  # Overwrite processed synced video-mocap data.
writeMPJE_condition = True  # Write MPJE for specific condition
writeMPJE_session = True  # Write MPJE for session

# Dict with frequencies for loading appropriate force data.
filtFrequencies = {'walking': '_filt6Hz',
                   'running': '_filt12Hz',
                   'squats': '_filt4Hz',
                   'STS': '_filt4Hz',
                   'SLDJ':'_filt999Hz',
                   'DLDJ':'_filt999Hz',
                   'DC':'_filt999Hz',
                   'C90':'_filt999Hz',
                   'CTH':'_filt999Hz',
                   'TH':'_filt999Hz',
                   'SLS':'_filt4Hz',
                   'DLS':'_filt4Hz'}

sessionDetails = {
    'Data': {
        # 'OpenCapData_S1': {}, #skipped
        # 'OpenCapData_S2': {}, #skipped
        'OpenCapData_S3': {}, #done
        'OpenCapData_S4': {}, #SLS_L1 and SLS_L2 too trimmed error
        'OpenCapData_S5': {}, #done
        'OpenCapData_S6': {}, #done
        'OpenCapData_S7': {}, #done
        'OpenCapData_S8': {}, #done
        'OpenCapData_S9': {}, #done
        'OpenCapData_S10': {}, #done
        'OpenCapData_S11': {}, #SLS_L2 too trimmed error
        'OpenCapData_S12': {}, #done
        'OpenCapData_S13': {}, #done
        'OpenCapData_S14': {}, #done
        # 'OpenCapData_S15': {}, #skipped
        'OpenCapData_S16': {}, #done
        'OpenCapData_S17': {}, #done
        'OpenCapData_S18': {}, #done
        'OpenCapData_S21': {}, #done
        'OpenCapData_S22': {}, #SLS_L3 was missing from mocap
        'OpenCapData_S23': {}, #done
        'OpenCapData_S24': {}, #done
        'OpenCapData_S25': {}, #done
        'OpenCapData_S26': {}, #done
        'OpenCapData_S28': {}, #done
        'OpenCapData_S29': {}, #done
        'OpenCapData_S30': {}, #done
        }
}

# Set stationary vs traversing here
is_stationary = True

# %% Paths to data dir.
dataDir = r"G:\Shared drives\HPL_Drive\ACL OpenCap Study\OpenCap Subject Data\InLabStationary" # OpenCap location after batchdownload
basemocapDir = r"G:\Shared drives\HPL_Drive\ACL OpenCap Study\MOCAP OpenSim Pipeline" # MoCap location on GDrive
baseTrialMappingFile = r"G:\Shared drives\HPL_Drive\ACL OpenCap Study\HPL OpenCap_ACL_MarkerEditingSheet_Excel.xlsx" # MoCap location on GDrive

# %% Editable script variables
# Parameters for different camera setups
# 0001 is stationary and 0002 is traversing
videoParameters = {}
videoParameters['0001'] = {}
#name the originmarker trial to match the one in the stationary or traversing marker folder
videoParameters['0001']['originName'] = 'Trimmed_origin1.trc'
videoParameters['0001']['r_fromMarker_toVideoOrigin_inLab'] = np.array(
    [0, 7, 0])  # mm in lab frame # this one is for large backwall 0001
videoParameters['0001']['R_video_opensim'] = R.from_euler('y', -90, degrees=True)
# videoParameters['0001']['R_opensim_xForward'] = R.from_euler('y', 90, degrees=True) # Original
videoParameters['0001']['R_opensim_xForward'] = R.from_euler('z', 90, degrees=True) # Original

videoParameters['0002'] = {}
videoParameters['0002']['originName'] = 'Trimmed_origin000002.trc'
videoParameters['0002']['r_fromMarker_toVideoOrigin_inLab'] = np.array(
    [-7, 0, 0])  # mm in lab frame # this one is for large backwall 0001
videoParameters['0002']['R_video_opensim'] = R.from_euler('y', 0, degrees=True)
videoParameters['0002']['R_opensim_xForward'] = R.from_euler('y', 0, degrees=True)

sessionInfoOverride = False  # true if you want to do specific trials below #!!! Not sure if this is doing anything


# %% Function definitions
def computeMarkerDifferences(trialName, mocapDir, videoTrcDir, markersMPJE, trialMappingFile, overwritevideo=False):
 
    print("trialName: ", trialName)
    
    # Define path to MOCAP file corresponding to opencap file #!!! UPDATE THIS BASED ON FINAL ORGANIZATIONAL STRUCTURE
    mocapTrcfile = "Trimmed_" + trialMappingFile['MOCAP Trial Name'][trialMappingFile['OpenCap Trial Name'] == trialName].values[0][:-4] + ".trc"
    mocapTrcPath = os.path.join(mocapDir, mocapTrcfile)
    print(mocapTrcPath)
    
    mocapTRC = dm.TRCFile(mocapTrcPath)
    mocapMkrNames = mocapTRC.marker_names
    mocapMkrNamesLower = [mkr.lower() for mkr in mocapMkrNames]
    mocapData = ut.TRC2numpy(mocapTrcPath, mocapMkrNames)[:, 1:]
    mocapTime = mocapTRC.time

    r_fromLabOrigin_toVideoOrigin_inLab = (np.mean(
        ut.TRC2numpy(mocapOriginPath, ['origin_marker'])[:, 1:], axis=0) +
                                           r_fromMarker_toVideoOrigin_inLab)  # add marker radius to y in mm

    # Force directory
    forceDir = os.path.join(os.path.join(mocapDir[:-20]), 'Forces') #!!! UPDATE THIS BASED ON FINAL ORGANIZATIONAL STRUCTURE
    
    filt_suffix = None
    for motion_type in filtFrequencies:
        if motion_type in trialName:
            filt_suffix = filtFrequencies[motion_type]
    if filt_suffix == None:
        raise ValueError('motion_type not recognized')
    
    # forceMotPath = os.path.join(forceDir, mocapTrcfile[8:-4] + '_forces' + filt_suffix + '.mot') #!!! UPDATE THIS BASED ON FINAL ORGANIZATIONAL STRUCTURE
    # headers_force = ['1_ground_force_vx', '1_ground_force_vy', '1_ground_force_vz',
    #                     '1_ground_force_px', '1_ground_force_py', '1_ground_force_pz',
    #                     '1_ground_torque_x', '1_ground_torque_y', '1_ground_torque_z',
    #                     '2_ground_force_vx', '2_ground_force_vy', '2_ground_force_vz',
    #                     '2_ground_force_px', '2_ground_force_py', '2_ground_force_pz',
    #                     '2_ground_torque_x', '2_ground_torque_y', '2_ground_torque_z',
    #                     '3_ground_force_vx', '3_ground_force_vy', '3_ground_force_vz',
    #                     '3_ground_force_px', '3_ground_force_py', '3_ground_force_pz',
    #                     '3_ground_torque_x', '3_ground_torque_y', '3_ground_torque_z']

    # forceData = ut.storage2df(forceMotPath, headers_force).to_numpy()[:, 1:]
    # forceTime = ut.storage2df(forceMotPath, headers_force).to_numpy()[:, 0]

    # Video-trc directory
    videoTrcPath = os.path.join(videoTrcDir, trialName + '.trc')
    videoTRC = dm.TRCFile(videoTrcPath)
    videoMkrNames = videoTRC.marker_names
    videoMkrNamesLower = [mkr.lower() for mkr in videoMkrNames]
    videoData = ut.TRC2numpy(videoTrcPath, videoMkrNames)[:, 1:]
    newTime = np.arange(videoTRC.time[0], np.round(videoTRC.time[-1] + 1 / mocapTRC.camera_rate, 6),
                        1 / mocapTRC.camera_rate)
    vidInterpFxn = scipy.interpolate.interp1d(videoTRC.time, videoData, axis=0, fill_value='extrapolate')
    videoData = vidInterpFxn(newTime)
    if videoTRC.units == 'm': videoData = videoData * 1000
    videoTime = newTime

    # Filter mocap data
    if mocapFiltFreq is not None:
        mocapData = ut.filter3DPointsButterworth(mocapData, mocapFiltFreq, mocapTRC.camera_rate, order=4)
        videoData = ut.filter3DPointsButterworth(videoData, mocapFiltFreq, mocapTRC.camera_rate, order=4)

    # Copy filtered mocap data
    mocapData_all = copy.deepcopy(mocapData)
    
    # Rotate camera data and subtract origin
    R_lab_opensim = R.from_euler('x', -90, degrees=True)
    r_labOrigin_videoOrigin_inOpensim = R_lab_opensim.apply(r_fromLabOrigin_toVideoOrigin_inLab)

    for iMkr in range(videoTRC.num_markers):
        # Original
        # videoData[:, iMkr * 3:iMkr * 3 + 3] = R_video_opensim.apply(
        #     videoData[:, iMkr * 3:iMkr * 3 + 3]) + r_labOrigin_videoOrigin_inOpensim
        
        videoData[:, iMkr * 3:iMkr * 3 + 3] = R_video_opensim.apply(
            videoData[:, iMkr * 3:iMkr * 3 + 3])
        
        R_tmp = R.from_euler('x', 90, degrees=True)
        videoData[:, iMkr * 3:iMkr * 3 + 3] = R_tmp.apply(
            videoData[:, iMkr * 3:iMkr * 3 + 3]) + r_labOrigin_videoOrigin_inOpensim
         
    # Select sync algorithm that minimizes the MPJEs
    markersSync = markersMPJE
    lag_markerError_sumabs, success_sumabs = syncMarkerError(
        mocapData, videoData, markersSync, mocapMkrNamesLower, videoMkrNamesLower)
    lag_markerError_norm, success_norm = syncMarkerError(
        mocapData, videoData, markersSync, mocapMkrNamesLower, videoMkrNamesLower, method='norm')
    lag_verticalVelocity = syncVerticalVelocity(
        mocapData, videoData, mocapTRC, videoTRC)

    if success_sumabs:
        outputMPJE_markerError_abs = getMPJEs(lag_markerError_sumabs, trialName, videoTime, mocapTime, mocapTRC,
                                              mocapData,
                                              videoData, mocapMkrNamesLower, videoMkrNamesLower, videoTRC, markersMPJE)
    else:
        outputMPJE_markerError_abs = {}
        outputMPJE_markerError_abs['MPJE_offsetRemoved_mean'] = 1e6

    if success_norm:
        outputMPJE_markerError_norm = getMPJEs(lag_markerError_norm, trialName, videoTime, mocapTime, mocapTRC,
                                               mocapData,
                                               videoData, mocapMkrNamesLower, videoMkrNamesLower, videoTRC, markersMPJE)
    else:
        outputMPJE_markerError_norm = {}
        outputMPJE_markerError_norm['MPJE_offsetRemoved_mean'] = 1e6

    outputMPJE_verticalVelocity = getMPJEs(lag_verticalVelocity, trialName, videoTime, mocapTime, mocapTRC, mocapData,
                                           videoData, mocapMkrNamesLower, videoMkrNamesLower, videoTRC, markersMPJE)
    outputMPJE_all = np.array([outputMPJE_markerError_abs['MPJE_offsetRemoved_mean'],
                               outputMPJE_markerError_norm['MPJE_offsetRemoved_mean'],
                               outputMPJE_verticalVelocity['MPJE_offsetRemoved_mean']])
    idx_min = np.argmin(outputMPJE_all)

    if idx_min == 0:
        MPJE_mean = outputMPJE_markerError_abs['MPJE_mean']
        MPJE_std = outputMPJE_markerError_abs['MPJE_std']
        MPJE_offsetRemoved_mean = outputMPJE_markerError_abs['MPJE_offsetRemoved_mean']
        MPJE_offsetRemoved_std = outputMPJE_markerError_abs['MPJE_offsetRemoved_std']
        MPJEvec = outputMPJE_markerError_abs['MPJEvec']
        MPJE_offVec = outputMPJE_markerError_abs['MPJE_offVec']
        videoDataOffsetRemoved = outputMPJE_markerError_abs['videoDataOffsetRemoved']
        syncTimeVec = outputMPJE_markerError_abs['syncTimeVec']
        mocapData = outputMPJE_markerError_abs['mocapData']
        videoData = outputMPJE_markerError_abs['videoData']
        videoDataAllOffsetRemoved = outputMPJE_markerError_abs['videoDataAllOffsetRemoved']
        timeVecVideoAll = outputMPJE_markerError_abs['timeVecVideoAll']
        videoDataAll = outputMPJE_markerError_abs['videoDataAll']
    elif idx_min == 1:
        MPJE_mean = outputMPJE_markerError_norm['MPJE_mean']
        MPJE_std = outputMPJE_markerError_norm['MPJE_std']
        MPJE_offsetRemoved_mean = outputMPJE_markerError_norm['MPJE_offsetRemoved_mean']
        MPJE_offsetRemoved_std = outputMPJE_markerError_norm['MPJE_offsetRemoved_std']
        MPJEvec = outputMPJE_markerError_norm['MPJEvec']
        MPJE_offVec = outputMPJE_markerError_norm['MPJE_offVec']
        videoDataOffsetRemoved = outputMPJE_markerError_norm['videoDataOffsetRemoved']
        syncTimeVec = outputMPJE_markerError_norm['syncTimeVec']
        mocapData = outputMPJE_markerError_norm['mocapData']
        videoData = outputMPJE_markerError_norm['videoData']
        videoDataAllOffsetRemoved = outputMPJE_markerError_norm['videoDataAllOffsetRemoved']
        timeVecVideoAll = outputMPJE_markerError_norm['timeVecVideoAll']
        videoDataAll = outputMPJE_markerError_abs['videoDataAll']
    elif idx_min == 2:
        MPJE_mean = outputMPJE_verticalVelocity['MPJE_mean']
        MPJE_std = outputMPJE_verticalVelocity['MPJE_std']
        MPJE_offsetRemoved_mean = outputMPJE_verticalVelocity['MPJE_offsetRemoved_mean']
        MPJE_offsetRemoved_std = outputMPJE_verticalVelocity['MPJE_offsetRemoved_std']
        MPJEvec = outputMPJE_verticalVelocity['MPJEvec']
        MPJE_offVec = outputMPJE_verticalVelocity['MPJE_offVec']
        videoDataOffsetRemoved = outputMPJE_verticalVelocity['videoDataOffsetRemoved']
        syncTimeVec = outputMPJE_verticalVelocity['syncTimeVec']
        mocapData = outputMPJE_verticalVelocity['mocapData']
        videoData = outputMPJE_verticalVelocity['videoData']
        videoDataAllOffsetRemoved = outputMPJE_verticalVelocity['videoDataAllOffsetRemoved']
        timeVecVideoAll = outputMPJE_verticalVelocity['timeVecVideoAll']
        videoDataAll = outputMPJE_verticalVelocity['videoDataAll']

    # write TRC with combined data
    outData = np.concatenate((mocapData, videoData, videoDataOffsetRemoved), axis=1)
    outVideoDataAll = np.concatenate((videoDataAll,videoDataAllOffsetRemoved),axis=1)

    # rotate so x is always forward
    for iMkr in range(int(outData.shape[1] / 3)):
        outData[:, iMkr * 3:iMkr * 3 + 3] = R_opensim_xForward.apply(outData[:, iMkr * 3:iMkr * 3 + 3])
    
    for iMkr in range(int(outVideoDataAll.shape[1]/3)):
        outVideoDataAll[:,iMkr*3:iMkr*3+3] = R_opensim_xForward.apply(outVideoDataAll[:,iMkr*3:iMkr*3+3])

    # rotate original mocap data
    for iMkr in range(int(mocapData_all.shape[1] / 3)):
        mocapData_all[:, iMkr * 3:iMkr * 3 + 3] = R_opensim_xForward.apply(mocapData_all[:, iMkr * 3:iMkr * 3 + 3])

    videoMkrNamesNoOffset = [mkr + '_offsetRemoved' for mkr in videoMkrNames]
    outMkrNames = mocapMkrNames + videoMkrNames + videoMkrNamesNoOffset

    outputDir = os.path.join(videoTrcDir, 'videoAndMocap')
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    pathOutputFile = os.path.join(outputDir, trialName + '_videoAndMocap.trc')

    if not os.path.exists(pathOutputFile) or overwritevideoAndMocap:
        with open(pathOutputFile, "w") as f:
            ut.numpy2TRC(f, outData, outMkrNames, fc=mocapTRC.camera_rate, units="mm", t_start=syncTimeVec[0])

    # This is exporting the full synced video data, including the offset removed data. This is useful to
    # have a buffer when generating dynamics simulations.
    pathOutputFileVideoAll = os.path.join(outputDir,trialName + '_video.trc')
    if not os.path.exists(pathOutputFileVideoAll) or overwritevideo:
        with open(pathOutputFileVideoAll,"w") as f:
            ut.numpy2TRC(f, outVideoDataAll, videoMkrNames + videoMkrNamesNoOffset, fc=mocapTRC.camera_rate, units="mm",t_start=timeVecVideoAll[0])

    if saveProcessedMocapData and overwriteMarkerDataProcessed:
        outputDirMocap = os.path.join(os.path.dirname(mocapDir), 'MarkerDataProcessed')
        os.makedirs(outputDirMocap, exist_ok=True)
        pathOutputFileMocap = os.path.join(outputDirMocap, trialName + '.trc')
        with open(pathOutputFileMocap, "w") as f:
            ut.numpy2TRC(f, mocapData_all, mocapMkrNames, fc=mocapTRC.camera_rate, units="mm", t_start=mocapTime[0])

    if saveProcessedMocapData and overwriteForceDataProcessed:
        outputDirMocapF = os.path.join(os.path.dirname(mocapDir), 'ForceDataProcessed')
        os.makedirs(outputDirMocapF, exist_ok=True)
        pathOutputFileMocapF = os.path.join(outputDirMocapF, trialName + '_forces.mot')
        labels = ['time'] + headers_force
        forceData_all = np.concatenate((np.expand_dims(forceTime, axis=1), forceData), axis=1)
        ut.numpy2storage(labels, forceData_all, pathOutputFileMocapF)

    return MPJE_mean, MPJE_std, MPJE_offsetRemoved_mean, MPJE_offsetRemoved_std, MPJEvec, MPJE_offVec


def writeMPJE(trialNames, videoTrcDir,
              MPJE_mean, MPJE_std, MPJEvec,
              MPJE_offsetRemoved_mean, MPJE_offsetRemoved_std, MPJE_offVec,
              headers):
    os.makedirs(videoTrcDir, exist_ok=True)
    with open(os.path.join(videoTrcDir, 'MPJE.csv'), 'w', newline='') as csvfile:
        csvWriter = csv.writer(csvfile)
        headers_all = ['trial'] + headers + ['mean'] + ['std']
        _ = csvWriter.writerow(headers_all)
        _ = csvWriter.writerow([''])
        _ = csvWriter.writerow(['With offset'])
        for idxTrial, tName in enumerate(trialNames):
            MPJErow = [tName]
            for c_m, marker in enumerate(headers):
                MPJErow.extend(['%.2f' % MPJEvec[idxTrial, c_m]])
            MPJErow.extend(['%.2f' % MPJE_mean[idxTrial], '%.2f' % MPJE_std[idxTrial]])
            _ = csvWriter.writerow(MPJErow)

        _ = csvWriter.writerow([''])
        _ = csvWriter.writerow(['Without offset'])
        for idxTrial, tName in enumerate(trialNames):
            MPJErow = [tName]
            for c_m, marker in enumerate(headers):
                MPJErow.extend(['%.2f' % MPJE_offVec[idxTrial, c_m]])
            MPJErow.extend(['%.2f' % MPJE_offsetRemoved_mean[idxTrial], '%.2f' % MPJE_offsetRemoved_std[idxTrial]])
            _ = csvWriter.writerow(MPJErow)


def writeMPJE_perSession(trialNames, outputDir, MPJE_session,
                         MPJE_offsetRemoved_session, analysisNames):
    with open(os.path.join(outputDir, 'MPJE_fullSession.csv'), 'w', newline='') as csvfile:
        csvWriter = csv.writer(csvfile)
        topRow = ['trial']
        for label in analysisNames:
            topRow.extend([label, '', ''])
        _ = csvWriter.writerow(topRow)
        secondRow = ['']
        secondRow.extend(["MPJE", "offsetRmvd", ''] * len(analysisNames))
        _ = csvWriter.writerow(secondRow)
        for idxTrial, tName in enumerate(trialNames):
            MPJErow = [tName]
            for MPJE, MPJE_offsetRemoved in zip(MPJE_session, MPJE_offsetRemoved_session):
                MPJErow.extend(['%.2f' % MPJE[idxTrial], '%.2f' % MPJE_offsetRemoved[idxTrial], ''])
            _ = csvWriter.writerow(MPJErow)


def syncMarkerError(mocapData, videoData, markersSync, mocapMkrNamesLower,
                    videoMkrNamesLower, method='sumabs'):
    mocapSubset = np.zeros((mocapData.shape[0], len(markersSync) * 3))
    videoSubset = np.zeros((videoData.shape[0], len(markersSync) * 3))
    for i, mkr in enumerate(markersSync):
        idxM = mocapMkrNamesLower.index(mkr.lower())
        idxV = videoMkrNamesLower.index(mkr.lower() + '_study')
        mocapSubset[:, i * 3:i * 3 + 3] = mocapData[:, idxM * 3:idxM * 3 + 3]
        videoSubset[:, i * 3:i * 3 + 3] = videoData[:, idxV * 3:idxV * 3 + 3]

    if mocapSubset.shape[0] >= videoSubset.shape[0]:
        lag = 0
        success = False
        return lag, success

    mkrDist = np.zeros(videoSubset.shape[0] - mocapSubset.shape[0])
    nMocapSamps = mocapSubset.shape[0]
    i = 0
    while i < len(mkrDist):  # this assumes that the video is always longer than the mocap
        # Use Euclidian instead
        if method == 'norm':
            mkrNorm = np.zeros((mocapSubset.shape[0], len(markersSync)))
            for j in range(0, len(markersSync)):
                mkrNorm[:, j] = np.linalg.norm(
                    videoSubset[i:i + nMocapSamps, j * 3:(j + 1) * 3] - mocapSubset[:, j * 3:(j + 1) * 3], axis=1)
            mkrDist[i] = np.sum(np.mean(mkrNorm, axis=1))
        elif method == 'sumabs':
            mkrDist[i] = np.sum(np.abs(videoSubset[i:i + nMocapSamps] - mocapSubset))
        i += 1
    lag = -np.argmin(mkrDist)
    success = True
    # if (lag == 0 or np.min(mkrDist)/(len(markersSync)*mocapSubset.shape[0]) > 75):
    #     success = False

    return lag, success


def syncVerticalVelocity(mocapData, videoData, mocapTRC, videoTRC):
    mocapY = mocapData[:, np.arange(1, mocapTRC.num_markers * 3, 3)]
    videoYInds = np.arange(1, videoTRC.num_markers * 3, 3).tolist()
    # del videoYInds[15:18] # delete face markers
    # del videoYInds[0]
    # TODO: might make sense to only select augmented markers...
    # if videoTRC.num_markers == 63:
    #     videoYInds = videoYInds[23:]
    # else:
    #     raise ValueError("Assumption about number of marker is wrong")
    videoY = videoData[:, videoYInds]
    d_mocapY = np.diff(mocapY, axis=0)
    d_videoY = np.diff(videoY, axis=0)
    d_mocapY_sum = np.sum(d_mocapY, axis=1) / np.max(np.sum(d_mocapY, axis=1))
    d_videoY_sum = np.sum(d_videoY, axis=1) / np.max(np.sum(d_videoY, axis=1))

    corVal, lag = ut.cross_corr(d_mocapY_sum, d_videoY_sum,
                                          visualize=False)  # neg lag means video started before mocap

    return lag


def getMPJEs(lag, trialName, videoTime, mocapTime, mocapTRC, mocapData,
             videoData, mocapMkrNamesLower, videoMkrNamesLower, videoTRC, markersMPJE):
    if lag > 0 and 'tatic' not in trialName:
        print('WARNING video starts {} frames after mocap.'.format(lag))

    # Sync based on lag computed
    # Make sure video data is longer than mocap data on both ends
    if len(videoTime) < len(mocapTime):# and 'running' in trialName:
        MPJE = np.nan
        MPJE_offsetRemoved = np.nan
        print('{} too trimmed'.format(trialName))
        return MPJE, MPJE_offsetRemoved
    videoTime = videoTime + lag / mocapTRC.camera_rate

    startTime = np.max([videoTime[0], mocapTime[0]])
    endTime = np.min([videoTime[-1], mocapTime[-1]])

    N = int(np.round(np.round((endTime - startTime), 6) * mocapTRC.camera_rate))
    NVideoAll = int(np.round(np.round((videoTime[-1] - videoTime[0]),2) * mocapTRC.camera_rate))
    syncTimeVec = np.linspace(startTime, endTime, N + 1)
    timeVecVideoAll = np.linspace(videoTime[0], videoTime[-1], NVideoAll+1)

    print('t_start = ' + str(syncTimeVec[0]) + 's')
    print('t_end = ' + str(syncTimeVec[-1]) + 's')

    mocapInds = np.arange(len(syncTimeVec)) + np.argmin(np.abs(mocapTime - startTime))
    videoInds = np.arange(len(syncTimeVec)) + np.argmin(np.abs(videoTime - startTime))

    # Copy videoData before offset removal
    videoDataAll = copy.deepcopy(videoData)

    mocapData = mocapData[mocapInds, :]
    videoData = videoData[videoInds, :]

    offsetMkrs = markersMPJE
    offsetVals = np.empty((len(offsetMkrs), 3))
    for iMkr, mkr in enumerate(offsetMkrs):
        idxM = mocapMkrNamesLower.index(mkr.lower())
        idxV = videoMkrNamesLower.index(mkr.lower() + '_study')
        offsetVals[iMkr, :] = np.mean(videoData[:, idxV * 3:idxV * 3 + 3] - mocapData[:, idxM * 3:idxM * 3 + 3], axis=0)
    offsetMean = np.mean(offsetVals, axis=0)
    videoDataOffsetRemoved = videoData - np.tile(offsetMean, (1, videoTRC.num_markers))
    videoDataAllOffsetRemoved = videoDataAll - np.tile(offsetMean,(1,videoTRC.num_markers))

    # compute per joint errors
    MPJEvec = np.empty(len(markersMPJE))
    MPJE_offVec = np.copy(MPJEvec)
    for i, mkr in enumerate(markersMPJE):
        idxM = mocapMkrNamesLower.index(mkr.lower())
        idxV = videoMkrNamesLower.index(mkr.lower() + '_study')
        MPJEvec[i] = np.mean(
            np.linalg.norm(videoData[:, idxV * 3:idxV * 3 + 3] - mocapData[:, idxM * 3:idxM * 3 + 3], axis=1))
        MPJE_offVec[i] = np.mean(
            np.linalg.norm(videoDataOffsetRemoved[:, idxV * 3:idxV * 3 + 3] - mocapData[:, idxM * 3:idxM * 3 + 3],
                           axis=1))
    MPJE_mean = np.mean(MPJEvec)
    MPJE_std = np.std(MPJEvec)
    MPJE_offsetRemoved_mean = np.mean(MPJE_offVec)
    MPJE_offsetRemoved_std = np.std(MPJE_offVec)

    outputMPJE = {}
    outputMPJE['MPJE_mean'] = MPJE_mean
    outputMPJE['MPJE_std'] = MPJE_std
    outputMPJE['MPJE_std'] = MPJE_std
    outputMPJE['MPJE_offsetRemoved_mean'] = MPJE_offsetRemoved_mean
    outputMPJE['MPJE_offsetRemoved_std'] = MPJE_offsetRemoved_std
    outputMPJE['MPJEvec'] = MPJEvec
    outputMPJE['MPJE_offVec'] = MPJE_offVec
    outputMPJE['videoDataOffsetRemoved'] = videoDataOffsetRemoved
    outputMPJE['syncTimeVec'] = syncTimeVec
    outputMPJE['mocapData'] = mocapData
    outputMPJE['videoData'] = videoData
    outputMPJE['videoDataAllOffsetRemoved'] = videoDataAllOffsetRemoved
    outputMPJE['timeVecVideoAll'] = timeVecVideoAll
    outputMPJE['videoDataAll'] = videoDataAll

    return outputMPJE

def get_filt_frequency(trialName):
    # Hard-code filter frequencies based on the activity
    if 'LS' in trialName: # squat
        return 4
    elif 'DJ' in trialName: # drop-jump
        return 30
    elif 'DC' in trialName: # drop-cut
        return 50
    elif 'TH' in trialName: # triple hop
        return 50
    elif 'C9' in trialName: # run-cut
        return 60
    else: # keep the previous settings
        return None
    
# %% Process data
MPJEs = {}

# %%
# print("sessionDetails: ", sessionDetails)
for subjectName in sessionDetails:
    if not subjectName in MPJEs:
        MPJEs[subjectName] = {}
    # print('\n\nProcessing {}'.format(subjectName))
    for sessionName in sessionDetails[subjectName]:
        print('\nProcessing {}'.format(sessionName))

        if not sessionInfoOverride:
            if is_stationary:
                subSession = '0001'
            elif not is_stationary:
                subSession = '0002'
            else:
                raise ValueError("Error sub-session")

            originTrialName = videoParameters[subSession]['originName']
            r_fromMarker_toVideoOrigin_inLab = videoParameters[subSession]['r_fromMarker_toVideoOrigin_inLab']
            R_video_opensim = videoParameters[subSession]['R_video_opensim']
            R_opensim_xForward = videoParameters[subSession]['R_opensim_xForward']

        mocapDir = os.path.join(basemocapDir, sessionName[12:], 'Motion', 'CortexTrimmed') #!!! UPDATE THIS BASED ON FINAL ORGANIZATIONAL STRUCTURE
        mocapOriginPath = os.path.join(mocapDir, originTrialName)
        trialMappingFile = pd.read_excel(baseTrialMappingFile, sheet_name = sessionName[12:] + "_inlab")

        # Loop over MarkerData files for different numbers of cameras
        subjectVideoDir = os.path.join(dataDir, sessionName)
        
        # Force order markerDataFolders to facilitate comparison with old data
        markerDataFolders = []
        markerDataFolders.append(os.path.join(subjectVideoDir, markerdataFolder))

        analysisNames = []
        MPJE_session = []
        MPJE_offsetRemoved_session = []
        for markerDataFolder in markerDataFolders:
            temp = markerDataFolder.split('\\')
            poseDetector = temp[-1]

            if not poseDetector in MPJEs[subjectName]:
                MPJEs[subjectName][poseDetector] = {}

            print('\nProcessing {}'.format(markerDataFolder))
            
            camComboFolders = []
            camComboFolders.append(os.path.join(markerDataFolder))

            for camComboFolder in camComboFolders:
                temp2 = camComboFolder.split('\\')
                cameraSetup = temp2[-1]

                if not cameraSetup in MPJEs[subjectName][poseDetector]:
                    MPJEs[subjectName][poseDetector][cameraSetup] = {}
                
                postAugmentationType = 'old' #!!! not sure what this is for/if we can remove

                if not postAugmentationType in MPJEs[subjectName][poseDetector][cameraSetup]:
                    MPJEs[subjectName][poseDetector][cameraSetup][postAugmentationType] = {}

                videoTrcDir = os.path.join(camComboFolder)

                # Get trialnames - hard code, or get all in video directory, as long as in mocap directory
                trialNames = [os.path.split(tName.replace('_LSTM', ''))[1][0:-4] for tName in
                              glob.glob(videoTrcDir + '/*.trc')]
                trialsToRemove = []
                
                for tName in trialNames:  # check if in mocap directory, if not, delete trial
                    # Check that there is a matching MOCAP trial name
                    if trialMappingFile['MOCAP Trial Name'][trialMappingFile['OpenCap Trial Name'] == tName].empty or pd.isnull(trialMappingFile['MOCAP Trial Name'][trialMappingFile['OpenCap Trial Name'] == tName]).values[0]:
                        trialsToRemove.append(tName)
                print("Removed trials: ", trialsToRemove)
                [trialNames.remove(tName) for tName in trialsToRemove]
                
                # Sort trialnames so same order each time
                trialNames.sort()

                if len(trialNames) == 0:
                    raise Exception('No matching trialnames. Check paths: ' + videoTrcDir + '\n  ' + mocapDir)

                # Markers for MPJE computation (add arms eventually) #!!! CHECK THAT THESE ARE THE ONES WE WANT TO USE
                markersMPJE = ['c7', 'r_shoulder', 'l_shoulder', 'r.ASIS', 'l.ASIS', 'r.PSIS', 'l.PSIS', 'r_knee',
                                'l_knee', 'r_ankle', 'l_ankle', 'r_calc', 'l_calc', 'r_toe', 'l_toe', 'r_5meta',
                                'l_5meta']

                # Compute and save MPJE
                MPJE_mean = np.zeros((len(trialNames)))
                MPJE_offsetRemoved_mean = np.zeros((len(trialNames)))
                MPJE_std = np.zeros((len(trialNames)))
                MPJE_offsetRemoved_std = np.zeros((len(trialNames)))
                MPJE_markers = np.zeros((len(trialNames), len(markersMPJE)))
                MPJE_offsetRemoved_markers = np.zeros((len(trialNames), len(markersMPJE)))
                for idxTrial, trialName in enumerate(trialNames):
                    # Set specific trial type here
                    # if 'DJ' not in trialName:
                    #     continue

                    try:
                        mocapFiltFreq = get_filt_frequency(trialName)
                        MPJE_mean[idxTrial], MPJE_std[idxTrial], MPJE_offsetRemoved_mean[idxTrial], \
                        MPJE_offsetRemoved_std[idxTrial], MPJE_markers[idxTrial, :], MPJE_offsetRemoved_markers[
                                                                                        idxTrial,
                                                                                        :] = computeMarkerDifferences(
                            trialName, mocapDir, videoTrcDir, markersMPJE, trialMappingFile)
                        if not 'headers' in MPJEs:
                            MPJEs_headers = markersMPJE.copy()
                            MPJEs_headers.append('mean')
                            MPJEs_headers.append('std')
                            MPJEs['headers'] = MPJEs_headers
                        c_MPJEs = np.zeros((len(markersMPJE) + 2,))
                        c_MPJEs[:len(markersMPJE), ] = MPJE_offsetRemoved_markers[idxTrial, :]
                        c_MPJEs[len(markersMPJE),] = MPJE_offsetRemoved_mean[idxTrial]
                        c_MPJEs[-1,] = MPJE_offsetRemoved_std[idxTrial]

                        MPJEs[subjectName][poseDetector][cameraSetup][postAugmentationType][trialName] = c_MPJEs
                    except Exception as e:
                        print(e)
                        nan_vec = np.nan * np.ones((len(markersMPJE),))
                        MPJE_mean[idxTrial], MPJE_std[idxTrial], MPJE_offsetRemoved_mean[idxTrial], \
                        MPJE_offsetRemoved_std[idxTrial], MPJE_markers[idxTrial, :], MPJE_offsetRemoved_markers[
                                                                                        idxTrial,
                                                                                        :] = np.nan, np.nan, np.nan, np.nan, nan_vec, nan_vec

                # Write to file
                outputDir = os.path.join(videoTrcDir, 'videoAndMocap')
                if writeMPJE_condition:
                    writeMPJE(trialNames, outputDir, MPJE_mean, MPJE_std, MPJE_markers,
                                MPJE_offsetRemoved_mean, MPJE_offsetRemoved_std, MPJE_offsetRemoved_markers,
                                markersMPJE)
                poseDetector = os.path.basename(os.path.normpath(markerDataFolder))
                nCams = os.path.basename(os.path.normpath(camComboFolder))
                analysisNames.append(poseDetector + '_' + nCams + '_' +
                                        postAugmentationType)
                MPJE_session.append(MPJE_mean)
                MPJE_offsetRemoved_session.append(MPJE_offsetRemoved_mean)

        # Write all MPJE to session file
        markerDataDir = os.path.join(subjectVideoDir, markerdataFolder)
        if writeMPJE_session:
            writeMPJE_perSession(trialNames, markerDataDir, MPJE_session,
                                 MPJE_offsetRemoved_session, analysisNames)