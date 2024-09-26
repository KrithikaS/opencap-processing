# # -*- coding: utf-8 -*-
# """
# Created on Mon Jul 15 023

# @author: Krithika
# """

# import pandas as pd

# # Load the Excel sheet
# excel_file_path = "trial_info.xlsx"  # Replace with the path to your Excel file
# df = pd.read_excel(excel_file_path, engine="openpyxl")

# # Initialize empty lists to store the results
# pid_list = []
# sid_list = []
# trial_list = []
# trial_clean_list = []

# # Iterate over the rows and filter based on the "trial_clean" column
# count = 0
# for index, row in df.iterrows():
#     if "10mwt" in str(row["trial_clean"]).lower():
#         if "mdf" in str(row["pid"]).lower():
#             pid_list.append(row["pid"])
#             sid_list.append(row["sid"])
#             trial_list.append(row["trial"])
#             trial_clean_list.append(row["trial_clean"])
            
#             print('{}: {{"pid": "{}", "sid": "{}", "trial": "{}", "trial_clean": "{}"}},'.format(100+count, str(row["pid"]), str(row["sid"]), str(row["trial"]), str(row["trial_clean"])))
#             count += 1
# Print the lists
# print("PID List:", pid_list)
# print("SID List:", sid_list)
# print("Trial List:", trial_list)
from utils import get_session_json

# %% Specific cases
def get_data_select_window():
    
    data = {
        # General structure was a dictionary with:
        # index: [start_time, end_time], ...
        }
    
    return data

def get_data_manual_alignment():
    
    # Define alignments for sessions with unique calibrations
    alignment_unique = {
        # General structure was a dictionary with:
        # session_id: angle, ...
    }
    
    data = {        
        # General structure was a dictionary with:
        # index: {'angle':alignment_unique['session_id']}, ...
        # A bit unclear why this needed two stages
        }
    
    return data
    

def get_data_select_previous_cycle():
    # Seems like this is to deal with errors in the cropping to choose previous stride/rep
    data = {
        # General structure was a dictionary with:
        # index: {'leg':['l'], ...
        }
    
    return data

def get_data_info_problems():
    # Seems like this is to deal with bad trials
    data = {
        # General structure was a dictionary with:
        # index: {'leg':['l'], ... # could also be ['l', 'r']
    }   

    return data

# Add optional argument to specify the trial index.
# If not specified, return all trials.
def get_data_info(trial_indexes=[]):
    # Seems like this gives info on participant ID, session ID, trial name, and some other information
    data = {
        # General structure was a dictionary with:
        # index: {"pid":"subj_id", "sid": "session_id", "trial": "trialname",...
        0: {"pid": "S30", "sid": "e742eb1c-efbc-4c17-befc-a772150ca84d", "trial": "DLS1"},
    }

    if trial_indexes:
        # Return data dict with only the specified trials.
        return {trial_index: data[trial_index] for trial_index in trial_indexes}
    else:
        return data
# %%

def get_data_info_by_session(session_list, activityList = ['LS']):
    data = {}

    index = 0
    for i in session_list:
        for key in i.keys():
            session = get_session_json(i[key])
            for trial in session['trials']:
                # print(trial['name'])
                if any(activity in trial['name'] for activity in activityList):
                    # print('got here')
                    data[index] = {'pid': key, 'sid': i[key], 'trial': trial['name']}
                    index += 1

    return data