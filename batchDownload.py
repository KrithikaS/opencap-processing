'''
    ---------------------------------------------------------------------------
    OpenCap processing: batchDownload.py
    ---------------------------------------------------------------------------

    Copyright 2022 Stanford University and the Authors
    
    Author(s): Antoine Falisse, Scott Uhlrich
    
    Licensed under the Apache License, Version 2.0 (the "License"); you may not
    use this file except in compliance with the License. You may obtain a copy
    of the License at http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
'''

from utils import download_session
import os

# List of sessions you'd like to download. They go to the Data folder in the 
# current directory.


sessionList = [
                # In-Lab Stationary
#                 'dc95f941-b377-4c0f-929f-c18fc3a202f1', # S1
#                 '332f5657-efa0-454b-8f82-e63488096006', # S2
#                 'd5ff6e67-c237-4000-86a4-47b10b31e33b', # S3
#                 '53a6aaea-8059-414b-84ca-f4a0f4a777d5', # S4
#                 '78e028a0-a2c2-4a1a-b1c1-1013b13963ea', # S5
#                 '2adc93da-0853-40ef-8a7a-6aa15e4b2e2a', # S6
#                 '94cfc971-1167-419d-9474-bed0b96e5206', # S7
#                 '71ca0104-9a13-44a9-a12e-e0865b4cc70f', # S8
#                 '8794230e-fa8e-4f44-ba62-5868ab88fde7', # S9
#                 'a119d7e6-1628-495e-b576-8eb892c8193d', # S10
#                 '44788707-43ee-4031-9e95-8724852da152', # S11
#                 '1ec22e49-cbe0-4b60-a4bd-5d2ad54459a6', # S12
#                 '5a3a1e42-4744-49a3-a6f6-2cc0ed53bcfe', # S13
#                 '2f4d561f-c4bf-469d-9a42-bf12ef2271c8', # S14
#                 '9fc09046-4247-4dbb-8bf4-98758c02211e', # S15
#                 'd8f5cdca-34de-4e23-9007-55c886616a28', # S16
#                 '031dbb98-9f5d-4ab4-bdd3-d4b3c52e9bdf', # S17
#                 'ff39bfb3-8293-4d27-b9ca-41972af28692', # S18
#                 '6e7901ec-23d7-4ed7-939c-a143e6d03f9f', # S21
#                 '34a2a803-3ffb-4718-a2d6-040d02f5baa7', # S22
#                 'a9ec7429-fd69-4922-b4b4-4ce41b4570c9', # S23
#                 '3409b96e-90cb-4ef8-a67e-b72d7407d0f4', # S24
#                 '373c45d0-dc2d-4eb4-bb0e-4cc6dce6301f', # S25
#                 'b011b3e2-203a-4e98-87fd-c6ea4d95acbf', # S26
#                 'af17abef-7507-48f6-941b-25d152d317ed', # S28
#                 'd10751a5-7e94-495a-94d0-2dd229ca39e0', # S29
#                 'e742eb1c-efbc-4c17-befc-a772150ca84d', # S30

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


# # alternatively, read list of session IDs from CSV column
# from pathlib import Path
# import pandas as pd
# fpath = Path('~/Documents/paru/session_ids_fshd.csv')
# df = pd.read_csv(fpath)
# sessionList = df['sid'].dropna().unique()

             
# base directory for downloads. Specify None if you want to go to os.path.join(os.getcwd(),'Data')
downloadPath = os.path.join(os.getcwd(),'Data')

for session_id in sessionList:
    # If only interested in marker and OpenSim data, downladVideos=False will be faster
    download_session(session_id,sessionBasePath=downloadPath,downloadVideos=True)