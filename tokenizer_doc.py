
import os
import pandas as pd
import json
import numpy as np
import sys

#======================================================
# path to parent file and placement of output folder
#======================================================


extracted_file = r'extracted_trafford\extracted'

def Folder_path(foldername):

        # finding where the file is
    try:
        if getattr(sys,'frozen',False):
            base_dir = os.path.dirname(sys.executable)

        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(base_dir, foldername)

            if os.path.exists(file_path):
                return file_path, base_dir

    except Exception as e:
        print(" There was an error while finding file path")



parent_folder_path, base_dir = Folder_path(extracted_file)

print(f'\n folder path is: {parent_folder_path}')
print(f'\n path to place output tokenized files is: {base_dir}')

#=========================================
# subfolder file paths
#=========================================



def subfolder_paths_func(foldername):

    subfolders_in_parent = os.listdir(foldername)
    subfolder_paths = [os.path.join(foldername, subpaths) for subpaths in subfolders_in_parent]
    return subfolder_paths, subfolders_in_parent

subfolder_paths_list, subfolder_names_list = subfolder_paths_func(parent_folder_path)

print(f'\n subfolder paths are: {subfolder_paths_list}')
print(f'\n subfolder names are: {subfolder_names_list}')


#==================================================
# creating output parent + subfolders
#==================================================

def create_output_and_subfolder_folders(subfolder_names, base_dir):
    '''
    maintain structure of files

    creates parent output folder = [base_dir, "tokenized_trafford"]

    creates tokenized extracted folder to maintain structure = [tokenized_trafford_path, "tokenized"]

    creates new subfiles with same names = [tokenized_path ,[subfilenames_itterated]]

    '''

    try:
        # creating parent folder
        output_parent_folder_path = os.path.join(base_dir, 'tokenized_trafford')
        os.makedirs(output_parent_folder_path, exist_ok=True)

        # creating tokenization folder to keep structure
        output_tokenized_folder_path = os.path.join(output_parent_folder_path, 'tokenized')
        os.makedirs(output_tokenized_folder_path, exist_ok=True)


        output_subfolder_paths = [os.path.join(output_tokenized_folder_path, subfolder_name) for subfolder_name in subfolder_names]


        # creates all subfolders
        for subfolder_path in output_subfolder_paths:
            os.makedirs(subfolder_path, exist_ok=True)

    except FileExistsError as e: # error should not run but just in case
        print(f"Folder already exists please check directory\n")
        print(f'Error message was:\n{e}')


    return output_subfolder_paths # for acesssing jsons


output_subfolder_paths = create_output_and_subfolder_folders(subfolder_names_list, base_dir)

print(f'\n new subfolder paths are: {output_subfolder_paths}') # path checking


#==================================================
# gathering json file paths
#==================================================

def json_paths(subfolder_paths, output_subfolder_pathing):

    # gives the names of the json files
    subfolder_json_file_names = [os.listdir(subfolder_path) for subfolder_path in subfolder_paths]

    # checking len of lists
    # print(len(subfolder_json_file_names))
    # print(len(output_subfolder_pathing))

    new_subfolder_json_pathing = [] # give absolute path so order irrelevant at this point
    old_subfolder_json_pathing = []

    for subfolder, output_path, old_path in zip(subfolder_json_file_names, output_subfolder_pathing, subfolder_paths):
        # print(f'\nthis is a subfolder:\n {subfolder}\n')
        for file in subfolder:
            print(f'\nthis is a file in a subfolder:\n {file}')
            print(f' this file is allocated to new path: {output_path}')
            print(f' this file is allocated to old path: {old_path}')

            new_file_json_pathing = os.path.join(output_path, file)
            old_file_json_pathing = os.path.join(old_path, file)

            new_subfolder_json_pathing.append(new_file_json_pathing)
            old_subfolder_json_pathing.append(old_file_json_pathing)

        print(f'\n#=================================#')
        print(f'#=================================#')
        print(f'#======== new subfolder ==========#')
        print(f'#=================================#')
        print(f'#=================================#\n')



        # subfolder_json_pathing = [os.path.join(output_path, files) for files in subfolders]

    return subfolder_json_file_names, new_subfolder_json_pathing, old_subfolder_json_pathing

json_file_names, output_json_file_paths, old_json_file_paths = json_paths(subfolder_paths_list, output_subfolder_paths)

print(f'\n new json names are:\n{json_file_names}') # name checking
print(f'\n old json paths are:\n{old_json_file_paths}') # path checking
print(f'\n new json paths are:\n{output_json_file_paths}') # path checking




#========================================================
# tokenizing json files and sending them to new path file
#========================================================




def tokenize_json(old_paths, new_paths):
    '''
    access json one by one

    tokenize the json file
    -

    send the file to new path in tokenised_trafford
    '''

    print(f'\n\nJSON data starts from here:')
    print(f'JSON data starts from here:')
    print(f'JSON data starts from here:')
    print(f'JSON data starts from here:')
    print(f'JSON data starts from here:\n\n')

    for json_path in old_paths:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            json_data = pd.DataFrame(json_data)
            print(f'\n{json_data.columns}\n')
            print(f'\n{json_data.rows}\n')


            # json_data = pd.DataFrame(json_data)
            # print(json_data['type'].value_counts())
            # # print(json_data.head(20))





            break


tokenize_json(old_json_file_paths, output_json_file_paths)


def iterate_through_jsons_in_subfiles():
    pass



























