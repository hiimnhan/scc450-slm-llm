
import os
import pandas as pd
import json
import numpy as np
import sys
from ftfy import fix_text
import re
import string
from nltk.corpus import stopwords
import unicodedata
from bs4 import BeautifulSoup
from gemma import gm
import math
from bs4 import MarkupResemblesLocatorWarning
import warnings
import time
from transformers import AutoTokenizer

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

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

# print(f'\n folder path is: {parent_folder_path}')
# print(f'\n path to place output tokenized files is: {base_dir}')

#=========================================
# subfolder file paths
#=========================================



def subfolder_paths_func(foldername):

    subfolders_in_parent = os.listdir(foldername)
    subfolder_paths = [os.path.join(foldername, subpaths) for subpaths in subfolders_in_parent]
    return subfolder_paths, subfolders_in_parent

subfolder_paths_list, subfolder_names_list = subfolder_paths_func(parent_folder_path)

# print(f'\n subfolder paths are: {subfolder_paths_list}')
# print(f'\n subfolder names are: {subfolder_names_list}')


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

# print(f'\n new subfolder paths are: {output_subfolder_paths}') # path checking


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
            # print(f'\nthis is a file in a subfolder:\n {file}')
            # print(f' this file is allocated to new path: {output_path}')
            # print(f' this file is allocated to old path: {old_path}')

            new_file_json_pathing = os.path.join(output_path, file)
            old_file_json_pathing = os.path.join(old_path, file)

            new_subfolder_json_pathing.append(new_file_json_pathing)
            old_subfolder_json_pathing.append(old_file_json_pathing)

        # print(f'\n#=================================#')
        # print(f'#=================================#')
        # print(f'#======== new subfolder ==========#')
        # print(f'#=================================#')
        # print(f'#=================================#\n')



        # subfolder_json_pathing = [os.path.join(output_path, files) for files in subfolders]

    return subfolder_json_file_names, new_subfolder_json_pathing, old_subfolder_json_pathing

json_file_names, output_json_file_paths, old_json_file_paths = json_paths(subfolder_paths_list, output_subfolder_paths)

# print(f'\n new json names are:\n{json_file_names}') # name checking
# print(f'\n old json paths are:\n{old_json_file_paths}') # path checking
# print(f'\n new json paths are:\n{output_json_file_paths}') # path checking




#======================================================================
# cleaning  json files and sending them to new path file
#======================================================================


def cleaner_func(text):
    '''
    - clean text
        - mojibake
        - lecture ---- example: fixed = broken.encode('latin-1').decode('utf-8') = â€œHelloâ€• = “Hello”
        - remove unwanted characters
        - strip and join to ensure clear and consistent spacing
        - remove filler words
        - remove punctuation
    - create tokenizer for chatGPT # diff func
    - create tokenizer for Gemma # different func
    - return value

    '''

    # Parsing html
    if text is None or (isinstance(text, float) and math.isnan(text)): # missing value in value for check box
        text = 'MISSING'
    elif isinstance(text,bool):
        text = str(text)

    soup = BeautifulSoup(text, "html.parser")

    # only take out the text
    cleaned_text = soup.get_text(separator=" ")



    # removing the mojibake from the text
    cleaned_text = fix_text(cleaned_text)


    # normalising any unicode chars to single format - umlow and accents
    cleaned_text = unicodedata.normalize("NFC", cleaned_text)


    # making sure that there are no invisible chars ---- replace, with, from
    cleaned_text = re.sub(r"[\u0000-\u001F\u007F-\u009F]", "", cleaned_text)



    # removing punctuation - replaced chars, replaced with, deleted chars( all punctuation) -- gonna add : back in when making text str
    # keep_apos_com = string.punctuation.replace(("',","")
    # cleaned_text = cleaned_text.maketrans('','', keep_apos_com)

    # not removing them, tranformer models do not need them removed for tokenization --- getting rid of stop words
    # stop_words = set(stopwords.words('english'))
    # cleaned_text = ' '.join(word for word in cleaned_text.split() if word not in stop_words)

    # getting rid of caps - there are business names, poss keep them
    # cleaned_text = cleaned_text.lower()



    # making sure that there is only 1 space between words
    cleaned_text = ' '.join(cleaned_text.split())




    return cleaned_text





def cleaned_json(old_paths, new_paths):
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
    # count = 0
    for json_path, new_path in zip(old_paths, new_paths):

        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            json_data = pd.DataFrame(json_data)
            # print(f'\n{json_data.columns}\n')
            # print(f'\n{json_data.rows}\n')
            # json_data.info()


            # if the num of cols == 0 return - image file
            # add col has_image == true and send file straight to new path

            if 'type' not in json_data.columns: # considering .empty to use
                # an image file as the json is empty
                # altered_json_data = json_data.copy()
                altered_json_data = pd.DataFrame([{'input': 'This file contains an image'}])
                altered_json_data = altered_json_data.to_dict(orient='records')
                with open(new_path, 'w', encoding='utf-8') as f_new:
                    for line in altered_json_data:
                        f_new.write(json.dumps(line, ensure_ascii=False) + '\n')
                    # json.dump(altered_json_data, fnew, indent=2)


            else:
                # applies cleaner func to all relevant sections
                altered_json_data = pd.DataFrame()

                # text = value and section
                if (json_data['type'] == 'text').any():
                    json_text_data = json_data[(json_data['type'] == 'text') & (json_data['value'].apply(lambda x: cleaner_func(x))) & (json_data['section'].apply(lambda x: cleaner_func(x)))] ###
                    altered_json_data = pd.concat([altered_json_data, json_text_data], ignore_index=True)

                # header = value and section
                if (json_data['type'] == 'header').any():
                    json_header_data = json_data[(json_data['type'] == 'header') & (json_data['value'].apply(lambda x: cleaner_func(x))) & (json_data['section'].apply(lambda x: cleaner_func(x)))] ####
                    altered_json_data = pd.concat([altered_json_data, json_header_data], ignore_index=True)


                # checkbox = question and section
                if (json_data['type'] == 'checkbox').any():
                    json_checkbox_data = json_data[(json_data['type'] == 'checkbox') & (json_data['question'].apply(lambda x: cleaner_func(x))) & (json_data['section'].apply(lambda x: cleaner_func(x)))]
                    altered_json_data = pd.concat([altered_json_data, json_checkbox_data], ignore_index=True)

                # image = section
                if (json_data['type'] == 'image').any():
                    json_image_data = json_data[(json_data['type'] == 'image') & (json_data['section'].apply(lambda x: cleaner_func(x)))] #####
                    altered_json_data = pd.concat([altered_json_data, json_image_data], ignore_index=True)

            # print(altered_json_data)

            #===========================================================================================
            # joining type into col for text as 'type:' ----- 'header: value, section, question' etc...
            #===========================================================================================
            # text_col = type - a,b,c - section - value/question - value


                joined_json_data = pd.DataFrame()
                if (altered_json_data['type'] == 'text').any():  # value and section
                    text_mask = altered_json_data['type'] == 'text'

                    altered_json_data.loc[text_mask, 'full_text'] = ('Text: ' + altered_json_data.loc[text_mask, 'section'].astype(str) + ' ' + altered_json_data.loc[text_mask, 'value'].astype(str))


                if (altered_json_data['type'] == 'header').any(): # value and section
                    header_mask = altered_json_data['type'] == 'header'

                    altered_json_data.loc[header_mask, 'full_text'] = ('Header: ' + altered_json_data.loc[header_mask, 'section'].astype(str) + ' ' + altered_json_data.loc[header_mask, 'value'].astype(str))


                if (altered_json_data['type'] == 'checkbox').any(): # question and section

                    checkbox_mask = altered_json_data['type'] == 'checkbox'

                    altered_json_data.loc[checkbox_mask, 'full_text'] = ('Checkbox: ' + altered_json_data.loc[checkbox_mask, 'section'].astype(str) + 'QUESTION:' + altered_json_data.loc[checkbox_mask, 'question'].astype(str)+ ' ANSWER:' + altered_json_data.loc[checkbox_mask, 'value'].astype(str))


                if (altered_json_data['type'] == 'image').any(): # section and number count
                    image_mask = altered_json_data['type'] == 'image'

                    altered_json_data.loc[image_mask, 'full_text'] = ('Image: ' + altered_json_data.loc[image_mask, 'section'].astype(str))

                joined_json_data['input'] = pd.DataFrame(altered_json_data['full_text'])
                # joined_json_data.insert(loc=0, column='Input', value='input') # adding the key

                # print(joined_json_data)


                # sending json to new file, after cleaning ---- using jsonl formatting
                joined_json_data = joined_json_data.to_dict(orient='records')
                with open(new_path, 'w', encoding='utf-8') as f:
                    for line in joined_json_data:
                        if line:
                            f.write(json.dumps(line, ensure_ascii=False) +'\n')
                print(f'CLEANED FILE SAVED AT: {new_path}')




                ##### may need to add NaN filter for rows that dont meet criteria - question - check other cols
                # if 'question' in altered_json_data.columns:
                #     altered_json_data = altered_json_data[(altered_json_data['type'] != 'checkbox') | (altered_json_data['question'].notna())]


                #========================================================
                #========================================================
                #========================================================
                #========================================================
                #========================================================
                #========================================================
                #========================================================

                # print(json_data['type'].value_counts())
                #
                # print(f'\n{json_data.columns}\n')
                # count += 1
        # break


    # print(f'count:\n{count}') ### 371 json empty
            #     print(json_data['type'].value_counts())

            # if type == 'checkbox' look for ['question'] and tokenize
            # send file to new path

            # if type == 'text' look for ['value'] and tokenize
            # ensure that vals like 'Planning Portal Reference: PP-11298341 == Planning Portal Reference: <PPR number>

            # if type == 'header'
            # if type == 'image'


            # json_data = pd.DataFrame(json_data)
            # # print(json_data.head(20))
            # break


cleaned_json(old_json_file_paths, output_json_file_paths)



#===============================================================
#===============================================================
# tokenizing jsonl files with phi and gemma specific tokenizers
#===============================================================
#===============================================================


def gemma_tokenized_jsons(new_json_paths):

    '''
    - get input form the files
    - get the gemma tokenizer
    - run the input line by line through it
    - send it back to the same file with the embeddings
    '''

    # initialising the model
    tokenizer = gm.text.Gemma3Tokenizer()


    # going through all paths
    for json_path in new_json_paths:
        tokenized = []
        # going through each file by line
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line: # no empty lines , messes with model - ignore, error was elsewhere
                    object = json.loads(line)
                    text = object['input']

                    # converting to ids for training and storing them
                    token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
                    tokenized.append({'input_ids': token_ids})

        # problem path
        # print (f'PROBLEM HERE{json_path}')
        # overwriting original file with new embeddings
        with open(json_path, 'w', encoding='utf-8') as f:
            for token in tokenized:
                f.write(json.dumps(token, ensure_ascii=False) + '\n')
        print(f'GEMMA TOKENIZED FILE SAVED AT: {json_path}')


        # break



            # json_data = json.load(f)
            # json_data = pd.DataFrame(json_data)
            # # print(f'\n{json_data.columns}\n')
            # # print(f'\n{json_data.rows}\n')
            # json_data.info()








def phi_tokenized_jsons(new_json_paths):
    '''
    phi 3 mini tokenizer
    '''

    # initialising the model
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-4k-instruct")

    # going through all paths
    for json_path in new_json_paths:
        tokenized = []
        # going through each file by line
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line:  # no empty lines , messes with model - ignore, error was elsewhere
                    object = json.loads(line)
                    text = object['input']


                    encoder = tokenizer(text, add_special_tokens=True)
                    tokenized.append({'input_ids': encoder['input_ids'], 'attention_mask':encoder['attention_mask']})

        with open(json_path, 'w', encoding='utf-8') as f:
            for token in tokenized:
                f.write(json.dumps(token, ensure_ascii=False) + '\n')
        print(f'PHI TOKENIZED FILE SAVED AT: {json_path}')





#==========================
#   buffer
#==========================

# # ensuring time to update
# print(f'\n...Please wait while files are tokenized...\n')
# time.sleep(3)

#============================================================================================
#============================================================================================
#============================================================================================
#============================================================================================
#============================================================================================
#============================================================================================
#============================================================================================
# comment out one below to change model tokenization embeddings for model ( gemma or gpt)
# comment out one below to change model tokenization embeddings for model ( gemma or gpt)
# comment out one below to change model tokenization embeddings for model ( gemma or gpt)
# comment out one below to change model tokenization embeddings for model ( gemma or gpt)
# comment out one below to change model tokenization embeddings for model ( gemma or gpt)
# comment out one below to change model tokenization embeddings for model ( gemma or gpt)
#============================================================================================
#============================================================================================
#============================================================================================
#============================================================================================
#============================================================================================
#============================================================================================
#============================================================================================
#============================================================================================
#============================================================================================


chosen_model = input('Please select:\n- 1 for gemma file tokenization \n- 2 for phi file tokenization\n Your input: ')

if chosen_model == 1:
    print(f'\n...Please wait while files are tokenized...\n')
    time.sleep(5)
    gemma_tokenized_jsons(output_json_file_paths)
elif chosen_model == 2:
    print(f'\n...Please wait while files are tokenized...\n')
    time.sleep(5)
    phi_tokenized_jsons(output_json_file_paths)
    pass
else:
    print('invalid input, tokenizing with gemma:')
    print(f'\n...Please wait while files are tokenized...\n')
    time.sleep(5)
    gemma_tokenized_jsons(output_json_file_paths)



