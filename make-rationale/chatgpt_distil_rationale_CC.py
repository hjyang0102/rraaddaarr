import json
import time
from tqdm import tqdm
import openai
from openai import OpenAIError

import re
from extract_file_name_part import extract_filename_part


openai.api_key = '___'
howmanyrationaleDoyouNeed = 10

input_for_GPT_file = 'train_data_new.json'
output_from_GPT_file = "train_data_new_augmented_CC.json"

chatgptmodel = 'gpt-3.5-turbo'
chatgptmodel_list = ['gpt-3.5-turbo-1106']
chatgptmodel_num = 0


stopped_convID = '7974-2'
flag = 1

outout_dict = {}
updated_data = []

with open(input_for_GPT_file, 'r', encoding='utf-8') as f:
    data = json.load(f)
    split_count = 0
    total_conv_num = len(data['data'])

    print(total_conv_num)


    for conv_info in tqdm(data['data']):

        if conv_info['ID'] == stopped_convID :
            flag = 0
            split_count = split_count + 1
            continue
        if flag == 1:
            split_count = split_count + 1
            continue


        prompt_next_response_not_included = 'Instruction:The conversation is an interaction between a user and a movie recommender. What contextual conditions (i.e., watching purpose or watching situation) will affect to a movie recommendation based on the conversation. (e.g., time pass, to get motivated, for entertainment, emotional impact, to acquire information, with family, with a partner, with friends, for a sleepover, for a movie club, for a family gathering, etc.) Do not generate a lengthy answer.'

        prompt_next_response_not_included = prompt_next_response_not_included + conv_info['context']
        prompt_next_response_included = prompt_next_response_included + conv_info['context']

        prompt_next_response_included = prompt_next_response_included + '\nNext_Response: ' + conv_info['next_response']




        list_rationale_CC_nextresponse_O = []
        list_rationale_CC_nextresponse_X = []

        try:
            response = openai.ChatCompletion.create(model=chatgptmodel,
                messages=[
                        {"role": "system", "content": prompt_next_response_included}
                    ]
            )
        except OpenAIError as e:
            time.sleep(15)
            print('stopped and go')
            chatgptmodel_num = chatgptmodel_num + 1
            chatgptmodel = chatgptmodel_list[chatgptmodel_num]

            response = openai.ChatCompletion.create(model=chatgptmodel,
                messages=[
                        {"role": "system", "content": prompt_next_response_included}
                    ]
            )


        answer = response["choices"][0]["message"]["content"]
        list_rationale_CC_nextresponse_O.append(answer)
        conv_info['list_rationale_CC_nextresponse_O'] = list_rationale_CC_nextresponse_O


        try:
            response = openai.ChatCompletion.create(model=chatgptmodel,
            messages=[
                    {"role": "system", "content": prompt_next_response_not_included}
                ]
        )
        except OpenAIError as e:
            time.sleep(15)
            print('stopped and go')
            chatgptmodel_num = chatgptmodel_num + 1
            chatgptmodel = chatgptmodel_list[chatgptmodel_num]

            response = openai.ChatCompletion.create(model=chatgptmodel,
            messages=[
                    {"role": "system", "content": prompt_next_response_not_included}
                ]
            )

        answer = response["choices"][0]["message"]["content"]
        list_rationale_CC_nextresponse_X.append(answer)
        conv_info['list_rationale_CC_nextresponse_X'] = list_rationale_CC_nextresponse_X

        updated_data.append(conv_info)


        if split_count % 300 == 0:
            file_name = extract_filename_part(output_from_GPT_file) +'_' + str(split_count) + '.json'
            outout_dict['version'] = "0.1.0"
            outout_dict['data'] = updated_data

            print(file_name)
            with open(file_name, "w") as f:
                json.dump(outout_dict, f, indent=4)

            outout_dict = {}
            updated_data = []
        elif split_count == total_conv_num-1:
            print("the end")
            print(conv_info["ID"])

            file_name = extract_filename_part(output_from_GPT_file) + str(split_count) + '.json'
            outout_dict['version'] = "0.1.0"
            outout_dict['data'] = updated_data

            print(file_name)
            with open(file_name, "w") as f:
                json.dump(outout_dict, f, indent=4)

            outout_dict = {}
            updated_data = []

        split_count = split_count + 1
