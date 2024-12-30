import json
import time
from tqdm import tqdm
from openai import OpenAI
#import openai
from openai import OpenAIError
import collections
import random


import re
from extract_file_name_part import extract_filename_part



client = OpenAI(
    api_key = '___',
)

howmanyrationaleDoyouNeed = 3# 4 #10
movie_dict_file = 'movies_with_age_rating.json'



input_for_GPT_file = 'train_data_new.json'
output_from_GPT_file = 'train_data_new_augmented_DK.json'




chatgptmodel = 'gpt-3.5-turbo'
chatgptmodel_list = ['gpt-3.5-turbo', 'gpt-3.5-turbo-0125', 'gpt-3.5-turbo-0301', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-1106','gpt-3.5-turbo-16k','gpt-3.5-turbo-16k-0613']
chatgptmodel_num = 0

conv_ID_20000 = '3005-1'
stopped_convID = '10483-2'
flag = 1
mothaemuckgetdda_max = 375
mothaemuckgetdda = 0

with open('entity2id_new.json', 'r', encoding='utf-8') as f:
    entity2id = json.load(f)

movie_data = json.load(open(movie_dict_file,'r'))

def extract_only_answer_from_rationale_DK(input_string):
    pattern = r'"(\w+)" *: *"(.*?)"'
    match = re.match(pattern, input_string)
    if match:
        attribute, value = match.groups()  # Extract attribute and value
        return value  # Return the extracted value
    else:
        return ""  # Return empty string if the format is incorrect


def processed_attr_text_to_entityID(list_rationale_DK_processed):
    if len(list_rationale_DK_processed) == 0:
        return [24635, 24635, 24635, 24635, 24635, 24635, 24635, 24635, 24635, 24635]

    temp_list = []
    for element in list_rationale_DK_processed:
        try:
            temp_list.append(entity2id[element])
        except KeyError:
            continue
    return temp_list


def fill_list_to_ten(input_list):
    if len(input_list) == 0:
        return [24635, 24635, 24635, 24635, 24635, 24635, 24635, 24635, 24635, 24635]


    if len(input_list) >= 10:
        return input_list

    while len(input_list) < 10:
        input_list.append(random.choice(input_list))

    return input_list


def most_frequent_element(input_list):
    counter = collections.Counter(input_list)
    most_common_element = counter.most_common(1)[0][0]  # Extract element from the tuple
    return most_common_element


outout_dict = {}
updated_data = []

with open(input_for_GPT_file, 'r', encoding='utf-8') as f:
    data = json.load(f)
    split_count = 0
    total_conv_num = len(data['data'])
    print(total_conv_num)

    for conv_info in tqdm(data['data']):

        if conv_info['ID'] == conv_ID_20000:
            exit()

        if conv_info['ID'] == stopped_convID :
            flag = 0
            split_count = split_count + 1
            continue
        if flag == 1:
            split_count = split_count + 1
            continue

        prompt = '/*Task Prompt*/ Instruction: The conversation is an interaction between a user and a movie recommender. The gold response is based on the given conversation, utilizing the domain knowledge. I will give you examples Refer to the examples. Question: Which attribute in the domain knowledge most affected to the recommendation in the gold response? And generate an answer to the question, following the format requirements. Formant Requirements: Identify the primary influencing attribute. Present your answer in the following format. * If the genre affected most, which genre? Present your answer: "genre": [genre]. * If the director affected most, which director? Present your answer: "director": [director]. * If the writer affected most, which writer? Present your answer: "writer": [writer]. * If the writer affected most, which star? Present your answer: "star": [star]. * If the age_rating affected most, what is the age_rating? Present your answer: "age_rating": [age_rating]. '
        prompt = prompt + conv_info['context']
        prompt = prompt + '\nNext_Response: ' + conv_info['next_response']

        prompt = prompt + '\n \nDomain_Knowledge: \n'

        if len(conv_info['context_movies']) > 0 :
            for context_a_movie_id, context_a_movie_name in conv_info['context_movies'].items():
                try:
                    prompt = prompt + str(movie_data[context_a_movie_id]) +'\n'
                except KeyError:
                    prompt = prompt + 'N/A \n'
        else:
            prompt = prompt + 'N/A \n'

        if len(conv_info['next_response_movies']) > 0 :
            for context_a_movie_id, context_a_movie_name in conv_info['next_response_movies'].items():
                try:
                    prompt = prompt + str(movie_data[context_a_movie_id]) +'\n'
                except KeyError:
                    prompt = prompt + 'N/A \n'
        else:
            prompt = prompt + 'N/A \n'

        #print
        prompt = prompt + '\n \nAnswer: '


        list_rationale_DK_raw = []
        list_rationale_DK_processed = []
        preferred_attrs = []
        most_preferred_attr = []

        for i in range(howmanyrationaleDoyouNeed):
            try:
                response = client.chat.completions.create(model=chatgptmodel,
                    messages=[
                            {"role": "system", "content": prompt}
                        ]
                )
            except OpenAIError as e:
                time.sleep(5)
                mothaemuckgetdda = mothaemuckgetdda + 1

                if mothaemuckgetdda > mothaemuckgetdda_max:
                    file_name = extract_filename_part(output_from_GPT_file) + str(split_count) + '.json'
                    outout_dict['version'] = "0.1.0"
                    outout_dict['data'] = updated_data
                    print("----mothaemuckgetdda----")
                    print(file_name)
                    with open(file_name, "w") as f:
                        json.dump(outout_dict, f, indent=4)
                    exit()


                if chatgptmodel_num == 6:
                    chatgptmodel_num = 0
                    #time.sleep(60)
                else:
                    chatgptmodel_num = chatgptmodel_num + 1
                chatgptmodel = chatgptmodel_list[chatgptmodel_num]

                response = client.chat.completions.create(model=chatgptmodel,
                    messages=[
                            {"role": "system", "content": prompt}
                        ]
                )


            answer = response.choices[0].message.content
            list_rationale_DK_raw.append(answer)

            processed_answer = extract_only_answer_from_rationale_DK(answer)
            list_rationale_DK_processed.append(processed_answer)

        conv_info['rationale_DK_raw'] = list_rationale_DK_raw
        conv_info['rationale_DK_processed'] = list_rationale_DK_processed


        preferred_attrs = fill_list_to_ten(processed_attr_text_to_entityID(list_rationale_DK_processed))

        most_preferred_attr = most_frequent_element(preferred_attrs)


        conv_info['preferred_attrs'] = preferred_attrs
        conv_info['most_preferred_attr'] = most_preferred_attr


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
