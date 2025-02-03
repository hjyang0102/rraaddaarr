import json
import re

# input_data_file = 'train_data.jsonl'
# output_data_file = "train_data_new.json"

# input_data_file = 'test_data.jsonl'
# output_data_file = "test_data_new.json"

input_data_file = 'test_data.jsonl'
output_data_file = "test_data_new.json"



with open('entity2id_new.json', 'r', encoding='utf-8') as f:
    entity2id = json.load(f)

def extract_movie_items_in_text(text, movie_id_to_name_dict):
    """Extracts movie IDs and their corresponding movie names from a text.

    Args:
        text: The input text string.
        movie_id_to_name_dict: A dictionary mapping movie IDs to movie names.

    Returns:
        A dictionary mapping movie IDs (without the '@' prefix) to their corresponding movie names.
    """

    # Regular expression pattern to match movie references
    pattern = r"@(\d+)"

    movie_items = {}  # Dictionary to store extracted movie items
    temp_list = []

    def replace_match(matchobj):
        movie_id = matchobj.group(1)  # Extract movie ID from the match
        if movie_id in movie_id_to_name_dict:
            movie_name = movie_id_to_name_dict[movie_id]
            item_id_without_at = movie_id[0:]
            movie_items[item_id_without_at] = movie_name
            if movie_name in entity2id:
                temp_list.append(entity2id[movie_name])
            else:
                temp_list.append(24635)
        else:
            # Handle unknown movie IDs (optional: return original reference)
            item_id_without_at = movie_id[0:]
            movie_items[item_id_without_at] = matchobj.group()  # Return original reference

        # Return empty string to prevent further replacements
        return ""

    # Replace movie references in the text
    re.sub(pattern, replace_match, text)

    return movie_items, temp_list


def replace_movie_num_to_name(text, movie_dic):
    pattern = r"@(\d+)"

    def replace_match(matchobj):
        movie_id = matchobj.group(1)  # Extract the movie ID from the match
        if movie_id in movie_dic:
            return movie_dic[movie_id]
        else:
            # Handle unknown movie IDs (optional: return original reference)
            return matchobj.group(0)

    return re.sub(pattern, replace_match, text)



outout_dict = {}
updated_data = []

with open(input_data_file,'r',encoding='utf-8') as f:
    lines = f.readlines()

    for line in lines:
        
        # new_context = new_dialogue['context']
        # new_next_response = new_dialogue['next_response']
        # new_mentioned_movie_list = new_dialogue['mentioned_movie_list']
        # new_mentioned_movies_n_attrs = new_dialogue['mentioned_movies_n_attrs']



        dialog_info = json.loads(line) 
        mentioned_movie_dic = dialog_info['movieMentions']

        recommender_ID = dialog_info['respondentWorkerId']
        user_ID = dialog_info['initiatorWorkerId']

        messages = dialog_info['messages']
        messages_len = len(dialog_info['messages'])


        conversationId = dialog_info['conversationId']


        count = 0
        

        for i in range(messages_len):
            context = ''
            next_response = ''
            mentioned_movies_n_attrs = []

            if messages[i]['senderWorkerId'] == recommender_ID and len(messages[i]['item']) > 0 :  
            #if in the response of recommender,, at least one recommendation ,, previous context combo
                new_conv_data = {}

                for ii in range(i):
                    for movies_n_attrs in messages[ii]['thread']:
                        mentioned_movies_n_attrs.append(movies_n_attrs)

                    if messages[ii]['senderWorkerId'] == recommender_ID: 
                        #if recommender
                        context =  context + 'recommender:' + messages[ii]['text'] +'\n'                               
                    else: 
                        #if user
                        context =  context + 'user:' + messages[ii]['text'] +'\n'


                #context = context + ' ' + messages[ii]['text']
                #next_response = replace_movie_num_to_name(messages[i]['text'], mentioned_movie_dic)

                new_conv_Id = str(conversationId)+'-'+str(count)
                count = count + 1
                new_conv_data['ID'] = new_conv_Id
                new_conv_data['context'] = replace_movie_num_to_name(context, mentioned_movie_dic)

                context_movies, ID_inKG_list = extract_movie_items_in_text(context, mentioned_movie_dic)
                new_conv_data['context_movies'] = context_movies
                #new_conv_data['rec_item'] = ID_inKG_list

                new_conv_data['next_response'] = replace_movie_num_to_name(messages[i]['text'], mentioned_movie_dic)

                next_response_movies, ID_inKG_list = extract_movie_items_in_text(messages[i]['text'], mentioned_movie_dic)
                new_conv_data['next_response_movies'] = next_response_movies
                new_conv_data['rec_item'] = ID_inKG_list[0]
                new_conv_data['rec_items'] = ID_inKG_list

                new_conv_data['mentioned_movie_dic'] = mentioned_movie_dic
                new_conv_data['mentioned_movies_n_attrs'] = mentioned_movies_n_attrs

                temp_list = []
                count = 0
                for movie_n_attr in mentioned_movies_n_attrs:
                    
                    if count == 20:
                        break
                    else:
                        count = count + 1 
                    temp_list.append(entity2id[movie_n_attr])
                
                loop_count = 20 - len(temp_list) 
                for i in range(loop_count):
                    temp_list.append(24635)

                new_conv_data['mentioned_movies_n_attrs_list'] = temp_list

                updated_data.append(new_conv_data)




outout_dict['version'] = "0.1.0"
outout_dict['data'] = updated_data

with open(output_data_file, "w") as f:
    json.dump(outout_dict, f, indent=4)


