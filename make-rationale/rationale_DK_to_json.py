answer = '"genre": "mystery"'
import json
import tqdm
import re

input_for_GPT_file = 'new_generated_data.json'


def extract_only_answer_from_rationale_DK(input_string):
    pattern = r'"(\w+)" *: *"(.*?)"'
    match = re.match(pattern, input_string)
    if match:
        attribute, value = match.groups()  # Extract attribute and value
        return value  # Return the extracted value
    else:
        return ""  # Return empty string if the format is incorrect

updated_data = {}
with open(input_for_GPT_file, 'r', encoding='utf-8') as f:
    data = json.load(f)
    count = 0
    for conv_id, conv_info in data.items():
        list_rationale_DK_raw = []
        list_rationale_DK_processed = []
        for i in range(10):
            list_rationale_DK_raw.append(answer)

            processed_answer = extract_only_answer_from_rationale_DK(answer)
            list_rationale_DK_processed.append(processed_answer)


        conv_info['rationale_DK_raw'] = list_rationale_DK_raw
        conv_info['rationale_DK_processed'] = list_rationale_DK_processed
        updated_data[conv_id] = conv_info

with open("data_augmented_DK.json", "w") as f:
    json.dump(updated_data, f, indent=4)
