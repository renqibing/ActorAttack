import os
import json
from datetime import datetime
from utils import gpt_call, get_client, read_data_from_json, read_prompt_from_file
from concurrent.futures import ThreadPoolExecutor
from config import DatasetConfig
import argparse
class DatasetBuilder:
    def __init__(self, config: DatasetConfig):
        self.attack_data_file = config.attack_data_file
        self.safe_prompt = read_prompt_from_file(config.safe_prompt)
        self.safe_model_name = config.safe_model_name
        self.data = read_data_from_json(config.attack_data_file)['data']
        self.client = get_client(config.safe_model_name)
        self.process_full = True
        self.config = config
    
    def preprocess(self, data_list):
        harm_dialog_list = []
        for data in data_list:
            for attempts in data['attempts']:
                dialog_hist = attempts['dialog_hist']
                is_succeed = False
                for i in range(0, len(dialog_hist),2):
                    if dialog_hist[i+1]['score'] == 5:
                        print(f"build: {data['instruction']}\n{dialog_hist[i]['content']}")
                        dialog_hist_truc = dialog_hist[:i+2]
                        is_succeed = True
                        if not self.process_full:
                            break
                if is_succeed:
                    harm_dialog_list.append({'instruction': data['instruction'], 'conversations': dialog_hist_truc})
        return harm_dialog_list

    def process_thread(self, data):
        prev_qs = ""
        user_q = data["instruction"]
        convs = data["conversations"]
        for i in range(0, len(convs), 2):
            if i == len(convs) - 2:
                cur_q, cur_a = convs[i]["content"], convs[i+1]["content"]
            else:
                prev_q = convs[i]["content"]
                prev_qs += f"{i//2}: {prev_q}"
        safe_prompt = self.safe_prompt.format(user_q=user_q, prev_qs=prev_qs, cur_q=cur_q, cur_a=cur_a)
        safe_a = gpt_call(self.client, safe_prompt, model_name=self.safe_model_name)
        data["conversations"][-1]["content"] = safe_a
        return data
    
    def process_thread_full(self, data):
        prev_qs = ""
        user_q = data["instruction"]
        convs = data["conversations"]
        for i in range(0, len(convs), 2):
            prev_q = convs[i]["content"]
            prev_qs += f"{i//2}: {prev_q}"
            if convs[i+1]['score'] == 5:
                cur_q, cur_a = convs[i]["content"], convs[i+1]["content"]
                safe_prompt = self.safe_prompt.format(user_q=user_q, prev_qs=prev_qs, cur_q=cur_q, cur_a=cur_a)
                data["conversations"][i+1]["content"] = gpt_call(self.client, safe_prompt, model_name=self.safe_model_name)
        return data
    
    def process(self, harm_dialog_list):
        safe_dialog_list = []
        with ThreadPoolExecutor(max_workers=50) as executor:
            if self.process_full:
                futures = executor.map(self.process_thread_full, harm_dialog_list)
            else:
                futures = executor.map(self.process_thread, harm_dialog_list)
            for res in futures:
                safe_dialog_list.append(res)
        return safe_dialog_list
    
    def postprocess(self, safe_dialog_list):
        post_dialog_list = []
        for data in safe_dialog_list:
            convs = data["conversations"]
            new_convs = []
            for i in range(0, len(convs), 2):
                new_convs.append(
                    {
                        "role": "user", "content": convs[i]["content"]
                    })
                new_convs.append(
                    {
                        "role": "assistant", "content": convs[i+1]["content"],
                    })
            post_dialog_list.append({'instruction':data["instruction"], 'messages':new_convs})
        return post_dialog_list
    
    def build(self):
        harm_dialog_list = self.preprocess(self.data)
        safe_dialog_list = self.process(harm_dialog_list)
        post_dialog_list = self.postprocess(safe_dialog_list)
        if not os.path.exists('./safety_dataset_result'):
            os.makedirs('./safety_dataset_result')
        with open(f'./safety_dataset_result/{self.safe_model_name.split("/")[-1].replace(".", "-")}_{datetime.now()}.json', 'w', encoding='utf-8') as f:
            json.dump(post_dialog_list, f, ensure_ascii=False, indent=4)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Construct Dataset')
    parser.add_argument("--attack_data_file", type=str, default='attack_result/deepseek-chat_350_2024-10-13 03:47:59.420645.json', help="Number of questions.")
    args = parser.parse_args()
    
    config = DatasetConfig(
        attack_data_file=args.attack_data_file
    )
    
    builder = DatasetBuilder(config)
    builder.build()