import os
import json
import pandas as pd
from datetime import datetime
from config import PreAttackConfig
from concurrent.futures import ThreadPoolExecutor
from utils import gpt_call, read_prompt_from_file, parse_json, check_file, get_client, gpt_call_append

class PreAttack:
    def __init__(self, config: PreAttackConfig):
        # file path
        self.behavior_csv = check_file(config.behavior_csv)
        self.extract_prompt = read_prompt_from_file(config.extract_prompt)
        self.network_prompt = read_prompt_from_file(config.network_prompt)
        self.actor_prompt = read_prompt_from_file(config.actor_prompt)
        self.query_prompt = read_prompt_from_file(config.query_prompt)
        self.more_actor_prompt = read_prompt_from_file(config.more_actor_prompt)
        self.format_prompt = read_prompt_from_file(config.json_format_prompt)
        # actor_num
        self.actor_num = config.actor_num
        df = pd.read_csv(self.behavior_csv)
        self.org_data = df['Goal'].tolist()
        # model
        self.model_name = config.model_name
        self.client = get_client(config.model_name)
        self.config = config
        
    def extract_harm_target(self, org_query):
        prompt = self.extract_prompt.format(org_query=org_query)
        for _ in range(5):
            try:
                res = gpt_call(self.client, prompt, model_name=self.model_name)
                data = parse_json(res)
                return data['target'], data['details']
            except Exception as e:
                print("Error in extract_harm_target:", e)
                continue
        return {}, {}
    
    def get_actors(self, harm_target):
        network_prompt = self.network_prompt.format(harm_target=harm_target)
        
        resp, dialog_hist = gpt_call_append(self.client, self.model_name, [], network_prompt)
        
        num_string = '10 actors' if self.actor_num > 10 else f"{self.actor_num} actors"
        actor_prompt = self.actor_prompt.format(num_string=num_string)
        more_actor_prompt = self.more_actor_prompt
        actors = []
        for _ in range(3):
            try:
                resp, dialog_hist = gpt_call_append(self.client, self.model_name, dialog_hist, actor_prompt)
                data = parse_json(resp)
                for item in data['actors']:
                    if item['actor_name'] not in [actor_item['actor_name'] for actor_item in actors]:
                        actors.append(item)
                dialog_hist = dialog_hist[:-2]
                if len(actors) >= self.actor_num:
                    return actors[:self.actor_num], dialog_hist
                resp, dialog_hist = gpt_call_append(self.client, self.model_name, dialog_hist, more_actor_prompt)
            except Exception as e:
                print("Error in get_actors:", e)
        
        return actors, dialog_hist
    
    def get_init_queries(self, harm_target, actor):
        actor_name = actor['actor_name']
        relationship = actor['relationship']
        query_prompt = self.query_prompt.format(harm_target=harm_target, actor_name=actor_name, relationship=relationship)
        for _ in range(5):
            try:
                query_resp = gpt_call(self.client, query_prompt, model_name=self.model_name)
                format_prompt = self.format_prompt.format(resp=query_resp)
                json_output = gpt_call(self.client, format_prompt, model_name=self.model_name)
                data = parse_json(json_output)
                queries = []
                for item in data["questions"]:
                    queries.append(item["question"])
                return queries, query_resp
            except Exception as e:
                print("Error in get_queries:", e)
                continue
        return queries, query_resp
    
    def infer_single(self, org_query: str):
        harm_target, query_details = self.extract_harm_target(org_query)
        actors, network_hist = self.get_actors(harm_target)
        data_list = []
        for actor in actors:
            try:
                queries, query_chain = self.get_init_queries(harm_target, actor)
                data_list.append({"actor":actor, "queries":queries, "query_chain":query_chain})
            except Exception as e:
                print(f"Error in infer_single: {actor}\n {e}")
                continue
        return {"instruction": org_query, "harm_target":harm_target, "query_details":query_details,"network_hist":network_hist, "actors":data_list}
    
    def infer(self, num=-1):
        json_data = self.config.__dict__
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = list(executor.map(self.infer_single, self.org_data[:num]))
        json_data["data"] = futures
        if not os.path.exists('./pre_attack_result'):
            os.makedirs('./pre_attack_result')
        file_path = f'./pre_attack_result/queries_for_{self.model_name.split("/")[-1].replace(".", "-")}_{num}_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.json'
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        return file_path
            
if __name__ == '__main__':
    config = PreAttackConfig()
    attacker = PreAttack(config)
    attacker.infer(num = 2)