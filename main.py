from preattack import PreAttack
from inattack import InAttack
from config import PreAttackConfig, InAttackConfig
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ActorAttack')
    # actor
    parser.add_argument("--questions", type=int, default=1, help="Number of questions.")
    parser.add_argument("--actors", type=int, default=3, help="Number of actors for each question.")
    parser.add_argument("--behavior", default="./data/harmbench.csv", help="Path of harmful behaviors CSV file.")
    # attack
    parser.add_argument("--attack_model_name", type=str, default="gpt-4o", help="Attack Model name.")
    parser.add_argument("--target_model_name", type=str, default="gpt-4o", help="Target Model name.")
    parser.add_argument("--early_stop", type=bool, default=True, help="early stop if the judge LLM yields success.")
    parser.add_argument("--dynamic_modify", type=bool, default=True, help="apply dynamic modification.")
    args = parser.parse_args()
    
    pre_attack_config = PreAttackConfig(
        model_name=args.attack_model_name,
        actor_num=args.actors,
        behavior_csv=args.behavior)
    pre_attacker = PreAttack(pre_attack_config)
    
    pre_attack_data_path = pre_attacker.infer(args.questions)
    print(f"pre-attack data path: {pre_attack_data_path}")

    in_attack_config = InAttackConfig(
        attack_model_name = args.attack_model_name,
        target_model_name = args.target_model_name,
        pre_attack_data_path = pre_attack_data_path,
        early_stop = args.early_stop,
        dynamic_modify = args.dynamic_modify
    )
    
    in_attacker = InAttack(in_attack_config)
    final_result_path = in_attacker.infer(args.questions)
    print(f"final attack result path: {final_result_path}")
