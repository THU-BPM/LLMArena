import requests
import os
import json
import shutil

def get_models():
    url = 'http://172.26.1.16:31251/get_model_list'  # 修改为您的服务器地址和端口
    response = requests.post(url)

    if response.status_code == 200:
        data = response.json()
        models = data.get('models', [])
        print('Models:', models)
        return models
    else:
        print('Error:', response.status_code)
        return []

def setup_directories(envs, models):
    for env in envs:
        for model in models:
            model_name = model.split('/')[-1].replace(".","")
            model_dir = f"{env}/{model_name}/"
            
            # 创建模型目录
            if not os.path.exists(model_dir) :
                os.makedirs(model_dir)
            
            opponent_path = os.path.join(model_dir, 'opponent.py')
            # 删除已存在的 opponent.py
            if os.path.exists(opponent_path) or os.path.islink(opponent_path):
                os.remove(opponent_path)
            
            # 创建新的链接
            os.symlink(os.path.abspath(f"{env}/opponent.py"), opponent_path)

            # 更新 info.json
            info_path = os.path.join(env, 'gpt-35-turbo', 'info.json')
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    info = json.load(f)
                info["api_base"] = "http://172.26.1.16:31251/v1"
                info["model_name"] = model
                info['api_key'] = "sk-"
                
                with open(os.path.join(model_dir, 'info.json'), 'w') as f:
                    json.dump(info, f, indent=4)
                print(f"Generate model: {model_dir}")

envs = ['chess_v6','connect_four_v3','go_v5','hanabi_v5','texas_no_limit_v6','tictactoe_v3']
models = get_models()
setup_directories(envs, models)
