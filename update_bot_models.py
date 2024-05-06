import json

if __name__ == "__main__":
	with open('models/config.json', 'r') as f:
		models = json.load(f)

	new_cfg = {}
	new_cfg['users_config'] = {}
	new_cfg['models_config'] = models
	with open('tgbot/bot_config.json', 'w') as f:
		json.dump(new_cfg, f)