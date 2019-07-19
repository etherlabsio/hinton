from bert_utils import *
import torch
import pickle


class ModelLoader(object):

	def __init__(self, config_path, model_path):

		self.config = BertConfig.from_json_file(config_path)
		self.state_dict = torch.load(model_path,map_location='cpu')

	def loadModel(self):

		tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		model = BertForPreTrainingCustom(self.config)
		model.load_state_dict(self.state_dict)
		model.eval()

		return model