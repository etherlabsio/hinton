from model_loader import ModelLoader
from text_utils import *

config_path = 'artifacts/bert_config.json'
model_path = 'artifacts/model.bin'

model_loader = ModelLoader(config_path, model_path)
model = model_loader.loadModel()

text1 = 'and with that in mind, i tried many approaches and frameworks for implementing the same pattern: single page applications (spa)'
text2 = 'how do you get your message to the right audience and do it effectively? how do you boost visibility and increase sales while sustaining a profit with a converting offer? today, with so much vying for our attention from social media, to search engine optimization, blogging and pay per click advertising, it is easy to see why most are ready to pull their hair out'


print(getKPBasedSimilarity(text1,text2, model))

text1_feat_set = getBERTFeatures(model, text1, attn_head_idx=-1)
text2_feat_set = getBERTFeatures(model, text2, attn_head_idx=-1)

print(getKPBasedSimilarity(text1,text2,model,text1_feat_set,text2_feat_set))
