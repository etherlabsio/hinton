from scipy import spatial
import numpy as np
import pandas as pd
import re
import json
import pickle

def CosineSim(vec1,vec2):
	return 1-spatial.distance.cosine(vec1,vec2)

def validVectorCheck(vec):
	if len(np.unique(vec))==1:
		return False
	else: 
		return True

def stripSpaces(text):
	text = re.sub('\s+', ' ',text) 
	return text.strip()

def getDFFromDict(dict_,col_names):
    dict_df = pd.DataFrame.from_dict(dict_,orient='index').reset_index()
    dict_df.columns = col_names
    return dict_df

def pickleObject(obj, protocol = pickle.HIGHEST_PROTOCOL):
	pickle_dumps = pickle.dumps(obj=obj, protocol=protocol)
	return pickle_dumps