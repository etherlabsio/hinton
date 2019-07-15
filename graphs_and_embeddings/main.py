import pickle 
from text_utils import *
from graph_utils import GraphBuilder
from candidate_generator import CandidateKPExtractor
from feature_extractor import *
import utils
import networkx as nx

from sanic import Sanic
from sanic.response import json
import json as js

if __name__ == '__main__':

	embedding_dict_path = "data/ether_engg_embedding_dict.pkl"
	word_weight_dict_path = "data/ether_engg_weight_dict.pkl"
	stop_words_path = 'data/long_stopwords.txt'

	app = Sanic()
	@app.route('/rankKeyphrases/', methods=["POST"])

	async def identifyPIMs(request):

		reqData = request.json
		input_text = reqData['segments'][0]['originalText']
		graph_name = reqData['contextId']

		with open(stop_words_path) as f:
		    stop_words = f.read().splitlines()

		embedding_dict = pickle.load(open(embedding_dict_path,'rb'))
		word_weight_dict = pickle.load(open(word_weight_dict_path,'rb'))

		p_list = [1,3,5]

		prep_obj = PreProcessText(input_text)
		filtered_text, sent_list = prep_obj.getFilteredTextSegment()

		cand_gen = CandidateKPExtractor(filtered_text, stop_words)
		candidate_list = cand_gen.getCandidatePhrases()


		graphBuilder = GraphBuilder()
		graphObj = graphBuilder.buildKPGraph(candidate_list)

		featObj = FeatureExtractor(embedding_dict, word_weight_dict, stop_words, p_list)

		sent_feat_list = featObj.getTextFeatList(sent_list)
		node_feat_list = featObj.getTextFeatList(candidate_list)

		nodeFeatureDict = dict(zip(candidate_list,node_feat_list))
		sentFeatureDict = dict(zip(sent_list,sent_feat_list))
		nodeWeightDict = getKPWeight(nodeFeatureDict, sentFeatureDict)

		#add node properties and save save to s3
		graphObj = graphBuilder.addNodeProperties(graphObj,nodeWeightDict, 'node_weight')
		graphObj = graphBuilder.addNodeProperties(graphObj,nodeFeatureDict, 'node_features')

		graphBuilder.uploadGraphObject(graphObj,graph_name)

		weighted_graph = graphBuilder.formEdgesWithNodeSimilarity(graphObj,nodeFeatureDict, nodeWeightDict)

		pagerankWeights = graphBuilder.getPageRankNodeWeights(weighted_graph)
		cosineWeights = {k:v for k, v in nodeWeightDict.items() if k in weighted_graph.nodes()}

		#aggregate pagerank and cosine similarity weights to get the 
		node_scores_page_rank = getDFFromDict(pagerankWeights,['key-phrase','pr_score']).sort_values(by='pr_score',ascending=False)
		node_scores_cosine = getDFFromDict(cosineWeights,['key-phrase','cs_score']).sort_values(by='cs_score',ascending=False)
		node_scores_page_rank['pr_score'] = node_scores_page_rank['pr_score'].apply(lambda x: x/max(node_scores_page_rank['pr_score']))
		node_scores_cosine['cs_score'] = node_scores_cosine['cs_score'].apply(lambda x: x/max(node_scores_cosine['cs_score']))

		df_final_scores = pd.merge(node_scores_page_rank,node_scores_cosine,on='key-phrase')
		df_final_scores['final_score'] = df_final_scores['pr_score']+df_final_scores['cs_score']
		df_final_scores = df_final_scores.sort_values(by='final_score', ascending=False)

		json_df = df_final_scores.to_json(orient='records')
		print(graphObj.nodes())

		return json({"id":1,"keyphrase_scores": js.loads(json_df)}) 

	app.run(host='0.0.0.0', port=8080)