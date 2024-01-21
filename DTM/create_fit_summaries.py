import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import scipy.io
from utilities import *

def main():

	# task_name = 'opto'
	# rois = ['M1','Thalamus','Insula']
	# nRois = len(rois)

	task_name = 'math1'
	rois = ['L SPL', 'R SPL', 'L IPS', 'R IPS', 'L Insula', 'R Insula',
            'dACC', 'SMA', 'L MFG', 'R MFG', 'R VLPFC', 'L VLPFC',
            'R FG', 'L FG', 'R Hipp', 'L Hipp']
	nRois = len(rois)

	gamma_file = 'gammas_out/%s_k20_group_gammas.pkl'%(task_name)
	model = 'fit_models/dtm_fit_%s_20k'%(task_name)

	with open(gamma_file, "rb") as input_file:
		gammas = pickle.load(input_file)

	indiv_gamma_file = 'gammas_out/%s_indiv_gammas.pkl'%(task_name)

	if "math" in task_name:
		with open(indiv_gamma_file, "rb") as input_file:
			indiv_gammas = pickle.load(input_file)
		np.save("model_output/%s/alpha/indiv_gammas"%task_name,indiv_gammas)

	np.savetxt("../for_paper/model_output/%s/alpha/gammas.csv"%task_name,gammas.gammas)
	np.savetxt("../for_paper/model_output/%s/alpha/topic_sums.csv"%task_name,gammas.topic_sums)
	np.savetxt("../for_paper/model_output/%s/alpha/labels.csv"%task_name,gammas.labels)
	np.savetxt("../for_paper/model_output/%s/alpha/ids.csv"%task_name,gammas.ids)

	get_response_matrices(gammas,task_name,nRois)


def get_response_matrices(gammas,task_name,nRois):
	plot_order = merged_order(gammas)
	topic_sums = [gammas.topic_sums[o] for o in plot_order]

	for tl_i, topic_list in enumerate(plot_order):
		topics = []
		topic_weights = topic_sums[tl_i]
		topic_weights = topic_weights/np.sum(topic_weights)
		for t_i, t in enumerate(topic_list):
		    topic = pd.DataFrame.from_csv('topics/%s_k20/dynamic_data_topic_%s'%(task_name,t))
		    topics.append(topic*topic_weights[t_i])
		topic = pd.concat(topics)
		weights = np.mean(topic,axis=1)
		weights = flipab(weights)

		activations = [['bb'],['aa'],['ba']]
		mat_types = ['active','deactive','ba']

		conn_mats = []
		for index, grp in enumerate(activations):

			conn_mat = np.zeros([nRois,nRois])
			for activity in grp:
				for word in weights.index:
					if activity in word:
						inds = [int(filter(str.isdigit, str(l)))-1 for l in word.split('_')]
						# WEIGHT BY SUM OF TOPIC PROBABILITY
						conn_mat[inds[0],inds[1]]+=np.sum(weights[word])
						conn_mat[inds[1],inds[0]]+=np.sum(weights[word])

			conn_mats.append(conn_mat)
		conn_mats = np.array(conn_mats)
		conn_mats = conn_mats/np.sum(conn_mats)

		mat_out = "../for_paper/model_output/%s/beta/topic%s_response_matrics.mat"%(task_name,str(tl_i+1))
		scipy.io.savemat(mat_out,mdict={mat_types[0]:conn_mats[0],
										mat_types[1]:conn_mats[1],
										mat_types[2]:conn_mats[2]})



def merged_order(group_result):
    '''
        Returns ordered list of topics in the order they were merged in
        Useful for plotting and re-ordering
    '''
    labels = np.unique(group_result.labels)
    mat_order = []
    for l in labels:
        grp_ts = []
        for t in range(len(group_result.labels)):
            if group_result.labels[t] == l:
                grp_ts.append(group_result.ids[t])
        mat_order.append(grp_ts)
    return mat_order


def flipab(weights):
	# reverse ab words
	for w in weights.index:
	    if 'ab' in w:
	        nodes = [int(filter(str.isdigit, str(l)))-1 for l in w.split('_')]
	        new_index = 'ba%s_%s'%(nodes[1],nodes[0])
	        weights = weights.rename(index={w:new_index})
	return weights

if __name__ == '__main__':
    main()