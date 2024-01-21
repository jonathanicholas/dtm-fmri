import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import scipy.io
import bctpy
from utilities import *

def main():

	#get_response_matrices(["math1","math2"])

	s1_mats = scipy.io.loadmat('state1_response_matrices.mat')

	sns.heatmap(s1_mats['pre_active'])






def plot_task_matrix(gammas):
    f = plt.figure
    with open(gammas, "rb") as input_file:
        group_result = pickle.load(input_file)

    plot_order = merged_order(group_result)
    topic_sums = [group_result.topic_sums[o] for o in plot_order]

    cmap="Reds"

    return f, plot_order, topic_sums

def flipab(weights):
	# reverse ab words
	for w in weights.index:
	    if 'ab' in w:
	        nodes = [int(filter(str.isdigit, str(l)))-1 for l in w.split('_')]
	        new_index = 'ba%s_%s'%(nodes[1],nodes[0])
	        weights = weights.rename(index={w:new_index})
	return weights

def get_response_matrices(task_names):
    #f, axes = plt.subplots(2,2)
    labels = ['L SPL', 'R SPL', 'L IPS', 'R IPS', 'L Insula', 'R Insula',
              'dACC', 'SMA', 'L MFG', 'R MFG', 'R VLPFC', 'L VLPFC',
              'R FG', 'L FG', 'R Hipp', 'L Hipp']
    plot_row = 0

    long_gammas = np.zeros([4,4])
    g_i = 0
    indiv_gammas = []
    nRois = 16
    conn_mats,mat_type = [], []
    for t_i, task_name in enumerate(task_names):
        gammas = 'gammas_out/%s_k20_group_gammas.pkl'%(task_name)
        model = 'dtm_fit_%s_20k'%(task_name)
        indiv_gamma = 'gammas_out/%s_indiv_gammas.pkl'%(task_name)
        fig1, plot_order, topic_sums = plot_task_matrix(gammas)

        with open(indiv_gamma, "rb") as input_file:
            group_result = pickle.load(input_file)
        indiv_gammas.append(group_result)

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

            activations = [['bb'],['aa']]

            for index, grp in enumerate(activations):
                mat_type.append((task_name,grp))

                conn_mat = np.zeros([nRois,nRois])
                for activity in grp:
                    for word in weights.index:
                        if activity in word:
                            inds = [int(filter(str.isdigit, str(l)))-1 for l in word.split('_')]
                            conn_mat[inds[0],inds[1]]+=np.sum(weights[word]) #mean
                            conn_mat[inds[1],inds[0]]+=np.sum(weights[word])
                conn_mats.append(conn_mat)


    print(conn_mats)

    post_active_s1 = conn_mats[4]
    pre_active_s1 = conn_mats[0]
    post_deactive_s1 = conn_mats[5]
    pre_deactive_s1 = conn_mats[1] 

    post_active_s2 = conn_mats[6]
    pre_active_s2 = conn_mats[2]
    post_deactive_s2 = conn_mats[7]
    pre_deactive_s2 = conn_mats[3] 




    # scipy.io.savemat('state1_response_matrices.mat', mdict={'pre_active': pre_active_s1,
    # 													   'pre_deactive': pre_deactive_s1,
    # 													   'post_active':post_active_s1,
    # 													   'post_deactive':post_deactive_s1,
    # 													   'labels':labels})

    # scipy.io.savemat('state2_response_matrices.mat', mdict={'pre_active': pre_active_s2,
    # 													   'pre_deactive': pre_deactive_s2,
    # 													   'post_active':post_active_s2,
    # 													   'post_deactive':post_deactive_s2,
    # 													   'labels':labels})


if __name__ == '__main__':
    main()





