import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from utilities import *


def main():
    nRois = 3

    gammas = 'opto_k20_group_gammas.pkl'
    # index of topics to switch in plotting
    switches = [(4,1),(2,1)]
    fig1, plot_order = plot_task_matrix(gammas,switches)
    #plt.show()

    fig2 = plot_response_matrices(nRois,plot_order)

    plt.show()

def plot_task_matrix(gammas,switches):
    f = plt.figure
    with open(gammas, "rb") as input_file:
        group_result = pickle.load(input_file)

    plot_order = merged_order(group_result)

    if len(switches) > 0:
        for switch in switches:
            #switch vals in matrix
            temp = np.copy(group_result.gammas[switch[0],:])
            group_result.gammas[switch[0],:] = group_result.gammas[switch[1],:]
            group_result.gammas[switch[1],:] = temp
            #switch vals in plot order
            temp = np.copy(plot_order[switch[0]])
            plot_order[switch[0]] = plot_order[switch[1]]
            plot_order[switch[1]] = temp

    cmap="Reds"#sns.cubehelix_palette(as_cmap=True)
    sns.heatmap(group_result.gammas,cmap=cmap)
    return f, plot_order

def flipab(weights):
    # reverse ab words
    for w in weights.index:
        if 'ab' in w:
            nodes = [int(s) for s in w if s.isdigit()]
            new_index = 'ba%s_%s'%(nodes[1],nodes[0])
            weights = weights.rename(index={w:new_index})
    return weights

def plot_response_matrices(nRois,plot_order):

    f, axes = plt.subplots(len(plot_order),2)
    #plt.tight_layout()
    plot_row = 0
    for topic_list in plot_order:
        topics = []
        for t in topic_list:
            topic = pd.DataFrame.from_csv('topics/opto_k20/dynamic_data_topic_%s'%(t))
            topics.append(topic)
        topic = pd.concat(topics)

        weights = np.mean(topic,axis=1)
        weights = flipab(weights)

        activations = [['aa','bb'],['ba']]

        for index, grp in enumerate(activations):

            conn_mat = np.zeros([nRois,nRois])
            for activity in grp:
                for word in weights.index:
                    if activity in word:
                        inds = [int(s)-1 for s in word if s.isdigit()]
                        conn_mat[inds[0],inds[1]]+=np.mean(weights[word])
                        if activity == 'aa' or activity == 'bb':
                            conn_mat[inds[1],inds[0]]+=np.mean(weights[word])

            axis = axes[plot_row,index]
            cmap="Reds"#sns.cubehelix_palette(as_cmap=True)
            sns.heatmap(conn_mat, vmax=1, cmap=cmap, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=axis)
            if activity[0] == 'b' and activity[1] == 'a':
                # Uncoordinated Response is a nonsymmetric ba matrix
                axis.set_title('Uncoordinated Response')
                axis.set_ylabel('Active (b)')
                axis.set_xlabel('Inactive (a)')
            else:
                # Coordinated response is aa probabilities + bb probabilities
                axis.set_title('Coordinated Response')

        plot_row+=1

    return f

if __name__ == '__main__':
    main()
