import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from utilities import *
from mpl_toolkits.mplot3d import Axes3D


def main():
    nRois = 11#3#6
    #labels = ['M1','CPu','Insula']
    #labels = ['lAI','rAI','lMFG','rMFG','lFEF','rFEF','lIPL','rIPL','lPCC','lVMPFC','rDMPFC']
    task_name = 'WM_RL'#'opto'#'tfMRI_MOTOR_LR'
    gammas = 'gammas_out/%s_k20_group_gammas.pkl'%(task_name)
    model = 'fit_models/dtm_fit_%s_20k'%(task_name) #'dtm_fit_opto_20k'

    plot_longitudinal(['math1','math2'])

    dtm = models.wrappers.DtmModel.load(model)

    fig1, plot_order, topic_sums = plot_task_matrix(gammas)

    #pcPlotter(gammas,dtm,nRois)
    #plot2d(gammas,dtm,nRois)
    #plot3d(gammas,dtm,nRois)

    #fig2 = plot_response_longitudinal(['math1','math2'])
    fig2 = plot_response_matrices(nRois,plot_order,topic_sums,task_name,labels)

    plt.show()


def pcPlotter(gammas,dtm,nRois):
    topPCs = getPCs(gammas,dtm,nRois)

    plt.plot(topPCs[0][:,0])
    plt.plot(topPCs[1][:,0])
    plt.show()


def getPCs(gammas,dtm,nRois):
    with open(gammas, "rb") as input_file:
        group_result = pickle.load(input_file)

    labels = np.unique(group_result.labels)
    topics = []
    for l in labels:
        t = np.array([i for i in range(len(group_result.labels)) if group_result.labels[i]==l])
        topics.append(t)

    num_topics = np.shape(group_result.gammas)[0]
    num_samples = np.shape(group_result.gammas)[1]

    topPCs = []
    for tops in topics:
        wordProbTS = np.zeros([nRois * 4,num_samples])
        for time in range(num_samples):
            topicProb = np.zeros([len(tops),nRois * 4])
            for top_i, top in enumerate(tops):
                topicOut = dtm.show_topic(topicid=top, time=time)
                w_ind = 0
                for p,w in topicOut:
                    print(w_ind)
                    topicProb[top_i,w_ind] = p
                    w_ind+=1
            wordProbTS[:,time] = np.sum(topicProb,axis=0)

        from sklearn import decomposition
        pca = decomposition.PCA()
        pca.fit(wordProbTS.T)
        X = pca.transform(wordProbTS.T)
        topPCs.append(X[:,:3])
    return topPCs


def plot2d(gammas,dtm,nRois):
    with open(gammas, "rb") as input_file:
        group_result = pickle.load(input_file)

    labels = np.unique(group_result.labels)
    topics = []
    for l in labels:
        t = np.array([i for i in range(len(group_result.labels)) if group_result.labels[i]==l])
        topics.append(t)

    num_topics = np.shape(group_result.gammas)[0]
    num_samples = np.shape(group_result.gammas)[1]

    for tops in topics:
        topicProb = np.zeros([len(tops),nRois * 4])
        fconn_bb = np.zeros([nRois,nRois,num_samples])
        fconn_aa = np.zeros([nRois,nRois,num_samples])
        fconn_ab = np.zeros([nRois,nRois,num_samples])
        fconn_ba = np.zeros([nRois,nRois,num_samples])
        for time in range(num_samples):
            topic = []
            for p,w in topic:
                locs = [int(filter(str.isdigit, str(l)))-1 for l in w.split('_')]
                if 'bb' in w:
                    fconn_bb[locs[0],locs[1],time]+=p
                    fconn_bb[locs[1],locs[0],time]+=p
                if 'aa' in w:
                    fconn_aa[locs[0],locs[1],time]+=p
                    fconn_aa[locs[1],locs[0],time]+=p
                if 'ba' in w:
                    fconn_ba[locs[0],locs[1],time]+=p
                    fconn_ba[locs[1],locs[0],time]+=p
                if 'ab' in w:
                    fconn_ab[locs[0],locs[1],time]+=p
                    fconn_ab[locs[1],locs[0],time]+=p

        fig = plt.figure()
        for r1 in range(nRois):
            for r2 in range(nRois):
                ts = fconn_bb[r1,r2,:]
                if np.mean(fconn_bb[r1,r2,:]) > 0.065: #1.0
                    plt.plot(fconn_bb[r1,r2,:],label='%s_%sbb'%(r1,r2))
                    plt.legend()
        plt.show()

def plot3d(gammas,dtm,nRois):
    from mpl_toolkits.mplot3d import Axes3D

    with open(gammas, "rb") as input_file:
        group_result = pickle.load(input_file)

    labels = np.unique(group_result.labels)
    topics = []
    for l in labels:
        t = np.array([i for i in range(len(group_result.labels)) if group_result.labels[i]==l])
        topics.append(t)

    num_topics = np.shape(group_result.gammas)[0]
    num_samples = np.shape(group_result.gammas)[1]
    for tops in topics:
        print(tops)
        fconn_bb = np.zeros([nRois,nRois,num_samples])
        fconn_aa = np.zeros([nRois,nRois,num_samples])
        fconn_ab = np.zeros([nRois,nRois,num_samples])
        fconn_ba = np.zeros([nRois,nRois,num_samples])
        for time in range(num_samples):
            topic = []
            for top in tops:
                topic.append(dtm.show_topic(topicid=top, time=time))
            topic = [i for j in topic for i in j]

            for p,w in topic:
                locs = [int(filter(str.isdigit, str(l)))-1 for l in w.split('_')]

                if 'bb' in w:
                    fconn_bb[locs[0],locs[1],time]+=p
                    fconn_bb[locs[1],locs[0],time]+=p
                if 'aa' in w:
                    fconn_aa[locs[0],locs[1],time]+=p
                    fconn_aa[locs[1],locs[0],time]+=p
                if 'ba' in w:
                    fconn_ba[locs[0],locs[1],time]+=p
                    fconn_ba[locs[1],locs[0],time]+=p
                if 'ab' in w:
                    fconn_ab[locs[0],locs[1],time]+=p
                    fconn_ab[locs[1],locs[0],time]+=p

        fig = plt.figure()
        ax1 = fig.add_subplot(2,2,1,projection='3d')
        ax2 = fig.add_subplot(2,2,2,projection='3d')
        ax3 = fig.add_subplot(2,2,3,projection='3d')
        ax4 = fig.add_subplot(2,2,4,projection='3d')

        ax1.plot(fconn_bb[0,1,:],fconn_bb[0,2,:],fconn_bb[1,2,:])
        ax2.plot(fconn_aa[0,1,:],fconn_aa[0,2,:],fconn_aa[1,2,:])
        ax3.plot(fconn_ba[0,1,:],fconn_ba[0,2,:],fconn_ba[1,2,:])
        ax4.plot(fconn_ab[0,1,:],fconn_ab[0,2,:],fconn_ab[1,2,:])

    plt.show()


def plot_task_prob(gammas):
    f = plt.figure

    with open(gammas, "rb") as input_file:
        group_result = pickle.load(input_file)

    plot_order = merged_order(group_result)

    n_states = group_result.gammas.shape[0] - 1
    state_gammas = group_result.gammas[:n_states,:]
    design = group_result.gammas[n_states,:]

    pd.DataFrame(state_gammas.T).plot()

    return f


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


def plot_longitudinal(task_names):
    long_gammas = np.zeros([4,4])
    labels = ['pre_1','pre_2','post_1','post_2']
    linestyles = ['--','--','-','-']
    colors = ['b','r','b','r']
    g_i = 0
    indiv_gammas = []
    for t_i, task_name in enumerate(task_names):
        gammas = 'gammas_out/%s_k20_group_gammas.pkl'%(task_name)
        model = 'dtm_fit_%s_20k'%(task_name)
        indiv_gamma = 'gammas_out/%s_indiv_gammas.pkl'%(task_name)

        with open(indiv_gamma, "rb") as input_file:
            group_result = pickle.load(input_file)
        indiv_gammas.append(group_result)

    data = {'probability':[],'state':[],'TR':[],'timepoint':[]}
    for tr in range(4):
        for state in range(2):
            for time in range(2):
                nTrials = np.shape(indiv_gammas[time])[0]
                for prob in range(nTrials):
                    data['probability'].append(indiv_gammas[time][prob,state,tr])
                    data['TR'].append(tr)
                    if time == 0 and state == 0:
                        data['timepoint'].append('pre')
                        data['state'].append('1')
                    elif time == 0 and state == 1:
                        data['timepoint'].append('pre')
                        data['state'].append('2')
                    elif time == 1 and state == 0:
                        data['timepoint'].append('post')
                        data['state'].append('1')
                    elif time == 1 and state == 1:
                        data['timepoint'].append('post')
                        data['state'].append('2')
    df = pd.DataFrame.from_dict(data)
    sns.set(style="whitegrid")
    g = sns.factorplot(x='TR',y='probability',hue='timepoint',row='state',data=df,
                       capsize=.2, palette="YlGnBu_d", size=6, aspect=1,legend=False)
    plt.legend(loc='upper left')
    plt.show()


def plot_response_longitudinal(task_names):
    f, axes = plt.subplots(2,2)
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

    # state 1 - want to see what is more active in post
    s1_bb = conn_mats[4] - conn_mats[0]
    s1_aa = conn_mats[5] - conn_mats[1]
    # state 2 - want to see what is less active in post
    s2_bb = conn_mats[6] - conn_mats[2]
    s2_aa = conn_mats[7] - conn_mats[3]

    conn_mats = [[s1_bb,s1_aa],[s2_bb,s2_aa]]
    for tl_i, topic_list in enumerate(plot_order):
        curr_mats = conn_mats[tl_i]
        curr_mats = np.array(curr_mats)
        curr_mats = curr_mats/np.sum(curr_mats)

        for index, cm in enumerate(curr_mats):
            activity = activations[index][0]
            axis = axes[plot_row,index]
            vmax = 0.06
            g = sns.heatmap(cm, square=True, linewidths=.5, #vmax=vmax,
                            vmax=vmax, cbar_kws={"shrink": .5},ax=axis)
            g.set_yticklabels(labels[::-1],rotation=0)
            g.set_xticklabels(labels,rotation=90)
            if activity[0] == 'b' and activity[1] == 'a':
                # Uncoordinated Response is a nonsymmetric ba matrix
                axis.set_title('Uncoordinated Response')
                axis.set_ylabel('Active (b)')
                axis.set_xlabel('Inactive (a)')
            elif activity == 'bb':
                axis.set_title('Coordinated Active Response')
            elif activity == 'aa':
                axis.set_title('Coordinated Deactive Response')

        plot_row+=1

    plt.tight_layout()
    plt.show()
    return f



def plot_response_matrices(nRois,plot_order,topic_sums,task_name,labels):
    f, axes = plt.subplots(len(plot_order),2)
    plot_row = 0

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

        conn_mats = []
        for index, grp in enumerate(activations):

            conn_mat = np.zeros([nRois,nRois])
            for activity in grp:
                for word in weights.index:
                    if activity in word:
                        inds = [int(filter(str.isdigit, str(l)))-1 for l in word.split('_')]
                        # WEIGHT BY SUM OF TOPIC PROBABILITY
                        #print(activity,weights[word])
                        conn_mat[inds[0],inds[1]]+=np.sum(weights[word]) #mean
                        conn_mat[inds[1],inds[0]]+=np.sum(weights[word])

        conn_mats.append(conn_mat)
        conn_mats = np.array(conn_mats)
        conn_mats = conn_mats/np.sum(conn_mats)
        for index, cm in enumerate(conn_mats):
            activity = activations[index][0]
            axis = axes[plot_row,index]
            cmap="Reds"
            if nRois == 3:
                vmax = 0.2
            else:
                vmax = 0.015
            g = sns.heatmap(cm, cmap=cmap, square=True, linewidths=.5,
                            vmax=vmax,cbar_kws={"shrink": .5},ax=axis)
            g.set_yticklabels(labels[::-1],rotation=0)
            g.set_xticklabels(labels,rotation=90)
            if activity[0] == 'b' and activity[1] == 'a':
                # Uncoordinated Response is a nonsymmetric ba matrix
                axis.set_title('Uncoordinated Response')
                axis.set_ylabel('Active (b)')
                axis.set_xlabel('Inactive (a)')
            elif activity == 'bb':
                axis.set_title('Coordinated Active Response')
            elif activity == 'aa':
                axis.set_title('Coordinated Deactive Response')

        plot_row+=1

    return f

if __name__ == '__main__':
    main()
