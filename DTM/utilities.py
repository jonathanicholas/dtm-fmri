import os
from gensim import corpora, models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Output:
    def __init__(self, ids, labels, gammas):
        self.ids = ids
        self.labels = labels
        self.gammas = gammas


def docGrabber(subjects, delimiter):
    '''
    Loads txt correlation matrix for each subject
    and returns dict of lists w/ words
    '''

    documents = {}
    for subject in subjects:
        index = 0
        words = []
        doc = open(subject, 'r')
        for word in doc.read().split(delimiter):
            index += 1
            word = word
            if word:
                words.append(word)

        documents[subject] = words

    return documents


def loadFiles(path):
    '''
    Returns list of subjects and dict of lists
    with text-wise correlation matrix
    '''

    subjects = sorted([os.path.join(path, fn) for fn in os.listdir(path)])
    for subject in subjects:
        if '.DS_Store' in subject: # just in case...
            subjects.remove(subject)
    delimiter = ' '
    return subjects, docGrabber(subjects, delimiter)


def runDTM(args):
    '''
    Runs Blei's Dynamic Topic Modeling Framework (2006)
    using gensim wrapper.
    '''

    print('Loading subjects...')

    num_samples = args['num_samples']
    path = 'texts/%s/'%(args['parent_dir'])
    subjects, full_data = loadFiles(path)

    window_list = []
    s_counter = 0
    for subject in subjects:
        s_counter+=1
        window_length = len(full_data[subject])/num_samples
        window_count = 0
        window = []
        window_boundary = window_length
        for word_counter in range(1,len(full_data[subject])+1):
            if word_counter == window_boundary:
                window_boundary+=window_length
                window_list.append(window)
                window = []
                window.append(full_data[subject][word_counter-1])
            else:
                window.append(full_data[subject][word_counter-1])

    print('Creating dictionary...')

    dictionary = corpora.Dictionary(window_list)

    print('Creating corpus...')

    corpus = [dictionary.doc2bow(window) for window in window_list]
    time_slices = [len(window_list)/num_samples] * num_samples

    out_file = 'fit_models/dtm_fit_%s_%sk'%(args['parent_dir'],args['num_topics'])

    print('Running Dynamic Topic Model...')

    dtm = models.wrappers.DtmModel('dtm-master/bin/dtm-darwin64',
                            corpus,
                            time_slices,
                            num_topics=args['num_topics'],
                            id2word=dictionary,
                            initialize_lda=True)
    dtm.save(out_file)

    return dtm


def clean_gammas(gammas):
    '''
        Drops topics that are zero
    '''

    from scipy import stats

    gamma_sum = np.sum(gammas,axis=0)
    gamma_mode = stats.mode(gamma_sum).mode[0]
    topic_inds = [ind for ind, g in enumerate(gamma_sum) if g != gamma_mode]
    gammas_clean = [gammas[:,ind] for ind in range(gammas.shape[1]) if ind in topic_inds]
    gammas_clean = np.array(gammas_clean).T
    return gammas_clean, topic_inds



def compute_pmi(**kwargs):
    '''
        Normalized Pointwise Mutual Information Calculation
            "Sentences" are defined as a single stimulation block
    '''

    num_samples = kwargs['num_samples']
    parent_dir = kwargs['parent_dir']
    topic1 = kwargs['topic1']
    topic2 = kwargs['topic2']
    stim_design = kwargs['stim_design']

    path = 'texts/%s/'%(parent_dir)
    subjects, full_data = loadFiles(path)

    textList = []
    for sub in subjects:
        textList.append(full_data[sub])
    nSubjects = len(textList)
    nTimepoints = len(textList) * num_samples # total number of timepoints

    def top_words(topic):
        '''
            Get the words that account for 95 percent probability in the topic
            or top 5 words
        '''
        top = np.mean(topic,axis=1)
        tmp = top.order(ascending=False)[:len(top)]
        total_prob = 0
        for ind,prob in enumerate(tmp):
            total_prob+=prob
            if total_prob >= 0.95 or ind == 5:
                nWords = ind
                break
        tmp = tmp[:nWords+1]
        return list(tmp.index)

    def id_blocks(stim_design):
        '''
            Differentiate between difference blocks
        '''
        blocks = np.zeros(len(stim_design))
        block_counter = 0
        for ind,stim in enumerate(stim_design):
            if ind == 0:
                prev_stim = stim
            if stim != prev_stim:
                block_counter+=1
            blocks[ind] = block_counter
            prev_stim = stim
        block_ids = np.unique(blocks)
        block_lengths = []
        for id in block_ids:
            block_lengths.append(list(blocks).count(id))
        return block_lengths

    def pTopics(topic1, topic2, block_lengths):
        '''
            Compute probability of each topic and joint probability
            NOTE: Assumes that subjects have the same stimulus design
        '''
        nTopic1 = 0
        nTopic2 = 0
        nJoint = 0
        for sub in range(nSubjects):
            text = textList[sub]
            start = 0
            end = block_lengths[0]
            for ind,block in enumerate(block_lengths):
                if ind == 0:
                    prev_block = 0
                context = text[prev_block:prev_block+block]
                t1_agreement = set(context).intersection(set(topic1))
                t2_agreement = set(context).intersection(set(topic2))
                if len(t1_agreement) == len(topic1):
                    nTopic1+=1
                if len(t2_agreement) == len(topic2):
                    nTopic2+=1
                if len(t1_agreement) == len(topic1) and len(t2_agreement) == len(topic2):
                    nJoint+=1
                prev_block+=block
        nComparisons = len(block_lengths)*nSubjects
        pTopic1 = float(nTopic1)/float(nComparisons)
        pTopic2 = float(nTopic2)/float(nComparisons)
        pJoint = float(nJoint)/float(nComparisons)
        return pTopic1, pTopic2, pJoint

    topic1 = top_words(topic1)
    topic2 = top_words(topic2)

    block_lengths = id_blocks(stim_design)

    pTopic1, pTopic2, jointP = pTopics(topic1, topic2, block_lengths)

    if pTopic1 != 0 and pTopic2 != 0 and jointP != 0:
        pmi = np.log(jointP/(pTopic1 * pTopic2))
        norm_pmi = pmi/-(np.log(jointP))
    else:
        norm_pmi = -1

    return norm_pmi



def get_pmi(**kwargs):
    '''
        Return matrix of normalized pointwise mutual information between all topics
    '''
    nTopics = kwargs['num_topics']
    parent_dir = kwargs['parent_dir']
    num_samples = kwargs['num_samples']
    stim_design = kwargs['stim_design']

    pmi_mat = np.zeros([nTopics,nTopics])
    for t1 in range(nTopics):
        topic1 = pd.DataFrame.from_csv('topics/%s_k%s/dynamic_data_topic_%s'%(parent_dir,nTopics,t1))
        for t2 in range(nTopics):
            topic2 = pd.DataFrame.from_csv('topics/%s_k%s/dynamic_data_topic_%s'%(parent_dir,nTopics,t2))
            norm_pmi = compute_pmi(topic1=topic1,topic2=topic2,parent_dir=parent_dir,num_samples=num_samples,stim_design=stim_design)
            pmi_mat[t1,t2] = norm_pmi
    np.savetxt('topics/%s_k%s/norm_pmi.csv'%(parent_dir,nTopics),pmi_mat)
    return pmi_mat



def save_dynamics(dtm,args):
    '''
        Save a local copy of each topic from the model for use in later functions
    '''
    nTopics = args['num_topics']
    nTimepoints = args['num_samples']
    parent_dir = args['parent_dir']

    dynamic_topics = {}
    for top in range(nTopics):
        dynamic_topics[top] = {}
        for ts in range(nTimepoints):
            dynamic_topics[top][ts] = {}
            tProbs = dtm.show_topic(top, ts)
            for word in tProbs:
                dynamic_topics[top][ts][word[1]] = word[0]
        s = []
        for ts in range(nTimepoints): #get unique words in topic
            dic = dynamic_topics[top][ts]
            key_list = list(set(key for key in dic.keys()))
            s = s + key_list
            s = list(set(word for word in s))

        df = pd.DataFrame(index=s, columns=range(nTimepoints))
        df = df.fillna(0)

        for ts in range(nTimepoints):
            dic = dynamic_topics[top][ts]
            for key in dic.keys():
                df.loc[key,ts] = dic.get(key)

        filename = 'topics/%s_k%i/dynamic_data_topic_%i'%(parent_dir,nTopics,top)

        df.to_csv(filename)

    print('Dynamics saved for all topics.\n')



def cluster_group(gammas,dataset):
    from sklearn.manifold import MDS
    import seaborn as sns
    import matplotlib.pyplot as plt
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    import scipy.cluster.hierarchy as hac

    corrcoef = np.corrcoef(gammas)
    sns.clustermap(corrcoef)
    plt.show()

    #labels = [1,3,1,1,1,1,2,3,1,1,3,3,2,2,2,3,3,1,3] #MOTOR RL
    #labels = [1,1,1,2,2,2,2,4,1,2,4,1,3,4,4,3,3,3,2] #MOTOR LR
    if dataset == 'opto':
        labels = [2,2,2,2,2,2,1,2,1,2,1,2] #OPTO LABELS
    elif dataset == 'WM_RL':
        labels = [1,2,1,1,1,2,1,1,2,1,2,2,1,2,1,2,1,2,2] #WM RL
    elif dataset == 'WM_LR':
        labels = [2,2,2,2,1,2,2,1,1,2,1,1,2,2,1,1,1,2,1] #WM LR
    elif dataset == 'math1':
        labels = [1,2,1,2,1,2,1,2,1,2,2,2,2,2,2,1,2,1,1] #math1
    elif dataset == 'math2':
        labels = [2,1,1,2,1,1,1,1,2,2,2,2,2,1,1,1,2,2,2] # math2

    ids = range(np.shape(gammas)[0])

    return labels, ids

def cluster_gammas(args,run_pmi=True):
    '''
        Perform clustering by:
            1. Calculating matrix of normalized pointwise mutual information
            2. Converting PMI to dissimilarity
            3. Multidimensional scaling on dissimilarity matrix
            4. K-Means for nIters with k = 2 ... n_topics - 1
            5. Choose optimal number of clusters using max gap statistic from above
            6. Consensus cluster using the cluster-based similarity partioning algorithm
                - Compute a similarity matrix of cluster agreement for each topic
                - Spectral clustering on sim matrix to produce final labels
    '''
    run_pmi = args['run_pmi']
    nIters = args['cluster_iters']
    parent_dir = args['parent_dir']
    num_topics = args['num_topics']
    num_samples = args['num_samples']
    stim_design = args['stim_design']

    stim_design = np.loadtxt(stim_design)

    if parent_dir == 'opto': #Fixes error in way data was saved
        stim_design = np.roll(stim_design,6)

    # compute or load matrix of normalized pointwise mutual information
    if run_pmi:
        pmi = get_pmi(num_topics=num_topics,parent_dir=parent_dir,num_samples=num_samples,stim_design=stim_design)
    else:
        pmi = np.loadtxt('topics/%s_k%s/norm_pmi.csv'%(parent_dir,num_topics))

    # convert pmi to dissimilarity matrix
    pmi = pmi - np.min(pmi)
    pmi = pmi/np.max(pmi)
    pmi_sums = np.sum(pmi,axis=0)
    remove_inds = [ind for ind, g in enumerate(pmi_sums) if g == 0]
    nonzero_topics = [ind for ind, g in enumerate(pmi_sums) if g != 0]
    while len(remove_inds) != 0:
        pmi = np.delete(pmi,remove_inds[0],axis=0)
        pmi = np.delete(pmi,remove_inds[0],axis=1)
        pmi_sums = np.sum(pmi,axis=0)
        remove_inds = [ind for ind, g in enumerate(pmi_sums) if g == 0]
    pmi = 1-pmi

    from sklearn.manifold import MDS
    # perform multidimensional scaling
    mds = MDS(n_components=2, max_iter=500, n_init=20, dissimilarity='precomputed')
    trans_data = mds.fit_transform(pmi)

    from gap import gap
    # perform kmeans clustering and compute gap statistics
    Ks = range(2,pmi.shape[1])
    k_labels = dict((el,[]) for el in Ks)
    mean_gaps = []
    for i in range(nIters):
        gap_stats,labels = gap(trans_data,ks=Ks)
        mean_gaps.append(gap_stats)
        for k in Ks:
            k_labels[k].append(labels[k])
    gaps_mean = np.mean(mean_gaps,axis=0)
    max_ind = np.argmax(gaps_mean)
    gap_mean = max(gaps_mean)
    optimal_k = Ks[max_ind]

    print("Optimal # clusters is: %s"%(optimal_k))
    print("Running Consensus Clustering...")

    # perform consensus clustering on labels from optimal k
    # use the cluster-based similarity partioning algorithm
    # to compute sim matrix, then use spectral clustering with optimal k
    from sklearn.cluster import SpectralClustering
    labels = np.array(k_labels[optimal_k])
    sim_labels = np.zeros([len(labels[0]),len(labels[0])])
    for label in labels:
        for l1 in range(len(label)):
            for l2 in range(len(label)):
                if label[l1] == label[l2]:
                    sim_labels[l1,l2]+=1
    sim_labels = np.divide(sim_labels,len(labels))

    sc = SpectralClustering(n_clusters=optimal_k,affinity='precomputed')
    sc.fit(sim_labels)
    sc_predict = sc.fit_predict(sim_labels)

    return sc_predict, nonzero_topics


def behavioral_analysis(topic_labels,grp_gammas,session):
    # There are no behavioral correlations with sum of probability.

    if session == 'LR':
        rmInds = [97]
    elif session == 'RL':
        rmInds = [52,73]

    behavData = pd.read_csv('behavioral_data/%s_WM_Stats_good_movmntandstats_rh.txt'%(session))
    subjectList = np.loadtxt('behavioral_data/subjectslist_good_movmntandstats_rh_122ss.txt')
    subjectList = [str(int(sub)) for sub in subjectList]
    for ind in rmInds:
        subjectList = np.delete(subjectList,ind)
        behavData = behavData.drop(ind,axis=0)

    bk0_acc_cols = [col for col in behavData.columns if '0BK'in col if 'ACC' in col]
    bk2_acc_cols = [col for col in behavData.columns if '2BK'in col if 'ACC' in col]
    bk2_grp_accs = np.zeros([len(subjectList)])
    for sub_i, sub in enumerate(subjectList):
        subData = behavData[behavData['subject'] == int(sub)]
        bk2_acc = []
        for col in bk2_acc_cols:
            try:
                bk2_acc.append(float(subData[col]))
            except:
                bk2_acc.append(np.NaN)
        if np.NaN not in bk2_acc:
            bk2_acc = np.mean(bk2_acc)
        else:
            bk2_acc = np.NaN
        bk2_grp_accs[sub_i] = bk2_acc

    labels = np.unique(topic_labels)
    indiv_gammas = []
    for l in labels:

        topics = np.array([grp_gammas[:,:,i] for i in range(len(topic_labels)) if topic_labels[i]==l])
        curr_topic = np.sum(topics,axis=0)
        indivTopicSums = np.sum(curr_topic,axis=1)
        print(indivTopicSums)

        sns.jointplot(bk2_grp_accs,indivTopicSums)
        plt.show()



        indiv_gammas.append(curr_topic)
    indiv_gammas = np.array(indiv_gammas)

def merge_indiv_gammas(gammas,topic_labels,args):
    '''
        Merge topics for each individual
    '''
    num_subjects = args['num_subjects']
    num_samples = args['num_samples']

    labels = np.unique(topic_labels)
    merged_gammas = np.zeros([len(labels),gammas.shape[0]])
    group_gammas = np.zeros([num_subjects,len(labels),num_samples])

    for l in labels:
        topics = np.array([gammas[:,i] for i in range(len(topic_labels)) if topic_labels[i]==l])
        curr_topic = np.sum(topics,axis=0)
        merged_gammas[l-1,:] = curr_topic


    start = 0
    end = num_samples
    for s in range(num_subjects):
        group_gammas[s,:,:] = merged_gammas[:,start:end]
        start = end
        end+=num_samples

    return group_gammas


def merge_gammas_nomax(gammas,topic_labels,args):
    '''
        Merge topics by averaging over the group
    '''
    num_subjects = args['num_subjects']
    num_samples = args['num_samples']

    labels = np.unique(topic_labels)
    merged_gammas = np.zeros([len(labels),gammas.shape[0]])
    group_gammas = np.zeros([len(labels),num_samples])

    for l in labels:
        topics = np.array([gammas[:,i] for i in range(len(topic_labels)) if topic_labels[i]==l])
        curr_topic = np.sum(topics,axis=0)
        merged_gammas[l-1,:] = curr_topic

    start = 0
    end = num_samples
    for s in range(num_subjects):
        group_gammas+=merged_gammas[:,start:end]
        start = end
        end+=num_samples
    group_gammas = group_gammas/num_subjects

    #group_gammas = group_gammas[~np.all(group_gammas == 0, axis=1)]

    return group_gammas


def merge_gammas(gammas,topic_labels,args):
    '''
        Merge topics by averaging over the group and taking max
    '''
    num_subjects = args['num_subjects']
    num_samples = args['num_samples']

    labels = np.unique(topic_labels)
    merged_gammas = np.zeros([len(labels),gammas.shape[0]])
    group_gammas = np.zeros([len(labels),num_samples])

    for l in labels:
        topics = np.array([gammas[:,i] for i in range(len(topic_labels)) if topic_labels[i]==l])
        curr_topic = np.sum(topics,axis=0)
        merged_gammas[l-1,:] = curr_topic

    start = 0
    end = num_samples
    for s in range(num_subjects):
        group_gammas+=merged_gammas[:,start:end]
        start = end
        end+=num_samples
    group_gammas = group_gammas/num_subjects

    # taking max gammas
    maxvals = np.argmax(group_gammas,axis=0)
    for i_max, m in enumerate(maxvals):
       group_gammas[:m,i_max] = np.zeros(len(group_gammas[:m,i_max]))
       group_gammas[m+1:,i_max] = np.zeros(len(group_gammas[m+1:,i_max]))

    group_gammas = group_gammas[~np.all(group_gammas == 0, axis=1)]

    return group_gammas



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
