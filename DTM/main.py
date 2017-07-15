'''
==============================================
====== DYNAMIC TOPIC MODELING FOR FMRI =======
==============================================

     Assumes subject timeseries have been
     processed through:
        1) binning
        2) text creation (corr matrix as docs)

==============================================
==============================================
==============================================
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from subprocess import call
from gensim import models
from bct import community_louvain
from utilities import *

def main():

    # set arguments

    # OPTO
    # args = {'num_topics': 20, # number of topics to estimate
    #         'num_samples': 480, # length of timeseries
    #         'num_subjects': 3, # number of subjects
    #         'parent_dir': 'opto', # data directory
    #         'run_model': False, # False assumes model has been previously run
    #         'stim_design': 'topics/opto_k20/stim_design.txt', # Location of stimulus design file
    #         }
    #WORKING MEMORY
    # args = {'num_topics': 20, # number of topics to estimate
    #         'num_samples': 405, # length of timeseries
    #         'num_subjects': 120, # number of subjects #121
    #         'parent_dir': 'WM_RL', # data directory #WM_LR
    #         'run_model': False, # False assumes model has been previously run
    #         'stim_design': 'stim_designs/WM_RL_stimdesign.txt', # Location of stimulus design file
    #         }

    #MATH LEARNING
    args = {'num_topics': 20, # number of topics to estimate
            'num_samples': 4, # length of timeseries
            'num_subjects': 398, # number of subjects #388
            'parent_dir': 'math2', # data directory #math1
            'run_model': False, # False assumes model has been previously run
            }

    # run model
    if args['run_model']:
        dtm = runDTM(args)
        save_dynamics(dtm,args)
    else:
        try:
            fit_model = 'fit_models/dtm_fit_%s_%sk'%(args['parent_dir'],args['num_topics'])
            dtm = models.wrappers.DtmModel.load(fit_model)
        except :
            print('No model fit could be found.')
            raise

    gammas = dtm.gamma_
    topic_sums = np.sum(gammas,axis=0)/np.sum(gammas)

    #get the topics with meaningful information
    gammas_clean, sig_topics = clean_gammas(gammas)
    s = 0
    e = args['num_samples']
    grp_gammas = np.zeros([args['num_subjects'],args['num_samples'],np.shape(gammas_clean)[1]])
    #grp_gammas = np.zeros([args['num_subjects'],12,args['num_topics']])
    for sub in range(args['num_subjects']):
        grp_gammas[sub,:,:] = gammas_clean[s:e,:]
        s=e
        e+=args['num_samples']
    group_gammas = np.transpose(np.mean(grp_gammas,axis=0))

    #behavioral_analysis(topic_labels,grp_gammas,'RL')
    topic_labels, topic_ids = cluster_group(group_gammas,args['parent_dir'])

    if 'math' not in args['parent_dir']:
        group_gammas = merge_gammas(gammas_clean,topic_labels,args)
        stim_design = np.loadtxt(args['stim_design']) # add stimulus design to final matrix
        if args['parent_dir'] == 'opto': #Fixes error in how this data was saved
            stim_design = np.roll(stim_design,6)
        group_gammas = np.vstack([group_gammas,stim_design])
    else: # grab the individual gamma matrix
        group_gammas = merge_gammas_nomax(gammas_clean,topic_labels,args)
        indiv_gammas = merge_indiv_gammas(gammas_clean,topic_labels,args)
        with open('gammas_out/%s_indiv_gammas.pkl'%(args['parent_dir']),'wb') as f:
            pickle.dump(indiv_gammas, f, pickle.HIGHEST_PROTOCOL)

    # save grouped topic probabilities, cluster labels, and original topic ids
    with open('gammas_out/%s_k%s_group_gammas.pkl'%(args['parent_dir'],args['num_topics']), 'wb') as out:
        output = Output(sig_topics, topic_labels, group_gammas, topic_sums)
        pickle.dump(output, out, pickle.HIGHEST_PROTOCOL)


class Output:
    def __init__(self, ids, labels, gammas, topic_sums):
        self.ids = ids
        self.labels = labels
        self.gammas = gammas
        self.topic_sums = topic_sums

if __name__ == '__main__':
    main()
