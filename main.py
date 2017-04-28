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
from gensim import models
from utilities import *

def main():

    # set arguments
    args = {'num_topics': 20, # number of topics to estimate
            'num_samples': 480, # length of timeseries
            'num_subjects': 3, # number of subjects
            'parent_dir': 'opto', # data directory
            'run_model': False, # False assumes model has been previously run
            'run_pmi': False, # False assumes clustering has already been run once
            'stim_design': 'topics/opto_k20/stim_design.txt', # Location of stimulus design file
            'cluster_iters': 1000 # number of iterations for clustering (takes max mean gap stat)
            }

    # run model
    if args['run_model']:
        dtm = runDTM(args)
        save_dynamics(dtm,args)
    else:
        try:
            fit_model = 'dtm_fit_%s_%sk'%(args['parent_dir'],args['num_topics'])
            dtm = models.wrappers.DtmModel.load(fit_model)
        except :
            print('No model fit could be found.')
            raise

    # topic proportion for each subject at each t
    gammas = dtm.gamma_

    # get the topics with meaningful information
    gammas_clean, sig_topics = clean_gammas(gammas)

    # cluster topics
    topic_labels, topic_ids = cluster_gammas(args)

    # merge topics that were clustered together
    group_gammas = merge_gammas(gammas_clean,topic_labels,args)

    # add stimulus design to final matrix
    stim_design = np.loadtxt(args['stim_design'])
    if args['parent_dir'] == 'opto': #Fixes error in how this data was saved
        stim_design = np.roll(stim_design,6)
    group_gammas = np.vstack([group_gammas,stim_design])

    # save grouped topic probabilities, cluster labels, and original topic ids
    with open('%s_k%s_group_gammas.pkl'%(args['parent_dir'],args['num_topics']), 'wb') as out:
        output = Output(topic_ids, topic_labels, group_gammas)
        pickle.dump(output, out, pickle.HIGHEST_PROTOCOL)

    # show heatmap of result
    sns.heatmap(group_gammas)
    plt.show()

class Output:
    def __init__(self, ids, labels, gammas):
        self.ids = ids
        self.labels = labels
        self.gammas = gammas

if __name__ == '__main__':
    main()
