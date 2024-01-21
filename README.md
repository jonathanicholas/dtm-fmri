## Uncovering time-varying latent brain states with dynamic topic modeling
###### Jonathan Nicholas, Kaustubh Supekar, Weidong Cai, Vinod Menon
 
## Data
#### Texts
Functional connectivity at each time point was converted via Symbolic Aggregate Approximation (SAX) and saved in documents in the DTM/texts directory for each dataset. This is input data for the dynamic topic model. Raw timeseries for HCP data are in this repository, but if you are part of SCSNL, the non-public timeseries are located in the copy on sherlock:

    $OAK/projects/jnichola/dtm-fmri

## Code
#### DTM/main.py
Runs a dynamic topic model for either the optogenetic, working memory, or math longitudinal datasets depending on input arguments. Will cluster data according to labels determined via hierarchical clustering and the NbClust R package.
#### DTM/utilities.py
Holds a number of functions used by main.py to load input data, run models, cluster, and save output
#### DTM/analysis.py
Contains all plotting functions. The main function contains variables that can be changed depending upon which dataset is being plotted and what plots you would like to run.
#### DTM/create_fit_summaries.py
Creates files that summarize the fits and extract the most important parts to be used in the paper. Then moves these to the for_paper directory.
#### SAX/runSAX.m
This is the primary script to run symbolic aggregate approximation (SAX) on an input timeseries of shape nSubjects x nRois x Time. Wrappers for sherlock job submission are included here as well.

## Paper
Everything needed to create figures and run stats on the fit models is in the 'for_paper' directory.
#### for_paper/dtm_figures.ipynb
This python notebook creates every panel that goes into all figures in the paper

## Results
#### Models
All fit models are in the directory DTM/fit_models
#### Matrices
Group and individual probabilitiy matrices for each topic are in DTM/gammas_out
#### Topics
Topics containing probability distributions over all words are in DTM/topics
#### Figures
<img src="http://i.imgur.com/k8T259F.png">
Figures used in the presentation labmeeting07-17-17.pptx can be found in DTM/figures

## Methods
<img src="https://github.com/boomsbloom/dtm-fmri/blob/master/DTM/figures/pipeline.png" width="500">
