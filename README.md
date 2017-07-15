## Uncovering time-varying latent brain states with dynamic topic modeling
###### Jonathan Nicholas, Kaustubh Supekar, Weidong Cai, Vinod Menon
 
## Data
#### Texts
Functional connectivity at each time point was converted via Symbolic Aggregate Approximation (SAX) and saved in documents in the DTM/texts directory for each dataset. This is input data for the dynamic topic model. Raw timeseries are not in this repository.

## Code
#### DTM/main.py
Runs a dynamic topic model for either the optogenetic, working memory, or math longitudinal datasets depending on input arguments. Will cluster data according to labels determined via hierarchical clustering and the NbClust R package.
#### DTM/utilities.py
Holds a number of functions used by main.py to load input data, run models, cluster, and save output
#### DTM/analysis.py
Contains all plotting functions. The main function contains variables that can be changed depending upon which dataset is being plotted and what plots you would like to run.

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
