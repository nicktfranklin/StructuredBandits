# StructuredBandits
The model fits for the structured bandits task / analyses . The main file of interest is
`Model Fits and Parameter plots.ipynb`, which is a jupyter notebook with the results plotted. `Posterior Predictive Checks.ipynb` also has figures used
in the paper.

Note: Many of these files will take days to run on a modern laptop.

Processing steps (run in order):
1. `fit_clustering.py`: estimates the best alpha value for the clustering model for
 for each experiment using grid-search
2. `get_bayesian_gp_means_std.py`: estimates a mixture of Gaussian-processes by
inferring which of three kernels (RBF, Linear, Noise) to use at each moment in
time using precomputed means and standard deviations of the models
3. `get_clustering_means_std.py`: estimates means and standard deviations of
the clustering model.
4. `choice_models.py`: fits each of the 9 choice model using hierarchical Bayesian
parameter estimation to each of the 5 experiments, saving the model fit statistics
(LOO) and parameter estimates.
5. `run_ppc.py`: re-fits the GP-RBF/Clustering model and generates a sample
from the posterior predictive distribution. Currently runs 2 out of 5 experiments
(the linear and the shifted experiments)

The model fitting uses means and standard deviations output from a Gaussian Process model,
which are stored in the folders `Data/EXP_NAME`, as are the raw data files and the intermediate means
and standard deviations created by the code here. We used a separate process to generate the means
and standard deviations for the GPs, which required several days on a computing cluster. Breifly, the
a Gaussian process was fit to each trial for each subject given all of the observations the subject
had seen in that round of trials. Then, the predicted mean and stdev of the process were used as
predictors of the next trial.
