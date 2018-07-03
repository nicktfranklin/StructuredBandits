# StructuredBandits
The model fits for the structured bandits task / analyses 

Note: Many of these files will take days to run on a modern laptop.

Files:
* `fit_clustering.py`: estimates the best alpha value for the clustering model for
 for each experiment using grid-search 
* `get_bayesian_gp_means_std.py`: estimates a mixture of Gaussian-processes by 
inferring which of three kernels (RBF, Linear, Noise) to use at each moment in 
time using precomputed means and standard deviations of the models
* `get_clustering_means_std.py`: estimates means and standard deviations of 
the clustering model.
* `choice_models.py`: fits each of the 9 choice model using hierarchical Bayesian 
parameter estimation to each of the 5 experiments, saving the model fit statistics
(LOO) and parameter estimates. 
