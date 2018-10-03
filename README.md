# Finding Structure in Multi-Armed Bandits
This repository contains the code for model fitting in our paper "Finding Structure in Multi-Armed Bandits". 

Our behavioral analyises are hosted seperately here:
* Experiment 1:  https://ericschulz.github.io/explin/exp1.html
* Experiment 2: https://ericschulz.github.io/explin/exp2.html
* Experiment 3: https://ericschulz.github.io/explin/exp3.html
* Experiment 4: https://ericschulz.github.io/explin/exp4.html
* Experiment 5: https://ericschulz.github.io/explin/exp5.html


The experiments themselves can be played by clicking the following links:
* Experiment 1: https://ericschulz.github.io/explin/index1.html
* Experiment 2: https://ericschulz.github.io/explin/index2.html
* Experiment 3: https://ericschulz.github.io/explin/index3.html
* Experiment 4: https://ericschulz.github.io/explin/index4.html
* Experiment 5: https://ericschulz.github.io/explin/index5.html

For the model fits, the main file of interest is
`Model Fits and Parameter plots.ipynb`, which is a jupyter notebook with the results plotted.

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

*N.B. Many of these files will take days to run on a modern laptop.*


The model fitting uses means and standard deviations output from a Gaussian Process model,
which are stored in the folders `Data/EXP_NAME`, as are the raw data files and the intermediate means
and standard deviations created by the code here. We used a separate process to generate the means
and standard deviations for the GPs, which required several days on a computing cluster. Breifly, the
a Gaussian process was fit to each trial for each subject given all of the observations the subject
had seen in that round of trials. Then, the predicted mean and stdev of the process were used as
predictors of the next trial.


## Description of Bayesian Regression
Hierarchical Bayesian linear regression is used to fit the decision function to mean and standard 
deviation of each arm of the bandit tasks. These means and standard deviations are separately estimated
(discussed below). All of the Bayesian models are defined in the file `fit_clustering.py` using the [PyMC3
 library](https://docs.pymc.io).

Each Bayesian model is more or less the same and the script includes preprocessing to unify the format 
of each of the tasks. The logic of the model is similar to multivariate regression, and we refer to the 
decision variables as "regression coefficents" and the values as "predictors" accordingly.

For each subject, a predictor vector is prepared for each model's mean and standard
deviation for each trial. In addition a one-hot vector for sticky choice is also prepared. For each subject,
a single (scalar valued) regression coefficient is multiplied to these vectors before they are added and 
passed through a softmax function.  These parameters are fit 'hierachically', meaning we assume a group 
distribution over each parameter.

Example code for fitting the Kalman filter is provided below:
```buildoutcfg
    with pm.Model() as hier_kal:
        mu_1 = pm.Normal('mu_beta_kal_mean', mu=0., sd=100.)
        mu_2 = pm.Normal('mu_beta_kal_stdv', mu=0., sd=100.)
        mu_3 = pm.Normal('mu_beta_stick',    mu=0., sd=100.)

        sigma_1 = pm.HalfCauchy('sigma_kal_means', beta=100)
        sigma_2 = pm.HalfCauchy('sigma_kal_stdev', beta=100)
        sigma_3 = pm.HalfCauchy('sigma_stick',     beta=100)

        b_1 = pm.Normal('beta_kal_mu',  mu=mu_1, sd=sigma_1, shape=n_subj)
        b_2 = pm.Normal('beta_kal_std', mu=mu_2, sd=sigma_2, shape=n_subj)
        b_3 = pm.Normal('beta_sc',      mu=mu_3, sd=sigma_3, shape=n_subj)

        rho = \
            tt.tile(tt.reshape(b_1[subj_idx], (n, 1)), d) * x_mu_kal + \
            tt.tile(tt.reshape(b_2[subj_idx], (n, 1)), d) * x_sd_kal + \
            tt.tile(tt.reshape(b_3[subj_idx], (n, 1)), d) * x_sc
        p_hat = softmax(rho)

        # Data likelihood
        yl = pm.Categorical('yl', p=p_hat, observed=y)

        # inference!
        trace_kal = pm.sample(**sample_kwargs)
```

In this model, each of the three regression coefficients (`b_1`, `b_2` & `b_3` for the means, standard 
deviations and sticky choice, respectively) is assumed to be normally distributed.  The means 
(`mu_1`, `mu_2`, `mu_3`) and standard deviations  (`sigma_1`, `sigma_2`, `sigma_3`) 
of these distributions are group level parameters and can be interpreted as group effects. We've 
assumed vague priors for these group level distributions, with the group means assumed to be 
normally distributed and the standard deviations assumed to be distribued via a Half-cauchy 
distribution (c.f. [Gelman 2006](https://projecteuclid.org/download/pdf_1/euclid.ba/1340371048) for more details)

PyMC3 currently uses the Theano tensor library to underly its calculatios. Because we have vectors as predictors,
 we have to use vector math to get the right representation for the softmax. The following codes casts the 
 regresion coefficients into an n_arm by n_trial matrix so and uses element-wise multiplication to create the input
 to the softmax.

```buildoutcfg
        rho = \
            tt.tile(tt.reshape(b_1[subj_idx], (n, 1)), d) * x_mu_kal + \
            tt.tile(tt.reshape(b_2[subj_idx], (n, 1)), d) * x_sd_kal + \
            tt.tile(tt.reshape(b_3[subj_idx], (n, 1)), d) * x_sc
        p_hat = softmax(rho)
```

Below is the code for inference:
```buildoutcfg
        # Data likelihood
        yl = pm.Categorical('yl', p=p_hat, observed=y)

        # inference!
        trace_kal = pm.sample(**sample_kwargs)
```
Most of this is automatically determined in PyMC3 and we encourage you to look at [the documentation here](https://docs.pymc.io) 
for more details!