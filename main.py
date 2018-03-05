import pints
import pints.toy
import pints.plot
import numpy as np
import matplotlib.pyplot as plt
import scipy

# Load a forward model
model = pints.toy.LogisticModel()

# Create some toy data
noise = 10
nexp = 10
mean = [0.015, 500.0]
stddev = [m / 5.0 for m in mean]
times = np.linspace(0, 1000, 1000)

# prior hyperparameters
alpha0 = 0
beta0 = 0
k_0 = 0
nu_0 = 1
mu_0 = np.array(mean)
Gamma_0 = 1.0e-9 * np.diag(mean)

# parameters = np.zeros((samples,len(mean)))
# values = np.zeros((samples,len(times)))
log_posteriors = []
samplers = []
exp_parameters = []
for i in range(nexp):
    # sample from parameter distribution
    parameters = np.random.normal(mean, stddev)
    print('exp', i, ': param =', parameters)
    exp_parameters.append(parameters)

    # generate from model + add noise
    values = model.simulate(parameters, times) + np.random.normal(0, noise)

    # Create a new log-likelihood function (adds an extra parameter!)
    problem = pints.SingleSeriesProblem(model, times, values)
    log_likelihood = pints.UnknownNoiseLogLikelihood(problem)

    # Create a new prior
    large = 1e9
    param_prior = pints.MultivariateNormalLogPrior(
        mean, large * np.eye(len(mean)))
    noise_prior = pints.NormalLogPrior(noise, large)
    log_prior = pints.ComposedLogPrior(param_prior, noise_prior)

    # Create a posterior log-likelihood (log(likelihood * prior))
    log_posterior = pints.LogPosterior(log_likelihood, log_prior)
    log_posteriors.append(log_posterior)

    sampler = pints.AdaptiveCovarianceMCMC(mean + [noise])
    samplers.append(sampler)


n_burn_in = 1000
for sample in range(n_burn_in):
    if sample % 10 == 0:
        print('x', end='', flush=True)
    # generate samples of hierarchical p1e9 * arams
    for i, (sampler, log_posterior) in enumerate(zip(samplers, log_posteriors)):
        x = sampler.ask()
        sampler.tell(log_posterior(x))


# Run a simple hierarchical gibbs-mcmc routine
n_samples = 10000
chain = np.zeros((n_samples, 2 * len(mean) + 1))
exp_chains = [np.zeros((n_samples, len(mean) + 1)) for i in range(nexp)]
for sample in range(n_samples):
    if sample % 10 == 0:
        print('.', end='', flush=True)

    # generate samples of lower level samplers
    xs = np.zeros((len(mean) + 1, nexp))
    for i, (sampler, log_posterior) in enumerate(zip(samplers, log_posteriors)):
        x = sampler.ask()
        xs[:, i] = sampler.tell(log_posterior(x))
        exp_chains[i][sample, :] = xs[:, i]

    # sample variance of noise from inverse gamma
    alpha = alpha0 + nexp / 2
    beta = beta0 + np.sum(xs[-1, :] * xs[-1, :]) / 2
    noise_variance_sample = scipy.stats.invgamma.rvs(alpha, scale=beta)

    # sample mean and covariance of other params from a normal inverse wishart
    xhat = np.mean(xs[0:len(mean), :], axis=1)
    C = np.zeros((len(mean), len(mean)))
    for x in xs[0:len(mean), :].T:
        C += np.outer(x - xhat, x - xhat)

    k = k_0 + nexp
    nu = nu_0 + nexp
    mu = (k_0 * mu_0 + nexp * xhat) / k
    tmp = xhat - mu_0
    Gamma = Gamma_0 + C + (k_0 * nexp) / k * np.outer(tmp, tmp)

    # generate means_sample and covariance_sample from normal-inverse-Wishart
    covariance_sample = scipy.stats.invwishart.rvs(df=nu, scale=Gamma)
    means_sample = scipy.stats.multivariate_normal.rvs(
        mean=mu, cov=covariance_sample / k)

    # store sample to chain
    chain[sample, 0:len(mean)] = means_sample
    chain[sample, len(mean):-1] = np.sqrt(np.diagonal(covariance_sample))
    chain[sample, -1] = np.sqrt(noise_variance_sample)

    # replace proposed values with sampled hierarchical parmas,
    # and update priors
    for i, (log_posterior, sampler) in enumerate(zip(log_posteriors, samplers)):
        i_start = i * len(mean)
        i_end = (i + 1) * len(mean)
        log_posterior._log_prior._priors[0]._mean = means_sample
        log_posterior._log_prior._priors[0]._cov = covariance_sample
        log_posterior._log_prior._priors[1]._sigma = np.sqrt(covariance_sample)

# drop first half of chain
chain = chain[int(n_samples / 2.0):, :]

# Look at distribution in chain
pints.plot.pairwise(chain, kde=False, true_values=mean + stddev + [noise])
plt.show()
pints.plot.trace(chain)
plt.show()
pints.plot.trace(*exp_chains)
plt.show()
