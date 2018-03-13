from __future__ import print_function
import pints
import sys
sys.path.insert(0, '../electrochemistry')
sys.path.insert(0, '../electrochemistry/build')
import electrochemistry
import pints.plot
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pickle
import os.path

DEFAULT = {
    'reversed': True,
    'Estart': 0.5,
    'Ereverse': -0.1,
    'omega': 9.0152,
    'phase': 0,
    'dE': 0.08,
    'v': -0.08941,
    't_0': 0.001,
    'T': 297.0,
    'a': 0.07,
    'c_inf': 1 * 1e-3 * 1e-3,
    'D': 7.2e-6,
    'Ru': 8.0,
    'Cdl': 20.0 * 1e-6,
    'E0': 0.214,
    'k0': 0.0101,
    'alpha': 0.53,
}

filenames = ['GC01_FeIII-1mM_1M-KCl_02_009Hz.txt',
             'GC02_FeIII-1mM_1M-KCl_02a_009Hz.txt',
             'GC03_FeIII-1mM_1M-KCl_02_009Hz.txt',
             'GC04_FeIII-1mM_1M-KCl_02_009Hz.txt',
             'GC05_FeIII-1mM_1M-KCl_02_009Hz.txt',
             'GC06_FeIII-1mM_1M-KCl_02_009Hz.txt',
             'GC07_FeIII-1mM_1M-KCl_02_009Hz.txt',
             'GC08_FeIII-1mM_1M-KCl_02_009Hz.txt',
             'GC09_FeIII-1mM_1M-KCl_02_009Hz.txt',
             'GC10_FeIII-1mM_1M-KCl_02_009Hz.txt']

model = electrochemistry.ECModel(DEFAULT)
data0 = electrochemistry.ECTimeData(
    filenames[0], model, ignore_begin_samples=5, ignore_end_samples=0)
max_current = np.max(data0.current)
sim_current, sim_times = model.simulate(use_times=data0.times)
plt.plot(data0.times, data0.current, label='exp')
plt.plot(sim_times, sim_current, label='sim')
plt.legend()
plt.savefig('default.pdf')
max_k0 = model.non_dimensionalise(1000, 'k0')
e0_buffer = 0.1 * (model.params['Ereverse'] - model.params['Estart'])
names = ['k0', 'E0', 'Cdl', 'Ru', 'alpha']
true = [model.params[name] for name in names]

lower_bounds = [
    0.0,
    model.params['Estart'] + e0_buffer,
    0.0,
    0.0,
    0.4,
    0.005 * max_current]

upper_bounds = [
    100 * model.params['k0'],
    model.params['Ereverse'] - e0_buffer,
    10 * model.params['Cdl'],
    10 * model.params['Ru'],
    0.6,
    0.05 * max_current]

print('lower true upper')
for u, l, t in zip(upper_bounds, lower_bounds, true):
    print(l, ' ', t, ' ', u)


# Load a forward model
pints_model = electrochemistry.PintsModelAdaptor(model, names)
values = pints_model.simulate(true, data0.times)
plt.clf()
plt.plot(data0.times, data0.current, label='exp')
plt.plot(data0.times, values, label='sim')
plt.legend()
plt.savefig('default_pints.pdf')


nexp = len(filenames)

# cmaes params
x0 = np.array([0.5 * (u + l) for l, u in zip(lower_bounds, upper_bounds)])
sigma0 = [0.5 * (h - l) for l, h in zip(lower_bounds, upper_bounds)]

# prior hyperparameters
k_0 = 0
nu_0 = 1
mu_0 = np.array(x0[:-1])
Gamma_0 = 1.0e-9 * np.diag(mu_0)

# parameters = np.zeros((samples,len(mean)))
# values = np.zeros((samples,len(times)))
pickle_file = 'samplers_and_posteriors.pickle'
if not os.path.isfile(pickle_file):
    log_posteriors = []
    samplers = []
    for filename in filenames:
        data = electrochemistry.ECTimeData(
            filename, model, ignore_begin_samples=5, ignore_end_samples=0)

        problem = pints.SingleSeriesProblem(
            pints_model, data.times, data.current)
        boundaries = pints.Boundaries(lower_bounds, upper_bounds)

        # Create a new log-likelihood function (adds an extra parameter!)
        log_likelihood = pints.UnknownNoiseLogLikelihood(problem)

        # Create a new prior
        large = 1e9
        param_prior = pints.MultivariateNormalLogPrior(
            mu_0, large * np.eye(len(mu_0)))
        param_prior2 = pints.UniformLogPrior(
            [lower_bounds[:-1]], [upper_bounds[:-1]])
        noise_prior = pints.UniformLogPrior(
            [lower_bounds[-1]], [upper_bounds[-1]])
        log_prior = pints.ComposedLogPrior(param_prior, noise_prior)
        log_prior2 = pints.ComposedLogPrior(param_prior2, noise_prior)

        # Create a posterior log-likelihood (log(likelihood * prior))
        log_posterior = pints.LogPosterior(log_likelihood, log_prior)
        log_posterior2 = pints.LogPosterior(log_likelihood, log_prior2)
        log_posteriors.append(log_posterior)
        score = pints.ProbabilityBasedError(log_posterior)

        found_parameters, found_value = pints.optimise(
            score,
            x0,
            sigma0,
            boundaries,
            method=pints.CMAES
        )

        values = pints_model.simulate(found_parameters, data.times)
        plt.clf()
        plt.plot(data.times, values, label='sim')
        plt.plot(data.times, data.current, label='exp')
        plt.legend()
        plt.savefig('fit%s.pdf' % filename)

        sampler = pints.AdaptiveCovarianceMCMC(found_parameters)
        samplers.append(sampler)

    pickle.dump((samplers, log_posteriors), open(pickle_file, 'wb'))
else:
    samplers, log_posteriors = pickle.load(open(pickle_file, 'rb'))


pickle_file = 'chain_and_exp_chains.pickle'
if not os.path.isfile(pickle_file):
    # burn in the individual samplers
    n_burn_in = 1000
    for sample in range(n_burn_in):
        if sample % 10 == 0:
            print('x', end='')
            sys.stdout.flush()
        # generate samples of hierarchical p1e9 * arams
        for i, (sampler, log_posterior) in enumerate(zip(samplers, log_posteriors)):
            x = sampler.ask()
            sampler.tell(log_posterior(x))


    # Run a simple hierarchical gibbs-mcmc routine
    n_samples = 10000
    chain = np.zeros((n_samples, 2 * len(mu_0)))
    exp_chains = [np.zeros((n_samples, len(x0))) for i in range(nexp)]
    for sample in range(n_samples):
        if sample % 10 == 0:
            print('.', end='')
            sys.stdout.flush()

        # generate samples of lower level samplers
        xs = np.zeros((len(x0), nexp))
        error = 0
        for i, (sampler, log_posterior) in enumerate(zip(samplers, log_posteriors)):
            x = sampler.ask()
            xs[:, i] = sampler.tell(log_posterior(x))
            exp_chains[i][sample,:] = xs[:, i]

        # sample mean and covariance from a normal inverse wishart
        xhat = np.mean(xs[:-1], axis=1)
        C = np.zeros((len(xhat), len(xhat)))
        for x in xs[:-1].T:
            C += np.outer(x - xhat, x - xhat)

        k = k_0 + nexp
        nu = nu_0 + nexp
        mu = (k_0 * mu_0 + nexp * xhat) / k
        tmp = xhat - mu_0
        Gamma = Gamma_0 + C + (k_0 * nexp) / k * np.outer(tmp, tmp)

        covariance_sample = scipy.stats.invwishart.rvs(df=nu, scale=Gamma)
        means_sample = scipy.stats.multivariate_normal.rvs(
            mean=mu, cov=covariance_sample / k)

        # store sample to chain
        chain[sample, 0:len(mu_0)] = means_sample
        chain[sample, len(mu_0):] = np.sqrt(np.diagonal(covariance_sample))

        # replace individual sampler's priors with hierarchical params,
        for i, (log_posterior, sampler) in enumerate(zip(log_posteriors, samplers)):
            log_posterior._log_prior._priors[0]._mean = means_sample
            log_posterior._log_prior._priors[0]._cov = covariance_sample

    pickle.dump((chain, exp_chains), open(pickle_file, 'wb'))
else:
    chain, exp_chains = pickle.load(open(pickle_file, 'rb'))
    n_samples = chain.shape[0] 


# drop first half of chain
chain = chain[int(n_samples / 2.0):,:]
exp_chains = [i[int(n_samples / 2.0):,:] for i in exp_chains]

# Look at distribution in chain
print('plotting', chain.shape)
print(len(mu_0))
pints.plot.pairwise(chain, kde=False)
plt.savefig('hpairwise.pdf')
pints.plot.trace(chain)
plt.savefig('htrace.pdf')
pints.plot.trace(*exp_chains)
plt.savefig('hchains.pdf')
