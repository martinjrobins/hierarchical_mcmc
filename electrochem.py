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

def trace2(names, chains, true_values=None):
    # If we switch to Python3 exclusively, bins and alpha can be keyword-only
    # arguments
    bins = 40
    alpha = 0.5
    chain = chains[0]
    n_sample, n_param = chain.shape
    if n_param > 6:
        n_param /= 2
        second = True
    else:
        second = False

    # Set up figure, plot first chain
    if second:
        fig, axes = plt.subplots(n_param, 2, figsize=(8, 1.5 * n_param))
    else:
        fig, axes = plt.subplots(n_param, 1, figsize=(4, 1.5 * n_param))
    for i in range(n_param):
        if second:
            # Add histogram subplot
            axes[i, 0].set_xlabel("Sample")
            axes[i, 0].set_ylabel(names[i])
            axes[i, 0].plot(chain[:, i], label='Chain 1')

            axes[i, 1].set_xlabel("Sample")
            axes[i, 1].set_ylabel(names[i+n_param])
            axes[i, 1].plot(chain[:, i+n_param], label='Chain 1')

        else:
            axes[i].set_xlabel("Sample")
            axes[i].set_ylabel(names[i])
            axes[i].plot(chain[:, i], label='Chain 1')

    # Plot additional chains
    if len(chains) > 1:
        for i_chain, chain in enumerate(chains[1:]):
            if chain.shape[1] != n_param:
                raise ValueError(
                    'All chains must have the same number of parameters.')
            for i in range(n_param):
                axes[i].plot(chain[:, i],
                             label='Chain ' + str(2 + i_chain))
        # axes[0, 0].legend()

    plt.tight_layout()
    return fig, axes

def trace(names, chains, true_values=None):
    # If we switch to Python3 exclusively, bins and alpha can be keyword-only
    # arguments
    bins = 40
    alpha = 0.5
    chain = chains[0]
    n_sample, n_param = chain.shape
    if n_param > 6:
        n_param /= 2
        second = True
    else:
        second = False

    # Set up figure, plot first chain
    if second:
        fig, axes = plt.subplots(n_param, 2, figsize=(8, 1.5 * n_param))
    else:
        fig, axes = plt.subplots(n_param, 1, figsize=(4, 1.5 * n_param))
    for i in range(n_param):
        if second:
            # Add histogram subplot
            axes[i, 0].set_xlabel(names[i])
            axes[i, 0].set_ylabel('Frequency')
            axes[i, 0].hist(
                chain[:, i], bins=bins, alpha=alpha, label='Chain 1')
            if true_values is not None:
                ymin_tv, ymax_tv = axes[i, 0].get_ylim()
                axes[i, 0].plot(
                    [true_values[i], true_values[i]],
                       [0.0, ymax_tv],
                       '--', c='k')

        # Add trace subplot
            axes[i, 1].set_xlabel(names[n_param + i])
            axes[i, 1].set_ylabel('Frequency')
            axes[i, 1].hist(
                chain[:, n_param + i], bins=bins, alpha=alpha, label='Chain 1')
            if true_values is not None:
                ymin_tv, ymax_tv = axes[i, 1].get_ylim()
                axes[i, 1].plot(
                    [true_values[n_param + i], true_values[n_param + i]],
                       [0.0, ymax_tv],
                       '--', c='k')

        else:
            # Add histogram subplot
            axes[i].set_xlabel(names[i])
            axes[i].set_ylabel('Frequency')
            axes[i].hist(chain[:, i], bins=bins, alpha=alpha, label='Chain 1')
            if true_values is not None:
                ymin_tv, ymax_tv = axes[i].get_ylim()
                axes[i].plot(
                    [true_values[i], true_values[i]],
                       [0.0, ymax_tv],
                       '--', c='k')

    # Plot additional chains
    if len(chains) > 1:
        for i_chain, chain in enumerate(chains[1:]):
            if chain.shape[1] != n_param:
                raise ValueError(
                    'All chains must have the same number of parameters.')
            for i in range(n_param):
                axes[i].hist(chain[:, i], bins=bins, alpha=alpha,
                             label='Chain ' + str(2 + i_chain))
        # axes[0, 0].legend()

    plt.tight_layout()
    return fig, axes


def pairwise(names, chain, kde=False, opacity=None):
    # Check chain size
    n_sample, n_param = chain.shape

    # Create figure
    fig_size = (10, 10)
    fig, axes = plt.subplots(n_param, n_param, figsize=fig_size)

    bins = 25
    for i in range(n_param):
        for j in range(n_param):
            if i == j:

                # Diagonal: Plot a histogram
                xmin, xmax = np.min(chain[:, i]), np.max(chain[:, i])
                xbins = np.linspace(xmin, xmax, bins)
                axes[i, j].set_xlim(xmin, xmax)
                axes[i, j].hist(chain[:, i], bins=xbins, normed=True)

                # Add kde plot
                if kde:
                    x = np.linspace(xmin, xmax, 100)
                    axes[i, j].plot(x, stats.gaussian_kde(chain[:, i])(x))

            elif i < j:
                # Top-right: no plot
                axes[i, j].axis('off')

            else:
                # Lower-left: Plot the samples as density map
                xmin, xmax = np.min(chain[:, j]), np.max(chain[:, j])
                ymin, ymax = np.min(chain[:, i]), np.max(chain[:, i])
                axes[i, j].set_xlim(xmin, xmax)
                axes[i, j].set_ylim(ymin, ymax)

                if not kde:
                    # Create scatter plot

                    # Determine point opacity
                    num_points = len(chain[:, i])
                    if opacity is None:
                        if num_points < 10:
                            opacity = 1.0
                        else:
                            opacity = 1.0 / np.log10(num_points)

                    # Scatter points
                    axes[i, j].scatter(
                        chain[:, j], chain[:, i], alpha=opacity, s=0.1)

                else:
                    # Create a KDE-based plot

                    # Plot values
                    values = np.vstack([chain[:, j], chain[:, i]])
                    axes[i, j].imshow(
                        np.rot90(values), cmap=plt.cm.Blues,
                        extent=[xmin, xmax, ymin, ymax])

                    # Create grid
                    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                    positions = np.vstack([xx.ravel(), yy.ravel()])

                    # Get kernel density estimate and plot contours
                    kernel = stats.gaussian_kde(values)
                    f = np.reshape(kernel(positions).T, xx.shape)
                    axes[i, j].contourf(xx, yy, f, cmap='Blues')
                    axes[i, j].contour(xx, yy, f, colors='k')

                    # Force equal aspect ratio
                    # See: https://stackoverflow.com/questions/7965743
                    im = axes[i, j].get_images()
                    ex = im[0].get_extent()
                    # Matplotlib raises a warning here (on 2.7 at least)
                    # We can't do anything about it, so no other option than
                    # to suppress it at this stage...
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', UnicodeWarning)
                        axes[i, j].set_aspect(
                            abs((ex[1] - ex[0]) / (ex[3] - ex[2])))

            # Set tick labels
            if i < n_param - 1:
                # Only show x tick labels for the last row
                axes[i, j].set_xticklabels([])
            else:
                # Rotate the x tick labels to fit in the plot
                for tl in axes[i, j].get_xticklabels():
                    tl.set_rotation(45)

            if j > 0:
                # Only show y tick labels for the first column
                axes[i, j].set_yticklabels([])

        # Set axis labels
        axes[-1, i].set_xlabel(names[i])
        axes[i, 0].set_ylabel(names[i])

    return fig, axes


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
synthetic = True 
if synthetic:
    pickle_file = 'syn_samplers_and_posteriors.pickle'
    sampled_true_parameters = np.zeros((len(names), len(filenames)))
    mu_truth = np.array([model.params[i] for i in names])
    print('mu_truth = ', mu_truth)
    stddev_truth = np.array([0.7, 0.06, 0.0007, 0.003, 0.005])
    print('stddev_truth = ', stddev_truth)
    noise = 0.027
else:
    pickle_file = 'samplers_and_posteriors.pickle'

if not os.path.isfile(pickle_file):
    log_posteriors = []
    samplers = []
    for i, filename in enumerate(filenames):

        data = electrochemistry.ECTimeData(
            filename, model, ignore_begin_samples=5, ignore_end_samples=0)
        if synthetic:
            while True:
                true_parameters = np.random.normal(mu_truth, stddev_truth)
                if np.all(true_parameters >= lower_bounds[:-1]) and np.all(true_parameters < upper_bounds[:-1]):
                    break
            sampled_true_parameters[:, i] = true_parameters
            current = pints_model.simulate(true_parameters, data.times)
            current = np.random.normal(current, noise)
            times = data.times
        else:
            current = data.current
            times = data.times

        problem = pints.SingleSeriesProblem(
            pints_model, times, current)
        boundaries = pints.Boundaries(lower_bounds, upper_bounds)

        # Create a new log-likelihood function (adds an extra parameter!)
        log_likelihood = pints.UnknownNoiseLogLikelihood(problem)

        # Create a new prior
        large = 1e9
        param_prior = pints.MultivariateNormalLogPrior(
            mu_0, large * np.eye(len(mu_0)))
        noise_prior = pints.UniformLogPrior(
            [lower_bounds[-1]], [upper_bounds[-1]])
        log_prior = pints.ComposedLogPrior(param_prior, noise_prior)

        # Create a posterior log-likelihood (log(likelihood * prior))
        log_posterior = pints.LogPosterior(log_likelihood, log_prior)
        log_posteriors.append(log_posterior)
        score = pints.ProbabilityBasedError(log_posterior)

        if synthetic:
            found_parameters = list(true_parameters)+[noise]
        else:
            found_parameters, found_value = pints.optimise(
                score,
                x0,
                sigma0,
                boundaries,
                method=pints.CMAES
            )

            values = pints_model.simulate(found_parameters, times)
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
    print('using starting points:')
    for i, sampler in enumerate(samplers):
        print('\t', sampler._x0)
        sampled_true_parameters[:, i] = sampler._x0[:-1]


if synthetic:
    pickle_file = 'syn_chain_and_exp_chains.pickle'
else:
    pickle_file = 'chain_and_exp_chains.pickle'
if not os.path.isfile(pickle_file):
    # burn in the individual samplers
    n_burn_in = 0 
    for sample in range(n_burn_in):
        if sample % 10 == 0:
            print('x', end='')
            sys.stdout.flush()
        # generate samples of hierarchical p1e9 * arams
        for i, (sampler, log_posterior) in enumerate(zip(samplers, log_posteriors)):
            x = sampler.ask()
            if np.any(x < lower_bounds) or np.any(x > upper_bounds):
                sampler.tell(-float('inf'))
            else:
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
            if np.any(x < lower_bounds) or np.any(x > upper_bounds):
                xs[:, i] = sampler.tell(-float('inf'))
            else:
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

if synthetic:
    sample_mean = np.mean(sampled_true_parameters, 1)
    sample_stddev = np.std(sampled_true_parameters, 1)

# Look at distribution in chain
print('plotting', chain.shape)
namesbase = [r'k_0', r'E_0', r'C_{dl}', r'R_u', r'\alpha']
print(names)
names = [r'$%s$' % i for i in namesbase]
names_std = [r'$\sigma_{%s}$' % i for i in namesbase]
print(names)
pairwise(names + names_std, chain, kde=False)
if synthetic:
    plt.savefig('syn_hpairwise.pdf')
    trace(names + names_std, [chain],
          true_values=list(sample_mean) + list(sample_stddev))
    plt.savefig('syn_htrace.pdf')
    trace2(names + names_std, [chain],
          true_values=list(sample_mean) + list(sample_stddev))
    plt.savefig('syn_htrace2.pdf')
else:
    plt.savefig('hpairwise.pdf')
    trace(names + names_std, chain)
    plt.savefig('htrace.pdf')
    trace2(names + names_std, [chain],
          true_values=list(sample_mean) + list(sample_stddev))
    plt.savefig('htrace2.pdf')

trace(names + [r'$n$'], exp_chains)
if synthetic:
    plt.savefig('syn_hchains.pdf')
else:
    plt.savefig('hchains.pdf')
trace2(names + [r'$n$'], exp_chains)
if synthetic:
    plt.savefig('syn_hchains2.pdf')
else:
    plt.savefig('hchains2.pdf')

