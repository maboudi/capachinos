import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import binom, kstwobign
from scipy.stats import norm, poisson


def gendata(N, distribution, *args):
    """
    Generates random data drawn from a specified probability distribution.

    Parameters:
    N (int): Number of data points
    distribution (list): Distribution name and parameters
        Format: ['distribution_name', [parameters]]
        Options:
            ['exponential', lambda] 
            ['lognormal', [mu, sigma]] Note: continuous distribution not yet available
            ['powerlaw', tau]
            ['truncated_powerlaw', [tau, lambda, xmin, xmax]] Note: continuous distribution not yet available
            ['exp_powerlaw', [tau, lambda]] Note: continuous distribution not yet available for this distribution

    Variable Inputs:
    (..., 'continuous'): Returns continuous data
    (..., 'inf', infimum): Smallest integer value drawn from distribution for discrete data [default: 1]
    (..., 'sup', supremum): Largest integer value drawn from distribution for discrete data [default: 100]
    (..., 'xmin', xmin): Smallest continuous value drawn from the distribution for continuous powerlaw [default: 1]
    (..., 'xmax', xmax): Largest continuous value drawn from the distribution for continuous powerlaw [default: inf]

    Returns:
    tuple:
        x (np.array): Random data
        xVals (np.array): X values of sampled pdf
        pdf (np.array): Probability density function from which the data are drawn

    Example usage:
    x = gendata(10000, ['powerlaw', 1.5])
        # Generate 10000 samples from a discrete power-law with exponent 1.5
    x = gendata(1000, ['truncated_powerlaw', [2, 0.125, 10, 50]])
        # Generate 1000 samples from a truncated power-law with exponent 2
        # between for 10 <= x <= 50, and exponential decay with constant
        # equal to 0.125 otherwise
    """

    dataType = 'INT'
    infimum = 1
    supremum = 100
    xmin = 1
    xmax = np.inf

    # Parsing variable arguments
    i = 0
    while i < len(args):
        argOkay = True
        if isinstance(args[i], str):
            if args[i] == 'continuous':
                dataType = 'CONT'
            elif args[i] == 'inf':
                infimum = args[i+1]
            elif args[i] == 'sup':
                supremum = args[i+1]
            elif args[i] == 'xmin':
                xmin = args[i+1]
            elif args[i] == 'xmax':
                xmax = args[i+1]
            else:
                argOkay = False
        if not argOkay:
            print(f'(GENDATA) Ignoring invalid argument #{i}')
        i += 1
    
    # Integer Data
    if dataType == 'INT':
        xVals = np.arange(infimum, supremum + 1)
        nX = len(xVals)

        # Distribution types
        if distribution[0] == 'exponential':
            lambda_ = distribution[1]
            pdf = np.exp(-lambda_ * xVals)

        elif distribution[0] == 'lognormal':
            mu, sigma = distribution[1]
            pdf = 1. / xVals * np.exp(-(np.log(xVals) - mu) ** 2 / (2 * sigma ** 2))

        elif distribution[0] == 'powerlaw':
            tau = distribution[1]
            pdf = xVals ** (-tau)

        elif distribution[0] == 'truncated_powerlaw':
            tau, lambda_, xmin, xmax = distribution[1]
            pdf = np.zeros(nX)

            # Handle distribution segments
            distHead = np.arange(infimum, xmin)
            for xi in distHead:
                pdf[xi - infimum] = (xmin ** (-tau) / np.exp(-lambda_ * xmin)) * np.exp(-lambda_ * xi)

            distBody = np.arange(xmin, xmax + 1)
            for xi in distBody:
                pdf[xi - infimum] = xi ** (-tau)

            distTail = np.arange(xmax + 1, supremum + 1)
            for xi in distTail:
                pdf[xi - infimum] = (xmax ** (-tau) / np.exp(-lambda_ * xmax)) * np.exp(-lambda_ * xi)

        elif distribution[0] == 'exp_powerlaw':
            tau, lambda_ = distribution[1]
            pdf = np.exp(-lambda_ * xVals) * (xVals ** (-tau))

        # Normalize pdf and generate data
        pdf = pdf / np.sum(pdf)
        counts = np.random.multinomial(N, pdf)

        x = np.zeros(N)
        idx = 0
        for xi in range(nX):
            for _ in range(counts[xi]):
                x[idx] = xVals[xi]
                idx += 1
        return x, xVals, pdf

    # Continuous Data
    elif dataType == 'CONT':
        u = np.random.rand(N)
        if distribution[0] == 'exponential':
            lambda_ = distribution[1]

            a = 1 - np.exp(-lambda_)
            u = (1 - a) * u + a
            x = -(1 / lambda_) * np.log(1 - u)

        elif distribution[0] == 'lognormal':
            raise ValueError("Continuous lognormal distribution not currently available")

        elif distribution[0] == 'powerlaw':
            tau = distribution[1]

            if tau <= 1:
                raise ValueError("tau must be greater than 1")

            a = 1 - xmin ** (1 - tau)
            b = 1 - xmax ** (1 - tau)
            u = (b - a) * u + a

            x = np.exp(np.log(1 - u) / (1 - tau))

        elif distribution[0] == 'truncated_powerlaw':
            raise ValueError("Continuous truncated powerlaw distribution not currently available")

        elif distribution[0] == 'exp_powerlaw':
            raise ValueError("Continuous exponentially modified distribution not currently available")
        
        return x, None, None

def rldecode(lengths, values):
    """
    Run-length decoding of run-length encoded data.

    Parameters:
    lengths (list or np.array): Lengths of each run
    values (list or np.array): Corresponding values of the runs

    Returns:
    np.array: Decoded values based on the provided lengths and values

    Example:
    rldecode([2, 3, 1, 2, 4], [6, 4, 5, 8, 7]) will return
    [6, 6, 4, 4, 4, 5, 8, 8, 7, 7, 7, 7]
    """
    lengths = np.array(lengths)
    values = np.array(values)
    
    # Keep only runs whose length is positive
    valid_runs = lengths > 0
    lengths = lengths[valid_runs]
    values = values[valid_runs]
    
    # Perform the actual run-length decoding
    indices = np.cumsum(lengths)
    decoded = np.zeros(indices[-1], dtype=values.dtype)
    decoded[indices[:-1]] = 1
    decoded[0] = 1
    expanded_values = np.cumsum(decoded).astype(int)
    
    return values[expanded_values - 1]

def pldist(num_starting_points, *args):
    """
    Generates power-law distributed data with optional truncation and exponential distribution outside the power-law regions.

    Parameters:
    num_starting_points (int): Number of starting points for the distribution

    Variable Inputs:
    (..., 'infimum', infimum): Sets the absolute minimum value of the data [default: 1]
    (..., 'supremum', supremum): Sets the absolute maximum value of the data [default: 10^(ceil(log10(num_starting_points)) + 1)]
    (..., 'slope', slope): Sets the slope for non-truncated power-law distribution [default: 1.5]
    (..., 'lambda', lambda_): Sets the lambda value used for non-power-law distributed data [default: 0.025]
    (..., 'upper', xmax): Generates power-law distributed data such that x <= xmax
    (..., 'lower', xmin): Generates power-law distributed data such that x >= xmin
    (..., 'double', xmin, xmax): Generates power-law distributed data such that xmin <= x <= xmax
    (..., 'plot'): Plots distribution

    Returns:
    np.array: Random data distributed within the power-law and exponential regions
    """

    # Default values
    infimum = 1
    supremum = 10 ** (int(np.ceil(np.log10(num_starting_points))) + 1)
    slope = 1.5
    lambda_ = 0.025
    dist_type = 'non-truncated'
    plot_flag = False

    # Parsing variable arguments
    i = 0
    while i < len(args):
        if args[i] == 'infimum':
            infimum = args[i+1]
            i += 2
        elif args[i] == 'supremum':
            supremum = args[i+1]
            i += 2
        elif args[i] == 'slope':
            slope = args[i+1]
            i += 2
        elif args[i] == 'lambda':
            lambda_ = args[i+1]
            i += 2
        elif args[i] == 'upper':
            dist_type = 'truncated above'
            xmax = args[i+1]
            i += 2
        elif args[i] == 'lower':
            dist_type = 'truncated below'
            xmin = args[i+1]
            i += 2
        elif args[i] == 'double':
            dist_type = 'double truncated'
            xmin = args[i+1]
            xmax = args[i+2]
            i += 3
        elif args[i] == 'plot':
            plot_flag = True
            i += 1
        else:
            print(f'(PLDIST) Ignoring invalid argument #{i + 1}')
            i += 1

    # Make probability density function for distribution
    if dist_type == 'non-truncated':
        pdf_values = np.round(num_starting_points * (np.arange(infimum, supremum + 1)) ** (-slope))

    elif dist_type == 'truncated above':
        assert xmax < supremum, 'xmax must be strictly less than supremum.'
        coeff1 = num_starting_points
        coeff2 = num_starting_points * (xmax ** -slope / np.exp(-lambda_ * xmax))
        pdf_values = np.round(np.concatenate([
            coeff1 * (np.arange(infimum, xmax + 1)) ** -slope,
            coeff2 * np.exp(-lambda_ * (np.arange(xmax + 1, supremum + 1)))
        ]))

    elif dist_type == 'truncated below':
        assert xmin > infimum, 'xmin must be strictly greater than infimum.'
        coeff1 = num_starting_points * (xmin ** -slope / np.exp(-lambda_ * xmin))
        coeff2 = num_starting_points
        pdf_values = np.round(np.concatenate([
            coeff1 * np.exp(-lambda_ * (np.arange(infimum, xmin))),
            coeff2 * (np.arange(xmin, supremum + 1)) ** -slope
        ]))

    elif dist_type == 'double truncated':
        assert xmin > infimum, 'xmin must be strictly greater than infimum.'
        assert xmax < supremum, 'xmax must be strictly less than supremum.'
        coeff1 = num_starting_points * (xmin ** -slope / np.exp(-lambda_ * xmin))
        coeff2 = num_starting_points
        coeff3 = num_starting_points * (xmax ** -slope / np.exp(-lambda_ * xmax))
        pdf_values = np.round(np.concatenate([
            coeff1 * np.exp(-lambda_ * (np.arange(infimum, xmin))),
            coeff2 * (np.arange(xmin, xmax + 1)) ** -slope,
            coeff3 * np.exp(-lambda_ * (np.arange(xmax + 1, supremum + 1)))
        ]))

    # Generate data
    data = rldecode(pdf_values.astype(int), np.arange(infimum, supremum + 1))

    # Plot if indicated
    if plot_flag:
        plplottool(data)

    return data

def plmle(x, *args):
    """
    Estimates the slope of power law distributed data using the method of maximum likelihood.

    Parameters:
    x (np.array): Random data to be fitted to the power law distribution p(x) ~ x^(-tau) for x >= xmin and x <= xmax.

    Variable Inputs:
    (..., 'xmin', xmin): Specifies the lower truncation of distribution for the fit (default: min(x))
    (..., 'xmax', xmax): Specifies the upper truncation of distribution for the fit (default: max(x))
    (..., 'tauRange', tauRange): Sets the range of taus to test for the MLE fit (default: [1, 5])
    (..., 'precision', precision): Sets the decimal precision for the MLE search (default: 10^-3)

    Returns:
    tuple:
        tau (float): Slope of power law region
        xmin (float): Lower truncation of distribution
        xmax (float): Upper truncation of distribution
        L (np.array): Log-likelihood that we wish to maximize for the MLE

    Example usage:
    x = pldist(10**4)
        # generates perfectly non-truncated power-law distributed data with slope of 1.5
    tau, xmin, xmax, L = plmle(x)
        # computes tau by MLE for x
    tau, xmin, xmax, L = plmle(x, 'precision', 10**-4)
        # computes tau to 4 decimal places
    x = pldist(10**4, 'upper', 50, 1.5)
        # generates perfectly power-law distributed data for x <= 50
    tau, xmin, xmax, L = plmle(x, 'xmax', 50)
        # computes tau for truncated region
    """

    # Default values
    xmin = np.min(x)
    xmax = np.max(x)
    tauRange = [1, 5]
    precision = 10**-3

    # Parsing variable arguments
    i = 0
    while i < len(args):
        argOkay = True
        if args[i] == 'xmin':
            xmin = args[i+1]
            i += 1
        elif args[i] == 'xmax':
            xmax = args[i+1]
            i += 1
        elif args[i] == 'tauRange':
            tauRange = args[i+1]
            i += 1
        elif args[i] == 'precision':
            precision = args[i+1]
            i += 1
        else:
            argOkay = False

        if not argOkay:
            print(f'(PLMLE) Ignoring invalid argument #{i+1}')
        i += 1

    # Error check the precision
    if np.log10(precision) != round(np.log10(precision)):
        raise ValueError("The precision must be a power of ten.")

    # Reshape data
    x = np.reshape(x, (-1, 1))

    # Determine data type
    if np.any(np.abs(x - np.round(x)) > 3 * np.finfo(float).eps):
        dataType = 'CONT'
    else:
        dataType = 'INT'
        x = np.round(x)

    # Truncate data
    z = x[(x >= xmin) & (x <= xmax)].flatten()
    unqZ = np.unique(z)
    nZ = len(z)
    nUnqZ = len(unqZ)
    allZ = np.arange(xmin, xmax + 1)
    nallZ = len(allZ)

    # MLE calculation
    r = xmin / xmax
    nIterations = int(-np.log10(precision))

    for iIteration in range(nIterations):
        spacing = 10 ** (-iIteration)
        
        if iIteration == 0:
            taus = np.arange(tauRange[0], tauRange[1] + spacing, spacing)
        else:
            if tauIdx == 0:
                taus = np.arange(taus[0], taus[1] + spacing, spacing)
            elif tauIdx == len(taus) - 1:
                taus = np.arange(taus[-2], taus[-1] + spacing, spacing)
            else:
                taus = np.arange(taus[tauIdx-1], taus[tauIdx+2], spacing)
        
        nTaus = len(taus)
        
        if dataType == 'INT':
            # Replicate arrays to equal size
            allZMat = np.tile(allZ.reshape(nallZ, 1), (1, nTaus)).astype(float)
            tauMat = np.tile(taus, (nallZ, 1)).astype(float)
            
            # Compute the log-likelihood function
            L = - np.log(np.sum(allZMat ** (-tauMat), axis=0)) - (taus / nZ) * np.sum(np.log(z))
            
        elif dataType == 'CONT':
            # Compute the log-likelihood function (method established via Deluca and Corral 2013)
            L = np.log((taus - 1) / (1 - r ** (taus - 1))) - taus * (1 / nZ) * np.sum(np.log(z)) - (1 - taus) * np.log(xmin)
            
            # Handle case when tau == 1 separately
            if 1 in taus:
                L[taus == 1] = -np.log(np.log(1 / r)) - (1 / nZ) * np.sum(np.log(z))
        
        tauIdx = np.argmax(L)
    
    # Pick the tau value that maximizes the log-likelihood function
    tau = taus[tauIdx]

    return tau, xmin, xmax, L


def plplottool(data, *args, ax=None):
    """
    Visualizes power-law distributed data and plots the probability distribution on a log-log scale.

    Parameters:
    data (list or np.array or dict): Data to be plotted. If a list of arrays or a dictionary of arrays is provided,
                                     it should contain multiple datasets to be plotted.
    (..., 'plotParams', plotParams): Dictionary providing information about fits to be plotted.
    (..., 'fitParams', fitParams): Dictionary providing information about fits to be plotted on top of data.
    (..., 'uniqueBins', unique_bins): String determining the nature of the bins. 'on' uses unique bins,
                                      'off' uses logarithmically spaced bins (default: 'on').
    (..., 'binDensity', bin_density): Number of logarithmically spaced bins per order of magnitude (default: 50).
    (..., 'plot', plot_status): String determining if the plot will be shown (default: 'on').
    (..., 'title', title_string): String to be shown as the title of the plot (default: '').
    (..., 'ax', ax): Matplotlib axes on which to plot the data. If None, a new figure and axes will be created.
    
    Returns:
    dict: Contains the organized plotted results for further analysis or plotting.
    """
    if not isinstance(data, (list, tuple)):
        data = [data]

    # Default values
    xmin = np.array([np.min(d) for d in data])
    xmax = np.array([np.max(d) for d in data])
    n_data = len(data)
    bin_density = 50
    unique_bins = 'on'
    plot_status = 'on'
    title_string = ''
    plot_params = {
        'color': cm.jet(np.linspace(0, 1, n_data)) if n_data >= 5 else [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        'linewidth': 1,
        'linestyle': '-',
        'dot': 'off',
        'dotsize': 5
    }
    fit_params = {
        'tau': [],
        'color': [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]],
        'linewidth': 1,
        'linestyle': '--',
        'dot': 'off',
        'dotsize': 5,
        'x2fit': np.ones(1000).astype(int)
    }

    # Parse command line arguments
    i = 0
    while i < len(args):
        if args[i] == 'uniqueBins':
            unique_bins = args[i+1]
            i += 2
        elif args[i] == 'binDensity':
            bin_density = args[i+1]
            i += 2
        elif args[i] == 'fitParams':
            new_fit_params = args[i+1]
            i += 2
        elif args[i] == 'plotParams':
            new_plot_params = args[i+1]
            i += 2
        elif args[i] == 'plot':
            plot_status = args[i+1]
            i += 2
        elif args[i] == 'title':
            title_string = args[i+1]
            i += 2
        elif args[i] == 'ax':
            ax = args[i+1]
            i += 2
        else:
            print(f'(Visualization) Ignoring invalid argument #{i + 1}')
            i += 1

    # Update plot parameters with new values
    if 'new_plot_params' in locals():
        plot_params.update(new_plot_params)

    # Update fit parameters with new values
    if 'new_fit_params' in locals():
        fit_params.update(new_fit_params)

    # Create ax if it does not exist
    if ax is None:
        _, ax = plt.subplots

    # Plotting the data
    data_dict = {'x': [], 'fit': []}
    all_x_bounds = []

    for i, d in enumerate(data):
        d = d.reshape(-1)
        if np.any(np.abs(d - np.round(d)) > 3 * np.finfo(float).eps):
            # Continuous data, use logarithmic bins
            x_edges = np.logspace(np.log10(xmin[i]), np.log10(xmax[i]), int((np.log10(xmax[i]) - np.log10(xmin[i])) * bin_density))
            x_centers = (x_edges[:-1] + x_edges[1:]) / 2
            fit_params['round'] = 'cont'
        else:
            # Discrete data, use unique bins or logarithmic bins
            if unique_bins == 'on':
                d = np.round(d)
                x_edges = np.unique(d)
                x_centers = (x_edges[:-1] + x_edges[1:]) / 2
                fit_params['round'] = 'disc'
            else:
                x_edges = np.logspace(np.log10(xmin[i]), np.log10(xmax[i]), int((np.log10(xmax[i]) - np.log10(xmin[i])) * bin_density))
                x_centers = (x_edges[:-1] + x_edges[1:]) / 2
                fit_params['round'] = 'cont'

        # Bin the data
        y, _ = np.histogram(d, bins=x_edges)
        y = y / len(d)

        all_x_bounds.append((x_edges, x_centers))

        # Plotting
        if plot_status == 'on':
            if plot_params['dot'] == 'off':
                ax.plot(x_centers, y, color=plot_params['color'][i], linewidth=plot_params['linewidth'], linestyle=plot_params['linestyle'])
            else:
                ax.plot(x_centers, y, color=plot_params['color'][i], linewidth=plot_params['linewidth'], linestyle=plot_params['linestyle'],
                         marker='o', markersize=plot_params['dotsize'], markerfacecolor=plot_params['color'][i])

        data_dict['x'].append((x_centers, y))

    # Plot fits
    if 'tau' in fit_params and len(fit_params['tau']) > 0:
        n_fits = len(fit_params['tau'])
        if n_fits > 4 and len(fit_params['color']) == 4:
            fit_params['color'] = cm.jet(np.linspace(0, 1, n_fits))

        if 'xmin' not in fit_params:
            fit_params.setdefault('xmin', np.zeros(n_fits))
            for i in range(n_fits):
                fit_params['xmin'][i] = np.min(data[fit_params['x2fit'][i]])
        
        if 'xmax' not in fit_params:
            fit_params.setdefault('xmax', np.zeros(n_fits))
            for i in range(n_fits):
                fit_params['xmax'][i] = np.max(data[fit_params['x2fit'][i]])

        for i in range(n_fits):
            if fit_params['round'][fit_params['x2fit'][i]] == 'disc':
                norm_factor = np.sum((data[fit_params['x2fit'][i]] >= fit_params['xmin'][i]) &
                                     (data[fit_params['x2fit'][i]] <= fit_params['xmax'][i])) / len(data[fit_params['x2fit'][i]])
                x_fit = np.arange(fit_params['xmin'][i], fit_params['xmax'][i] + 1)
                y_fit = x_fit ** (-fit_params['tau'][i])
            else:
                x_start = np.searchsorted(all_x_bounds[fit_params['x2fit'][i]][0], fit_params['xmin'][i])
                x_end = np.searchsorted(all_x_bounds[fit_params['x2fit'][i]][0], fit_params['xmax'][i])
                norm_factor = np.sum((data[fit_params['x2fit'][i]] >= all_x_bounds[fit_params['x2fit'][i]][0][x_start]) &
                                     (data[fit_params['x2fit'][i]] <= all_x_bounds[fit_params['x2fit'][i]][0][x_end])) / len(data[fit_params['x2fit'][i]])
                x_fit = all_x_bounds[fit_params['x2fit'][i]][1][x_start:x_end]
                u_lim = all_x_bounds[fit_params['x2fit'][i]][0][x_start + 1:x_end + 1]
                l_lim = all_x_bounds[fit_params['x2fit'][i]][0][x_start:x_end]

                if fit_params['tau'][i] != 1:
                    y_fit = (1 / (1 - fit_params['tau'][i])) * ((u_lim ** (1 - fit_params['tau'][i])) - (l_lim ** (1 - fit_params['tau'][i])))
                else:
                    y_fit = np.log(u_lim / l_lim)

            y_fit = y_fit * (norm_factor / np.sum(y_fit))
            data_dict['fit'].append((x_fit, y_fit))

            if plot_status == 'on':
                if fit_params['dot'] == 'off':
                    ax.plot(x_fit, y_fit, color=fit_params['color'][i], linewidth=fit_params['linewidth'], linestyle=fit_params['linestyle'])
                else:
                    ax.plot(x_fit, y_fit, color=fit_params['color'][i], linewidth=fit_params['linewidth'], linestyle=fit_params['linestyle'],
                             marker='o', markersize=fit_params['dotsize'], markerfacecolor=fit_params['color'][i])

    # Add title, labels, legend, and log-log scale
    if plot_status == 'on':
        legend_text = [f'Data {i+1}' for i in range(n_data)]
        if 'tau' in fit_params and len(fit_params['tau']) > 0:
            legend_text += [f'Fit {i+1}' for i in range(n_fits)]
        ax.legend(legend_text)
        ax.set_xlabel('x')
        ax.set_ylabel('p(x)')
        if title_string:
            ax.set_title(title_string)
        ax.set_xscale('log')
        ax.set_yscale('log')

    return data_dict

# Example usage:
# x = power_law_dist(10000)
# print(x)  # Generate perfectly power-law distributed data

# data = plplottool(x, 'plot')  # Generate and plot the data

# fitParams = {'tau': [1.5], 'color': [[1, 0, 0]]}
# plotParams = {'dot': 'on'}
# data = plplottool(x, 'plotParams', plotParams, 'fitParams', fitParams)

import itertools
from scipy.special import comb

def plparams(x, *args):
    """
    Automated computation of power-law parameters using MLE.

    Higher level macro and automated algorithm that computes the exponent for power-law distributed data
    and its uncertainty, the p-value and KS statistic for the fit, and searches for optimal support [xmin, xmax].
    The function utilizes a "smart" greedy search algorithm to settle on a support pair. Prior to initiating the greedy search,
    all support pairs for which xmin and xmax differ by 1 are removed; the remainder are then sorted by log(xmax/xmin).

    Parameters:
    x (np.array): Random data that we would like to fit to the power-law distribution p(x) ~ x^(-tau) for x >= xmin and x <= xmax.

    Variable Inputs:
    (..., 'samples', nSamples): The number of sample distributions to draw (default: 500).
    (..., 'threshold', pCrit): Critical p-value to halt the computation if the likelihood of a successful trial drops below a pre-determined value (default: 0.2).
    (..., 'likelihood', likelihood): Likelihood threshold for binomial process (default: 1e-3).

    Returns:
    tuple:
        tau (float): Slope of power law region.
        xmin (float): Lower truncation of distribution.
        xmax (float): Upper truncation of distribution.
        sigmaTau (float): Error of the tau original fit estimated using the samples drawn from the model fit.
        p (float): Proportion of sample distributions with KS statistics larger than the KS statistic between the empirical pdf and the model distribution.
        pCrit (float): Critical p-value used to truncate computation.
        ks (float): Kolmogorov-Smirnov statistic between the empirical pdf and the model.

    Example usage:
    x = bentpl(10000)
    tau, xmin, xmax, sigma, p, pCrit, ks = plparams(x)
    tau = plparams(x, 'threshold', 1)
    tau, _, _, _, p = plparams(x, 'threshold', 1, 'samples', 1000)
    """

    # Default values
    nSamples = 500
    pCrit = 0.2
    likelihood = 1e-3

    # Parsing variable arguments
    i = 0
    while i < len(args):
        if args[i] == 'samples':
            nSamples = args[i+1]
            i += 2
        elif args[i] == 'threshold':
            pCrit = args[i+1]
            i += 2
        elif args[i] == 'likelihood':
            likelihood = args[i+1]
            i += 2
        else:
            print(f'(PLPARAMS) Ignoring invalid argument #{i + 1}')
            i += 1

    nX = len(x)
    x = np.reshape(x, (nX, 1))
    unqX = np.unique(x)

    # Get all support pairs
    support = np.array(list(itertools.combinations(unqX, 2)))

    # Remove adjacent unique points
    to_remove = []
    for i, val in enumerate(unqX):
        rows, _ = np.where(support == val)
        idx = rows[0]
        to_remove.append(idx)
    support = np.delete(support, to_remove, axis=0)

    rInv = support[:, 1] / support[:, 0]
    nSupport = support.shape[0]
    lnRInv = np.log(rInv) / np.log(np.max(rInv))
    rank = lnRInv ** 2
    idx = np.argsort(rank)[::-1]
    support = support[idx, :]

    # Initiate greedy search for optimal support
    sweepFlag = True
    iSupport = 0
    while sweepFlag and iSupport < nSupport:
        xmin, xmax = support[iSupport, :]
        results = plmle(x, 'xmin', xmin, 'xmax', xmax)
        tau = results[0]
        p,_,_ = pvcalc(x, tau, 'xmin', xmin, 'xmax', xmax, 'samples', nSamples, 'threshold', pCrit, 'likelihood', likelihood)
        if p >= pCrit:
            sweepFlag = False
        else:
            iSupport += 1

    p, ks, sigmaTau = pvcalc(x, tau, 'xmin', xmin, 'xmax', xmax, 'samples', nSamples, 'threshold', 1)
    return tau, xmin, xmax, sigmaTau, p, pCrit, ks

# # Example usage
# x = gendata(100000, {’truncated_powerlaw’, [1.5, 0.125, 10, 75]});
# tau, xmin, xmax, sigma, p, pCrit, ks = plparams(x)
# print(f'tau: {tau}, xmin: {xmin}, xmax: {xmax}, sigmaTau: {sigma}, p-value: {p}, critical p-value: {pCrit}, KS statistic: {ks}')

def pvcalc(x, tau, *args):
    """
    Compute p-value for the power-law probability distribution fit.

    Calculates the p-value by Monte Carlo. The method uses a raw probability 
    density function (pdf) with a model that assumes a power law distribution 
    with exponent tau for the section of the data bounded by support [xmin, xmax]. 
    The function generates many sample distributions using a model that is 
    composed of a power-law within [xmin, xmax]. p is the proportion of sample 
    distributions with KS statistics larger than the KS statistic between the 
    original pdf and the model distribution. For computational efficiency, the 
    function continually updates the likelihood of successful results using the 
    binomial distribution and halts for statistically unlikely results.

    Parameters:
    x (np.array): Empirical data, assumed to be power-law distributed.
    tau (float): The exponent of the power-law distribution.

    Variable Inputs:
    (..., 'xmin', xmin): Sets lower truncation of distribution (default: min(x)).
    (..., 'xmax', xmax): Sets upper truncation of distribution (default: max(x)).
    (..., 'samples', nSamples): The number of sample distributions to draw (default: 500).
    (..., 'threshold', pCrit): Critical p-value to halt computation early if the likelihood 
                               of a successful trial drops below a pre-determined value (default: 1).
    (..., 'likelihood', likelihood): Likelihood threshold for binomial process (default: 1e-3).

    Returns:
    tuple:
        p (float): Proportion of sample distributions with KS statistics larger than 
                   the KS statistic between the empirical pdf and the model distribution.
        ks (float): Kolmogorov-Smirnov statistic between the empirical pdf and the model.
        sigmaTau (float): Error of the tau original fit estimated using the samples drawn from the model fit.
                          If p is small, this error is not valid.

    Example usage:
    x = power_law_dist(10**4)
    tau = plmle(x)
    p, ks, sigma = pvcalc(x, tau)
    p = pvcalc(x, tau, 'threshold', 1)
    p = pvcalc(x, tau, 'likelihood', 1e-5)
    """
    
    # Default values
    xmin = np.min(x)
    xmax = np.max(x)
    nSamples = 500
    pCrit = 1
    likelihood = 1e-3

    # Parsing variable arguments
    i = 0
    while i < len(args):
        if args[i] == 'xmin':
            xmin = args[i+1]
            i += 2
        elif args[i] == 'xmax':
            xmax = args[i+1]
            i += 2
        elif args[i] == 'samples':
            nSamples = args[i+1]
            i += 2
        elif args[i] == 'threshold':
            pCrit = args[i+1]
            i += 2
        elif args[i] == 'likelihood':
            likelihood = args[i+1]
            i += 2
        else:
            print(f'(PVCALC) Ignoring invalid argument #{i + 1}')
            i += 1

    x = np.reshape(x, (len(x), 1))

    if np.all(np.isclose(x, np.round(x))):  # integer data
        pdf_x = np.histogram(x, bins=np.arange(xmin, xmax+2), density=False)[0]
        nSupportEvents = np.sum(pdf_x).astype(int)
        pdf_x = pdf_x / nSupportEvents
        pdfFit = (np.arange(xmin, xmax + 1)) ** (-tau)
        pdfFit = pdfFit / np.sum(pdfFit)
        cdf_x = 1 - np.cumsum(pdf_x)
        cdfFit = 1 - np.cumsum(pdfFit)
    else:  # continuous data
        x = x[(x >= xmin) & (x <= xmax)]
        sorted_x = np.sort(x, axis=0).flatten()
        cdf_x = np.arange(1, len(x) + 1) / len(x)
        cdfFit = (1 - (sorted_x ** (1 - tau)) - (1 - xmin ** (1 - tau))) / ((1 - xmax ** (1 - tau)) - (1 - xmin ** (1 - tau)))

    empirical_ks = np.max(np.abs(cdf_x - cdfFit))
    ks = [empirical_ks, np.zeros(nSamples)]
    successCounts = np.zeros(nSamples, dtype=int)
    nSuccesses = 0
    thisLikelihood = 1
    binomialFlag = True 
    criticalThreshold = nSamples * pCrit
    iSample = 0
    sampleTau = np.zeros(nSamples)

    while iSample < nSamples and thisLikelihood > likelihood and binomialFlag:
        if np.all(np.isclose(x, np.round(x))):  # integer data
            xSampleNo = mymnrnd(nSupportEvents, pdfFit)
            xSample = rldecode(xSampleNo, np.arange(xmin, xmax + 1))
            pdfSample = np.histogram(xSample, bins=np.arange(xmin, xmax+2), density=False)[0]
            pdfSample = pdfSample / nSupportEvents

            if pCrit == 1:
                results = plmle(xSample, 'xmin', xmin, 'xmax', xmax)
                thisTau = results[0]
                sampleTau[iSample] = thisTau

            cdfSample = 1 - np.cumsum(pdfSample)
        else:  # continuous
            xSample = pldist(len(x), 'exponential', tau, 'continuous', 'xmin', xmin, 'xmax', xmax)
            if pCrit == 1:
                results = plmle(xSample, 'xmin', xmin, 'xmax', xmax)
                thisTau = results[0]
                sampleTau[iSample] = thisTau

            sortedSample = np.sort(xSample.flatten())
            cdfSample = np.arange(1, len(x) + 1) / len(x)
            cdfFit = (1 - (sortedSample ** (1 - tau)) - (1 - xmin ** (1 - tau))) / ((1 - xmax ** (1 - tau)) - (1 - xmin ** (1 - tau)))

        sampleKS = np.max(np.abs(cdfSample - cdfFit))
        ks[1][iSample] = sampleKS

        if empirical_ks <= sampleKS:
            successCounts[iSample] = 1
            nSuccesses += 1

        if nSuccesses == criticalThreshold:
            binomialFlag = False

        if pCrit != 1:
            thisLikelihood = 1 - binom.cdf(criticalThreshold - nSuccesses - 1, nSamples - iSample - 1, pCrit)

        iSample += 1

    p = np.sum(successCounts) / nSamples
    sigmaTau = np.std(sampleTau) if iSample == nSamples else np.nan

    return p, ks, sigmaTau

# # Example usage
# x = power_law_dist(10000)
# tau = plmle(x)
# p, ks, sigma = pvcalc(x, tau)
# print(f'p-value: {p}, KS statistic: {ks}, Sigma Tau: {sigma}')

def mymnrnd(n, p):
    """
    Custom-made multinomial random number generator.

    This function generates multinomially distributed random numbers similar to MATLAB's `mnrnd`.
    It uses binomial, Poisson, and normal (Gaussian) approximations based on input parameters to 
    efficiently generate the random numbers.

    Parameters:
    ----------
    n : int
        Number of trials for each multinomial outcome (sample size). Must be a positive integer.
    p : np.array
        A 1-D array of multinomial probabilities. `p` must sum to 1, and each entry must be non-negative.
        Length of `p` (K) determines the number of bins or categories. Must be greater than 1.

    Returns:
    -------
    y : np.array
        A 1-D array containing the counts for each of the K multinomial bins. The length of `y` will be 
        the same as the length of `p`.

    Raises:
    ------
    ValueError:
        If the sum of probabilities `p` is not equal to 1.
        If any entry in `p` is negative.
        If `n` or any of the parameters are not appropriate (non-integer n, inappropriate types).

    Example:
    -------
    >>> y = mymnrnd(1000, np.array([0.2, 0.3, 0.5]))
    >>> print(y)
    array([195, 297, 508])

    Notes:
    -----
    This implementation follows a similar logic to the MATLAB `mnrnd` function, with additional
    checks and optimizations for large `n`. The function uses different probabilistic distributions 
    to approximate the counts in each bin:
      - Normal distribution for large categories.
      - Poisson approximation for smaller categories.
      - Binomial distribution as a fall-back for other cases.

    The custom approximations aim to balance precision and computational efficiency.

    """

    # Constants for the threshold between Poisson and normal approximations.
    coeff = [-1.060102688009665, -0.781485955904560]

    # Validate inputs
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("The number of trials n must be a positive integer.")
        
    if not isinstance(p, np.ndarray) or p.ndim != 1 or len(p) < 2:
        raise ValueError("The probability vector p must be a 1-D numpy array with more than one element.")
        
    if not np.isclose(np.sum(p), 1):
        raise ValueError("The probabilities in p must sum to 1.")
        
    if np.any(p < 0):
        raise ValueError("Probabilities in p must be non-negative.")

    # Number of multinomial bins or categories.
    k = len(p)

    # Sort the probabilities.
    sortedp = np.sort(p)
    idx = np.argsort(p)
    unsortidx = np.argsort(idx)

    y = np.zeros_like(p, dtype=int)

    # Set initial values.
    iSmProb = 0
    iBgProb = k - 1
    nTemp = n
    renormFactor = 1.0
    
    for i in range(k - 1):
        if nTemp >= 1000:
            # Use Poisson or Gaussian approximations.
            threshold = (2**coeff[1]) * (nTemp**(coeff[0] / np.log2(10)))
            
            if ((sortedp[iBgProb] / renormFactor) >= threshold) and ((sortedp[iBgProb] / renormFactor) <= (1 - threshold)):
                # Use a normal distribution to approximate the largest p.
                mean = nTemp * sortedp[iBgProb] / renormFactor
                stddev = np.sqrt(nTemp * (sortedp[iBgProb] / renormFactor) * (1 - (sortedp[iBgProb] / renormFactor)))
                y[iBgProb] = int(np.round(norm.rvs(loc=mean, scale=stddev)))
                
                # Recalculate nTemp.
                nTemp -= y[iBgProb]
                
                # Adjust the renormalization factor.
                renormFactor -= sortedp[iBgProb]
                
                # Adjust the index for the highest probability.
                iBgProb -= 1
            else:
                # Use a Poisson distribution to approximate the smallest p.
                rate = nTemp * sortedp[iSmProb] / renormFactor
                y[iSmProb] = poisson.rvs(rate)
                
                # Recalculate nTemp.
                nTemp -= y[iSmProb]
                
                # Adjust the renormalization factor.
                renormFactor -= sortedp[iSmProb]
                
                # Adjust the index for the lowest probability.
                iSmProb += 1
        else:
            # Use binomial random number generator.
            prob = sortedp[iSmProb] / renormFactor
            y[iSmProb] = binom.rvs(nTemp, prob)
            
            # Recalculate nTemp.
            nTemp -= y[iSmProb]
            
            # Adjust the renormalization factor.
            renormFactor -= sortedp[iSmProb]
            
            # Adjust the index for the lowest probability.
            iSmProb += 1
    
    # Fill in the last random number
    y[iSmProb] = nTemp

    # Reorder the counts.
    y = y[unsortidx]
    
    return y

# Example usage:
# y = mymnrnd(1000, np.array([0.2, 0.3, 0.5]))
# print(y)
