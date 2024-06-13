# capachinos
Analyzing EEG Data Obtained from Anesthesiology Studies

- **Experiment** 
	- Anesthesia status data:
		- minimum alveolar concentration (MAC)
		- ...
	- Make sure all event marks are available. Otherwise, use proper surrogate markers. 
	- Define distinct analysis periods based on the time stamps corresponding to the clinically relevant events, such as 
		- pre-operative eyes-closed resting (baseline) 
		- pre-oxygenation, 
		- loss of consciousness (LOC), 
		- anesthetic maintenance, 
		- peri-extubation
		- Anesthetic emergence (drug infusion),
		- PACU eyes-closed resting.
	
- **EEG preprocessing** 
	-  Take the EEG signals as input. The original data in the BrainVision EEG data format, made up from three separate files:
		- *.vhdr*: A text header file with metadata, including recording parameters
		- *.vmrk*: A text marker file with information about events in the data
		- *.eeg*: A binary data file with the EEG voltage values and other signals recorded at the same time
	- Downsample the EEG to 250 Hz (Nyquist frequency = 125 Hz. We might need restrict the bandwidth, before downsampling)
	- Detrend (see [Vlisides et al., 2019]([https://doi.org/10.1097/ALN.0000000000002677](https://doi.org/10.1097/ALN.0000000000002677)): using a local linear regression method with a 10-s window at a 5-s step size in Chronux analysis toolbox)
	- Re-reference the signal to the average of all EEG signals
	- Bandpass filter from 0.5 to 45 Hz (or 55 Hz) ([Vlisides et al., 2019]([https://doi.org/10.1097/ALN.0000000000002677](https://doi.org/10.1097/ALN.0000000000002677)): via a fifth-order Butterworth filter using a zero-phase forward and reverse algorithm)
	- Plot the raw EEG waveform to
		- Identify bad channels and their rejection 
		- identify noisy periods and marking them for later removal.
			- Reject epochs containing amplitudes larger than 250 $\micro V$. 
			- Li et al., 2019 Divide the signal into 2-s windows and reject 2-s windows if its average amplitude was more than four times the average, or its SD as greater than 2 SD value of the whole signal, in at least one channel. Do this separately for the different epochs.  
			- Mark periods with burst suppression:
				- Calculate instantaneous power at 5–30 Hz
					- bandpass filter the data at 5–30 Hz via a 4-order Butterworth filter 
					- Calculate the envelope using Hilbert transform followed by application of a 0.5 s smoothing window.
				- Apply a threshold calculated from the manually labeled suppression periods (mean + 4 SD). See ref . 18 from Li et al. 2019
	- Apply independent component analysis and remove cardiac artifact, eye blinks, muscle movement, and other artifacts if present (using extended-infomax algorithm in EEGLAB) 
	- Segment each epoch into shorter intervals, such as 20- or 30-second durations (maybe with 50 percent overlap).

- **Power spectral analysis - basic**
	- Whiten the signal - remove the $\frac{1}{f}$ noise (using autoregressive method). 
	- Calculate the power spectral for select EEG channels using multi-taper method in Chronux MATLAB toolbox with window length of 2s with 50% overlap, time-bandwidth product equal to 2 and number of tapers equal to 3 ([Vlisides et al., 2019]([https://doi.org/10.1097/ALN.0000000000002677](https://doi.org/10.1097/ALN.0000000000002677))).
	- Calculate group-level spectrogram by taking (median) averages over anatomically related groups of EEG channels, such as frontal, prefrontal, and parietal regions. 
	- Display the power spectra for each EEG channel together with the experimental time stamps using *Bokeh*.
	- Calculate the spectral power for each channel and oscillation by summing across the frequency band of interest. These frequency bands include: delta 0.5–4 Hz, theta 4–8 Hz, alpha 8–12 Hz, high alpha 10–14 Hz, beta 14–25 Hz, gamma 24–45 Hz.  
	
- **Power spectral analysis - spectral slope and intercept** 
	- Use [Fitting Oscillations and One-Over-F (FOOOF)](http://dx.doi.org/10.1038/s41593-020-00744-x) method to dissect the power spectra within each time-segment  into periodic and aperiodic components. 
	- Package: https://pypi.org/project/fooof/ 
	- Tutorial: https://fooof-tools.github.io/fooof/auto_tutorials/plot_02-FOOOF.html
	- Characterize the aperiodic activity by calculating the offset and exponent. The aperiodic component is not necessarily linear, so the slope might be calculated for different ranges. 
		- The aperiodic component, $L$ is modeled using a Lorentzian function: $L = b  - log(k + F^χ)$,  with $b$ as broadband offset, χ as exponent, and $k$ is the knee parameter. With $k = 0$, this is equal to a line in log-log space, with slope, $a = -χ$.
	- Characterize the periodic activity in terms of center frequencies, power at center frequencies, and frequency bands. 
	- Calculate the peak frequencies changes over time. 
	- The calculated frequency bands using FOOOF might be used in the other analysis, as alternatives to the pre-defined bands. 
	
- **Power spectral analysis - delta vs spindle dominant SWA** 
	- Categorize the patterns into delta or spindle dominant SWA or non-SWA emergence patterns. For details, see Chander et al., 2014. 

- **Power spectral analysis - metastable spectral dynamic**:
	- [Hudson et al., 2014]([https://doi.org/10.1073/pnas.1408296111](https://doi.org/10.1073/pnas.1408296111))
	- Construct a spectral power vector, $X(t) = {x_1, ..., x_n}$ for each time window $t$, containing $n$ elements that represent the proportion of power at different frequency intervals, which span from TBD-TBD Hz. These elements should be aggregated across individual EEG channels or averaged over clusters of EEG channels corresponding to distinct anatomical regions. 
	- Concatenate the spectral power vectors across all participants. 
	- Remove outlier windows
	- Apply PCA for dimensionality reduction 
	- (optional) Make sure that the dimensionality reduction in each participant is comparable to that in the concatenated data set. Fig. S4 analyses:
		- Calculate variance explained by the PCs for each participant. 
		- Calculate the correlation between the points in two feature space, one based on individual animal and one based on concatenated dataset. 
	-  Categorize the dimensionality-reduced feature vectors into $N_c$ clusters. 
	- Determine $N_c$ by quantifying cluster quality measures such as:  
		- silhouette value: $S_i = \frac{(b_i − a_i)} {max(a_i, b_i)}$, where $a_i$ is the average of the distance from $i$-th point to all other points in the same cluster and $b_i$ is the average of the distance from the $i$-th point to points in all other clusters. 
	- (optional) Determine cluster consistency between the participants using the approach in page 1 of [Hudson et al., 2014]([https://doi.org/10.1073/pnas.1408296111](https://doi.org/10.1073/pnas.1408296111)) supplementary information. Briefly, use each participant as *template* and remaining participants as *test*. Use hamming distance between the cluster indices using clustering algorithms from template or test data.
	- (More details on transition probability matrix can be found in the paper)

- **Cortical connectivity analysis - weighted phase lag index**
	- reference: [Vlisides et. al, 2019]([https://doi.org/10.1097/ALN.0000000000002677](https://doi.org/10.1097/ALN.0000000000002677)) 
	- Divide EEG to 30-s windows at a 10-s steps, which further divided into 2-s sub-windows with 50% overlap. 
	- Calculate cross-spectral density in sub-windows of 2-s with 50% overlap, using multi-taper taper method, with time-band-width product = 2 and number of tapers = 3
	- Calculate weighted phase lag index (wPLI) between all pairs of channels within each sub-window using a custom-written function adopted from the Fieldtrip toolbox. 
	- Generate shuffled data, by temporally shuffling one signal while keeping the other signal intact, calculate the shuffle wPLI, and subtract it from the data. 
	- Display the cortical connectivity's evolution over time in a grid structure or using *Bokeh*
	- Calculate inter-regional connectivity by averaging wPLI over anatomical groups:
		- Frontal-parietal 
		- Prefrontal-frontal 
	- Calculate the mean wPLI in defined frequency bands, such as delta 0.5–3 Hz, theta 3–7 Hz, and alpha 7–15 Hz.
	
	*Statistical analysis* 
	- Use [[Statistics/Linear Mixed models|Linear Mixed models]] to test the difference between the epochs in connectivity values at each frequency band.
		- Fixed effects: studied epochs, region pairs, their interactions 
		- Random effect: participants. The model estimates a slope and an intercept for each participant. Calculate a covariance structure of random effects.
	- Post-hoc pairwise comparison between each epoch and the baseline  
	
- **Cortical connectivity analysis - connectivity dynamics**
	- Calculate frontal-parietal and prefrontal-frontal wPLI in 30-s windows with 10-s steps. 
	- Exclude the time windows with suppression ratio greater than 20%. 
	- Construct observation $O(t)$ at each time point as a 140-dimensional vector: 2 sets (frontal-prefrontal $fp$ and frontal-parietal $fr$) of frequency estimates in 0.5–35 Hz with the frequency resolution ($\Delta f$) of 0.5Hz. $O(t) = \left[ f_{fp}(0.5Hz), f_{fp}(0.5+\Delta f), \ldots, f_{fp}(35Hz), f_{fr}(0.5Hz), f_{fr}(0.5+\Delta f), \ldots, f_{fr}(35Hz) \right]$
	- Aggregate $O(t)$ across all time windows across the participants
	- Apply PCA for dimensionality reduction, resulting in $M$-dimensional feature. 
	- Classify the $M$-dimensional vectors into $N_c$ clusters using k-means algorithm. 
	- For each state, calculate the mean frontal-parietal and frontal-prefrontal connectivity patterns by averaging the connectivity patterns over all time windows representing that state. 
	- Determine $M$ and $N_c$ based on:
		- Amount of explained variance by the $M$ principal components 
		- Normalized minimum Hamming distance between different clustering solutions (??)
		- Interpretability of the clustering results
		-ref: [Vlisides et. al, 2019]([https://doi.org/10.1097/ALN.0000000000002677](https://doi.org/10.1097/ALN.0000000000002677))
	- Assigned a ==connectivity state== index to each cluster, characterized by distinct spectral and spatial properties.
		- A time point is assigned a cluster index if it either falls within a predefined distance from a cluster's centroid or is among the top 95% of points closest to the centroid (see  [Hudson et al., 2014]([https://doi.org/10.1073/pnas.1408296111](https://doi.org/10.1073/pnas.1408296111)))
	- Define an additional state as "BS" for all windows with burst suppression (number of states = $N_c + 1$)
	- Calculate connectivity state index time series for each participant. 
	- Calculate the occurrence rate together with dwell time for each state.
	- Calculate a transition matrix for a Markov chain model of the connectivity state time series. One for the transition frequencies/counts and one for transition probabilities, ensure each row in the transition probability matrix sums to exactly one.
		- Exploratory analysis: determine the most probable state of arrival or departure for travel to and from each state. For this, properly normalize the probabilities in row or columns.
	
	*Statistical analysis*
	- To test the statistical significance of the inter-state transitions, perform following steps:
		- Generate $N = 1000$ surrogate time series, by randomly permuting the state identities across windows, after removing the state stays. 
		- Compute the transition probabilities for each surrogate series.
		- Compare the original transition probability with those from the surrogate series by counting the instances where the surrogate transition probabilities match or exceed the original values and then divide by $N$.

- **Criticality analysis - autocorrelation function** 
	- Bandpass filter the EEG signals of 20-s duration into frequency bands of interest, e.g., delta, theta, alpha, etc., using a zero-phase third order Butterworth filter. 
	- Extract the envelope $x(t)$ of the bandpass EEG by taking the absolute value of the Hilbert transform. 
	- Calculate the mean $\mu$ and variance $\nu$ for $x(t)$ 
	- Calculate the ACF for different time lag ($s$) varying between TBD and TBD for each EEG channel signal as 
		$ACF(s) = \frac{\displaystyle\sum_{t=1}^{N-s} (x(t) - \mu)(x(t+s) - \mu)}{\nu}, \: s = 1, ..., \frac{N}{2}$

	- Quantify the autocorrelation function decay by capturing its value at lag one: ACF(1).
	- Plot ACF(1) as time-frequency heatmap 

- **Criticality analysis - phase lag entropy topography** 
	- Bandpass the EEG signal in frequency bands of interest, such as alpha, delta, or a broader low-frequency band 2–20 Hz.  
	- Calculate the instantaneous phase of the signal using the Hilbert transform.
	- Calculate the entropy of phase lag for each channel pair using the following procedure:
		- Calculate the phase difference of the two signals $\Delta\theta_t$
		- Binarize the phase difference: $s_t = 1$ if $\Delta\theta_t > 0$ and $s_t = 0$ if $\Delta\theta_t < 0$
		- Calculate a pattern of phase differences around each time point as $S_t = \{ s_t, s_{t+l}, ...,s_{t+(m-1)l} \}, \; t = 1,2, ..., T-(m-1)l$ where $m$ is the length of the pattern and $l$ is the time lag. We can start with $m=3$ and $l = 1$, that yields eight phase lead-lag patterns. 
		- Calculate the probability $p_k$ of each phase lead-lag pattern across the period of interest. 
		- Calculate PLE as: $PLE=-\frac{1}{log(2^m)}\displaystyle\sum_{k} p_k\;log(p_k)$
	- Compute the average phase lag entropy for each channel by averaging the phase lag entropy values obtained between that channel and all other channels.
	- Map the topography of the average phase lag entropy across the specified frequency bands and during key periods, including baseline and various epochs of anesthetic emergence.
	- Compare the spatial pattern of phase lag entropy between the emergence and baseline periods. To do this, we can either find the correlation between the pair-wise phase lag entropy values (n = $nCh^2$) or between the averages (n = $nCh$).
	
- **Criticality analysis - Neuronal avalanches**
	- Divide the EEG signals into bins (various bin durations can be used to evaluate the dependency of the results on the time bin duration). 
	- Calculate the z-scored EEG signals. 
	- Detect avalanches as periods with excursions beyond a threshold - typically two standard deviations, though various thresholds can be tested - on one or more EEG channels. 
	- Calculate the silence periods$-$defined as periods of no threshold exceedances on any EEG channels$-$preceding and following the detected avalanches. 
	- Calculate the duration of each avalanche as the time between threshold crossings
	- Calculate the size of each avalanche as the summation of z-scored activity across all channels as in Scarpetta et al., 2023, or max z-score activity across all EEG channels. 
	- Characterize the distribution of avalanche durations and sizes. Previous studies have indicated that the avalanche sizes and durations follow power-law distributions. Here are the steps:
		-  (optional) check to what extent the data follow power law or alternatives such exponential distributions. Use likelihood ratio for comparing between the distributions ([Scarpetta et. al, 2023](https://doi.org/10.1016/j.isci.2023.107840)). 
		- Characterize the power law distributions of avalanche sizes and durations in terms of their power law exponents: $$
f(S) \propto S^{-\tau}, \:
f(T) \propto D^{-\alpha}
$$Use the [powerlaw Python package](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0085777) to estimate the power law exponents using maximum likelihood estimation. The estimation method computes the exponents by minimizing the [Kolmogorov-Smirnov distance] between the original and fitted values. See [Ma et al., 2019](https://doi.org/10.1016/j.neuron.2019.08.031) for other methods/implementations. 
		- Analyze the relationship between avalanche sizes and durations. To test whether average avalanche size scaled with duration according to $$
<s>\;  \propto D^\beta 
$$, estimate the fitted $\beta$ by applying linear regression to the data on a log10-log10 scale [[Avalanche_size_duration_relationship.png]].
		- Calculate theoretical $\beta^\prime$ as $$
\beta^\prime = \frac{\alpha - 1}{\tau -1} 
$$
		- Calculate the deviation from criticality (DCC) as the absolute difference between $\beta$ and theoretical criticality index $\beta^\prime$ as $DCC = |\beta - \beta^\prime|$

- **Compare caffeine with placebo** 
	- Investigate the relationship between each emergence trajectory (spectral or connectivity) and placebo/caffeine, utilizing univariate or multiple regression analysis. 

- **Misc. Statistical analysis**
	- For multiple regression analysis diagnose multicollinearity among the predictors using:
		- Scatterplots (variable 1 vs variable 2) together with the correlation matrix. Correlation values above 0.7 or 0.8 indicate multicollinearity. 
		- Variance inflation factor (VIF). VIFs above 10 or 5 (less permissive) can be used as threshold. 
		- Remove one of the correlated variables, especially if it's not theoretically important. or replace the variables by calculating an index based on their combination. 
		- *lasso regression* can deal with multilinearity by penalizing the coefficients of the regression model. 



[[Definition of Object Classes and Methods]]
