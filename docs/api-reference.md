# API Reference

## varateMC.data

### load_mining_data(data_dir="data", filename="coal_mining_accident_data.dat")
Loads the source matrix, flattens it in Fortran order, computes cumulative event times,
and returns aligned event indices.

### mean_rates(total_events, total_period, number_of_accidents)
Computes mean event rates in per-day and per-year units.

## varateMC.priors

### order_statistics_pdfs(x, total_period)
Returns two dictionaries of prior curves (`plain` and `even`) evaluated at `x`.

## varateMC.constant_rate

### log_constant_rate_prior(h, alpha, beta)
Log prior kernel for a constant Poisson rate under a Gamma prior parameterization.

### constant_rate_likelihood(h, total_events, total_period)
Likelihood kernel for a homogeneous Poisson process.

### log_constant_rate_likelihood(h, total_events, total_period)
Log-likelihood kernel.

### log_constant_rate_posterior(h, alpha, beta, total_events, total_period)
Unnormalized log posterior as prior plus log-likelihood.

### evaluate_constant_rate_grid(total_events, total_period, alpha=1, beta=200, n=1000)
Builds a grid of rate values and evaluates prior/posterior diagnostics.

### exp_scaled_log_integrand(h, total_events, total_period)
Scaled integrand used in numerical evidence approximation.

### compute_scaled_evidence(total_events, total_period, lower=0.0, upper=1.0)
Runs quadrature over the scaled integrand.

## varateMC.change_point

### one_change_log_likelihood(params, event_times, total_period)
Log-likelihood for a single change point with rates before/after the breakpoint.

### one_change_log_prior(params, total_period, alpha=1, beta_param=200)
Log-prior over rates and change-point location.

### one_change_log_posterior(params, event_times, total_period, alpha=1, beta_param=200)
Combined unnormalized log posterior.

### initialize_one_change_walkers(...)
Creates valid positive-rate walker initialization for `emcee`.

### run_one_change_emcee(...)
Executes ensemble MCMC and returns chain and diagnostics.

### one_change_prior_transform(u, total_period, beta_param=200)
Unit-cube to physical-parameter transform for nested sampling.

### run_one_change_nested(...)
Runs nested sampling with `dynesty` and returns sampler + results.

## varateMC.rjmcmc

### load_chains(path="mcmc_chains.npy")
Loads serialized RJMCMC chain state arrays.

### extract_model_sizes(chains)
Infers change-point counts from compact state encoding.

### post_burn_in_chain(chains, burn_in=10000)
Drops warm-up states.

### map_model_size(k_samples)
Returns the modal model-size value.

### evaluate_rates_across_chain(chain, total_period, n_eval=1000)
Evaluates each chain state on a shared time grid.

### summarize_rate_bands(rate_evaluations)
Computes median and central 50%/90% posterior bands.

## varateMC.stats

### height_prior_pdf(h, alpha, beta)
Evaluates the Gamma-family prior density for a segment rate.

### change_point_prior_pdf(s, L)
Evaluates a Dirichlet prior over normalized segment gaps implied by ordered change points.
