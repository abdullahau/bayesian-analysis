# Check transitions that ended with a divergence
check_div <- function(fit, quiet=FALSE) {
  sampler_params <- get_sampler_params(fit, inc_warmup=FALSE)
  divergent <- do.call(rbind, sampler_params)[,'divergent__']
  n = sum(divergent)
  N = length(divergent)

  if (!quiet) cat(sprintf('%s of %s iterations ended with a divergence (%s%%)\n',
                  n, N, 100 * n / N))
  if (n > 0) {
    if (!quiet) cat('  Try running with larger adapt_delta to remove the divergences\n')
    if (quiet) return(FALSE)
  } else {
    if (quiet) return(TRUE)
  }
}

# Check transitions that ended prematurely due to maximum tree depth limit
check_treedepth <- function(fit, max_depth = 10, quiet=FALSE) {
  sampler_params <- get_sampler_params(fit, inc_warmup=FALSE)
  treedepths <- do.call(rbind, sampler_params)[,'treedepth__']
  n = length(treedepths[sapply(treedepths, function(x) x == max_depth)])
  N = length(treedepths)

  if (!quiet)
    cat(sprintf('%s of %s iterations saturated the maximum tree depth of %s (%s%%)\n',
                n, N, max_depth, 100 * n / N))
  
  if (n > 0) {
    if (!quiet) cat('  Run again with max_treedepth set to a larger value to avoid saturation\n')
    if (quiet) return(FALSE)
  } else {
    if (quiet) return(TRUE)
  }
}

# Checks the energy fraction of missing information (E-FMI)
check_energy <- function(fit, quiet=FALSE) {
  sampler_params <- get_sampler_params(fit, inc_warmup=FALSE)
  no_warning <- TRUE
  for (n in 1:length(sampler_params)) {
    energies = sampler_params[n][[1]][,'energy__']
    numer = sum(diff(energies)**2) / length(energies)
    denom = var(energies)
    if (numer / denom < 0.2) {
      if (!quiet) cat(sprintf('Chain %s: E-FMI = %s\n', n, numer / denom))
      no_warning <- FALSE
    }
  }
  if (no_warning) {
    if (!quiet) cat('E-FMI indicated no pathological behavior\n')
    if (quiet) return(TRUE)
  } else {
    if (!quiet) cat('  E-FMI below 0.2 indicates you may need to reparameterize your model\n')
    if (quiet) return(FALSE)
  }
}

# Checks the effective sample size per iteration
check_n_eff <- function(fit, quiet=FALSE) {
  fit_summary <- summary(fit, probs = c(0.5))$summary
  N <- dim(fit_summary)[[1]]

  iter <- dim(extract(fit)[[1]])[[1]]

  no_warning <- TRUE
  for (n in 1:N) {
    ratio <- fit_summary[,5][n] / iter
    if (ratio < 0.001) {
      if (!quiet) cat(sprintf('n_eff / iter for parameter %s is %s!\n',
                      rownames(fit_summary)[n], ratio))
      no_warning <- FALSE
    }
  }
  if (no_warning) {
    if (!quiet) cat('n_eff / iter looks reasonable for all parameters\n')
    if (quiet) return(TRUE)
  }
  else {
    if (!quiet) cat('  n_eff / iter below 0.001 indicates that the effective sample size has likely been overestimated\n')
    if (quiet) return(FALSE)
  }
}

# Checks the potential scale reduction factors
check_rhat <- function(fit, quiet=FALSE) {
  fit_summary <- summary(fit, probs = c(0.5))$summary
  N <- dim(fit_summary)[[1]]

  no_warning <- TRUE
  for (n in 1:N) {
    rhat <- fit_summary[,6][n]
    if (rhat > 1.1 || is.infinite(rhat) || is.nan(rhat)) {
      if (!quiet) cat(sprintf('Rhat for parameter %s is %s!\n',
                      rownames(fit_summary)[n], rhat))
      no_warning <- FALSE
    }
  }
  if (no_warning) {
    if (!quiet) cat('Rhat looks reasonable for all parameters\n')
    if (quiet) return(TRUE)
  } else {
    if (!quiet) cat('  Rhat above 1.1 indicates that the chains very likely have not mixed\n')
    if (quiet) return(FALSE)
  }
}

check_all_diagnostics <- function(fit, quiet=FALSE) {
  if (!quiet) {
    check_n_eff(fit)
    check_rhat(fit)
    check_div(fit)
    check_treedepth(fit)
    check_energy(fit)
  } else {
    warning_code <- 0
    
    if (!check_n_eff(fit, quiet=TRUE))
      warning_code <- bitwOr(warning_code, bitwShiftL(1, 0))
    if (!check_rhat(fit, quiet=TRUE))
      warning_code <- bitwOr(warning_code, bitwShiftL(1, 1))
    if (!check_div(fit, quiet=TRUE))
      warning_code <- bitwOr(warning_code, bitwShiftL(1, 2))
    if (!check_treedepth(fit, quiet=TRUE))
      warning_code <- bitwOr(warning_code, bitwShiftL(1, 3))
    if (!check_energy(fit, quiet=TRUE))
      warning_code <- bitwOr(warning_code, bitwShiftL(1, 4))

    return(warning_code)
  }
}

parse_warning_code <- function(warning_code) {
  if (bitwAnd(warning_code, bitwShiftL(1, 0)))
    cat("n_eff / iteration warning\n")
  if (bitwAnd(warning_code, bitwShiftL(1, 1)))
    cat("rhat warning\n")
  if (bitwAnd(warning_code, bitwShiftL(1, 2)))
    cat("divergence warning\n")
  if (bitwAnd(warning_code, bitwShiftL(1, 3)))
    cat("treedepth warning\n")
  if (bitwAnd(warning_code, bitwShiftL(1, 4)))
    cat("energy warning\n")
}

# Returns parameter arrays separated into divergent and non-divergent transitions
partition_div <- function(fit) {
  nom_params <- extract(fit, permuted=FALSE)
  n_chains <- dim(nom_params)[2]
  params <- as.data.frame(do.call(rbind, lapply(1:n_chains, function(n) nom_params[,n,])))

  sampler_params <- get_sampler_params(fit, inc_warmup=FALSE)
  divergent <- do.call(rbind, sampler_params)[,'divergent__']
  params$divergent <- divergent

  div_params <- params[params$divergent == 1,]
  nondiv_params <- params[params$divergent == 0,]

  return(list(div_params, nondiv_params))
}
