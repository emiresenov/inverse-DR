# TODO

## Case 0 field dependence
- Change analytical solution because it is different now
- Update evals where R is constant, figure out how to log this
- Better naming for stuff


# Completed

~~- Write abstract.~~

~~- Experiment with LBFGS~~

~~- Figure out scaling mechanism for very large R and very small C.~~

~~- Case 3.~~

~~- Come up with official reference solutions for all cases (?). Alternatively, propose a general scaling method.~~

~~- Build eval for all cases.~~

~~- Make loglog versions case 2, build from case 1 loglog~~

~~- Change plot y-axis. Remove R,C letters and use (ohm) instead of (\Omega).~~

~~- Change n-samples for Case 2 and fig size to match Case 0 and 1.~~

~~- Write a comprehensive report on the entire project.~~


## Summary of problems with Case 2 loglog

### More importantly, since I is the only input in a real experiment, we can't do this function transformation since we don't have enough information.

- We have two expressions summed together in a logarithm – no easy way to split the contributions like Case 2 linear.
- Instead of having two output nodes, we can try to solve the differential equation for the logarithmic sum as a single output. However, the expressions become really messy, and numerical issues begin to creep in due to very high exponential values and dividing by very small numbers.


## Stability for different parameter values
- Previous TODO: Test different solutions with different parameter values to see how stable convergence is.

Summary: if the problem is well-conditioned (sufficient curvature for each polarization mechanism), we sample enough data (everything except highly sparse sampling), and the numbers are in an appropriate range, R_i, C_i, t ~ [0.1,100] (solved with nondimensionalization), then we can expect to converge to a good approximation of the true parameter values.
