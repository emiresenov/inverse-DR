# TODO

- Test different solutions with different parameter values to see how stable convergence is.
- Write a comprehensive report on the entire project.
- Case 3.





# Summary of problems with Case 2 loglog

- We have two expressions summed together in a logarithm – no easy way to split the contributions like Case 2 linear.
- Instead of having two output nodes, we can try to solve the differential equation for the logarithmic sum as a single output. However, the expressions become really messy, and numerical issues begin to creep in due to very high exponential values and dividing by very small numbers.


~~- Come up with official reference solutions for all cases (?). Alternatively, propose a general scaling method.~~

~~- Build eval for all cases.~~

~~- Make loglog versions case 2, build from case 1 loglog~~


~~- Change plot y-axis. Remove R,C letters and use (ohm) instead of (\Omega).~~

~~- Change n-samples for Case 2 and fig size to match Case 0 and 1.~~