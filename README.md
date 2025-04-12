# Code for inverse modeling of DR in time domain using PINNs

## Summary of the code

- Examples follow the suggested structure of the [JAX-PI library](https://github.com/PredictiveIntelligenceLab/jaxpi) examples and depend on our [forked version](https://github.com/emiresenov/jaxpi) for inverse modeling and custom subnet architecture.
- Case 0-2 considers only scalar parameters without any temperature effects. Remaining examples are temperature-dependent.

---
### Case 0

- One RC branch
- Analytical solution:
  
$$I(t) =\frac{U}{R_{1}} \cdot \exp \left( - \frac{t}{R_{1}C_{1}} \right)$$

- Initial condition:

$$I(t=0) = \frac{U}{R_{1}}$$

- Residual:

$$\frac{dI}{dt} + \frac{I}{R_{1}C_{1}} = 0$$

- Objective: recover $R_1, C_1$
---
### Case 1

- One steady-state branch, one RC branch
- Analytical solution:
  
$$I(t) = \frac{U}{R_{0}} + \frac{U}{R_{1}} \cdot \exp \left(  - \frac{t}{R_{1}C_{1}} \right) $$

- Initial condition:

$$I(t=0) = \frac{U}{R_{0}} + \frac{U}{R_{1}}$$

- Residual:

$$\frac{dI}{dt} + \frac{1}{R_{1}C_{1}} \left( I-\frac{U}{R_{0}} \right)  = 0$$

- Objective: recover $R_0, R_1, C_1$
---
### Case 2

- One steady-state branch, two RC branches
- Analytical solution:

$$I(t) = \frac{U}{R_{0}} + \frac{U}{R_{1}} \cdot \exp \left(  - \frac{t}{R_{1}C_{1}} \right) + \exp \left(  - \frac{t}{R_{2}C_{2}} \right)$$

- Initial condition:
  
$$I(t=0) = \frac{U}{R_{0}} + \frac{U}{R_{1}} + \frac{U}{R_{2}}$$

- Residuals:

$$\frac{dI_{01}}{dt} + \frac{1}{R_{1}C_{1}} \left( I-\frac{U}{R_{0}} \right)  = 0$$
$$\frac{dI_2}{dt} + \frac{I}{R_{2}C_{2}} = 0$$

- Objective: recover $R_0, R_1, C_1, R_2, C_2$
---
### $R_0(T)$ Case 1

![](doc/subnets.drawio.svg)

- One temperature-dependent steady-state branch, one RC branch
- Since resistance values coming from Arrhenius function had high numerical variance, leading to highly varied current values, we solved temperature-dependent cases with logarithmic current 
- Analytical solution:

$$\ln(I) = \ln \left( \frac{U}{R_{0}(T)} + \frac{U}{R_{1}} \cdot \exp \left( -\frac{t}{R_{1}C_{1}} \right)  \right)$$

- Initial condition:

$$\ln \left( I(t=0) \right) = \ln \left( \frac{U}{R_{0}} + \frac{U}{R_{1}} \right)$$

- Residual:

$$\frac{d\ln(I)}{dt} + \frac{1}{R_{1}C_{1}} \left( 1 - \frac{U}{R_{0}(T)} e^{ - \ln(I) }\right)$$

---
### $R_0(T), R_1(T)$ Case 1

- Analagous to previous case, except $R_1(T)$ is now a temperature-dependent function instead of a scalar.
