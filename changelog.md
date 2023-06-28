# List of important changes in Lompe

This document lists important changes in the Lompe code that have happened after the publication of the [Lompe paper](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022JA030356)


#### 2023-06
Changed how weighting of the data is implemented. Instead of the `scale` keyword (deprecated!), the user should now provide `error` and `iweight`. `error` is an array of measurement uncertainties in SI units. It should be non-zero even if the measurement uncertainty is low. `iweight` is a number that represents the relative importance of the dataset compared to others. For example, ground magnetometer data has low measurement uncertainty (error), but their relation to the electric field (which is the primary Lompe model parameter) is indirect and depends on several assumptions (conductance, ground induction contribution, current sheet height, ...). Taking this model uncertainty into account can be done by reducing `iweight` relative to other datasets. Reducing `iweight` and increasing `error` give effectively the same effect, but we use two different keywords in order to keep the measurement and model errors separate. 

Notebooks have been changed to use `iweight` and `error` instead of `scale`. That means that the output may be slightly different from what was presented in the papers. The changes will be described in an upcoming publication. 

#### 2023-04-17
Added changelog!

