

// Define global macros for dependent variable (Y), independent variables (X), and treatment variable (D)
global Y y
global X x1 x2 x3 i.id
global D d

// Set random seed for reproducibility
set seed 42

// Initialize double/debiased machine learning with partial linear model and 5-fold cross-validation
ddml init partial, kfolds(5)

// First stage: Estimate conditional expectation of treatment D given covariates X using neural network
ddml E[D|X]: pystacked $D $X, type(reg) method(nnet)

// First stage: Estimate conditional expectation of outcome Y given covariates X using neural network
ddml E[Y|X]: pystacked $Y $X, type(reg) method(nnet)

// Perform cross-fitting to reduce overfitting bias in machine learning estimates
ddml crossfit

// Estimate the treatment effect with robust standard errors
ddml estimate, robust