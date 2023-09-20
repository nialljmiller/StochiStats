import gpytorch
import torch
from gpytorch.kernels import RBFKernel, PeriodicKernel, ScaleKernel, AdditiveKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
import matplotlib.pyplot as plt
import tqdm as tqdm

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        rbf_kernel = RBFKernel()
        periodic_kernel1 = PeriodicKernel(period_length_prior=gpytorch.priors.SmoothedBoxPrior(0.1, 1.0))
        periodic_kernel2 = PeriodicKernel(period_length_prior=gpytorch.priors.SmoothedBoxPrior(0.1, 1.0))
        self.covar_module = AdditiveKernel(
            ScaleKernel(rbf_kernel),
            ScaleKernel(AdditiveKernel(periodic_kernel1))#, periodic_kernel2))
        )
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



def GP(x, y, yerr):
    # Convert data to PyTorch tensors
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    y = y - torch.mean(y)
    yerr = torch.tensor(yerr, dtype=torch.float32)


    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(x, y, likelihood)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)



    training_iterations = 1000
    for i in tqdm(range(training_iterations)):
        optimizer.zero_grad()
        output = model(x)
        loss = -mll(output, y)
        loss.backward()
        optimizer.step()

    fit_lightcurve(x, y, yerr, model, likelihood)

    # After optimization, extract parameters
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        predicted = model(x)
        rbf_params = model.covar_module.kernels[0]#.base_kernel  # RBFKernel parameters
        period_params0 = model.covar_module.kernels[1].base_kernel.kernels[0]# PeriodicKernel parameters
#        period_params1 = model.covar_module.kernels[1].base_kernel.kernels[1]# PeriodicKernel parameters

    return rbf_params.base_kernel.lengthscale.item(), rbf_params.raw_outputscale.item(), 10**period_params0.raw_period_length.item()#, period_params1.raw_period_length.item()


def fit_lightcurve(x, y, yerr, model, likelihood):
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(min(x), max(x), 1000)  # Generate test points
        observed_pred = likelihood(model(test_x))
        pred_mean = observed_pred.mean
        lower, upper = observed_pred.confidence_region()

    # Plot the fit and uncertainty
    plt.figure(figsize=(10, 6))
    plt.errorbar(x, y, yerr=yerr, fmt='o', label='Observed Data', markersize=5)
    plt.plot(test_x, pred_mean, 'b', label='GP Mean')
    plt.fill_between(test_x, lower, upper, alpha=0.2, color='blue', label='GP Confidence Region')
    plt.xlabel('Time')
    plt.ylabel('Flux')
    plt.legend()
    plt.title('GP Fit with Uncertainty')
    plt.show()


