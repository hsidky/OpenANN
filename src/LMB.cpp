#define OPENANN_LOG_NAMESPACE "LMB"

#include <OpenANN/optimization/LMB.h>
#include <OpenANN/optimization/Optimizable.h>
#include <OpenANN/optimization/StoppingCriteria.h>
#include <OpenANN/optimization/StoppingInterrupt.h>
#include <OpenANN/util/AssertionMacros.h>
#include <OpenANN/util/OpenANNException.h>
#include <OpenANN/util/EigenWrapper.h>
#include <OpenANN/io/Logger.h>
#include <Eigen/Dense>

namespace OpenANN 
{

LMB::LMB(double l, double s, double lm)
: lambda(l), lscale(s), lmax(lm), 
  alpha(0), beta(0), gamma(0), currerr(0), iteration(0)
{
}

LMB::~LMB()
{
}

void LMB::setOptimizable(Optimizable& opt)
{
  this->opt = &opt;
}

void LMB::setStopCriteria(const StoppingCriteria& stop)
{
  this->stop = stop;
}

void LMB::optimize()
{
  OPENANN_CHECK(opt);

  // Initialization.

  // Optimizable doesn't provide us with the output size (typically scalar), 
  // but this is resized in the errorJacobian call. 
  E.resize(opt->examples(), 1); 
  J.resize(opt->examples(), opt->dimension());
  I = Eigen::MatrixXd::Identity(opt->dimension(), opt->dimension());
  JJ.resize(opt->dimension(), opt->dimension());
  JE.resize(opt->examples());
  Wb = opt->currentParameters();
  optimum = opt->currentParameters();

  // Initialize regularization parameters.
  currerr = 2.*opt->error();
  gamma = opt->dimension();
  beta = (opt->examples() - gamma)/(2.0*currerr);
  beta = beta <= 0 ? 1. : beta;
  double XX = Wb.transpose()*Wb; 
  alpha = gamma/(2.*XX);
  currerr = beta*currerr + alpha*XX;
  
  StoppingInterrupt interrupt;
  while(step() && !interrupt.isSignaled())
  {
    ++iteration;
    OPENANN_DEBUG << "Iteration #" << iteration
                  << ", training error = "
                  << sse
                  << ", lambda = "
                  << lambda
                  << ", grad = "
                  << 2.*JE.norm()
                  << ", gamma = "
                  << gamma;
  }
  OPENANN_DEBUG << "Terminated:";
  OPENANN_DEBUG << iteration << " iterations";
  OPENANN_DEBUG << "Error = " << sse;
  OPENANN_DEBUG << "Effective params = " << gamma;
  OPENANN_DEBUG << "Marquardt param = " << lambda*lscale;
  iteration = 0;
}

bool LMB::step()
{
  OPENANN_CHECK(opt);

  // Current best parameters.
  Wb = optimum;

  // Iteratre increase our trust region until we see
  // a decrease in the error.
  double preverr = currerr;
  sse = 0, swe = 0;
  lambda /= lscale;
  do
  {
    lambda *= lscale;
    opt->setParameters(Wb);
    opt->errorJacobian(currerr, E, J);
    JE = J.transpose()*E/opt->examples(); 
    JJ = J.transpose()*J/opt->examples();
    optimum = Wb - (beta*JJ + (lambda+alpha)*I).ldlt().solve(beta*JE+alpha*Wb);
    swe = optimum.transpose()*optimum;
    opt->setParameters(optimum);
    sse = 2.*opt->error();
    currerr = beta*sse + alpha*swe;
  }
  while(currerr >= preverr && lambda < lmax);

  // Update regularization parameters. 
  if(lambda <= lmax)
  {
    opt->errorJacobian(currerr, E, J);
    JE = J.transpose()*E/opt->examples(); 
    JJ = J.transpose()*J/opt->examples();
    gamma = opt->dimension() - alpha*(beta*JJ + alpha*I).inverse().trace();
    alpha = swe == 0 ? 1.0 : 0.5*gamma/swe;
    beta = sse == 0 ? 1.0 : 0.5*(opt->examples() - gamma)/sse;
    currerr = beta*sse + alpha*swe;
  }

  const bool run = (stop.maximalIterations == // Maximum iterations reached?
                    StoppingCriteria::defaultValue.maximalIterations ||
                    iteration < stop.maximalIterations) &&
                   (stop.minimalSearchSpaceStep == // Gradient too small?
                    StoppingCriteria::defaultValue.minimalSearchSpaceStep ||
                    2.*JE.norm() >= stop.minimalSearchSpaceStep) &&
                  (stop.minimalValueDifferences == // No function improvement?
                   StoppingCriteria::defaultValue.minimalValueDifferences ||
                   std::abs(currerr - preverr) >= stop.minimalValueDifferences) &&
                  (stop.minimalValue == // Target performance reached? 
                   StoppingCriteria::defaultValue.minimalValue || 
                   sse >= stop.minimalValue) && 
                  (lambda <= lmax); // Maximum trust region radius)

  // Scale lambda lower since it did a good job
  // and assign current error.
  lambda = std::max(lambda/lscale, 1.0e-20);

  return run;
}

std::string LMB::name()
{
  std::stringstream stream;
  stream << "Bayesian Regularized Levenberg-Marquardt";
  return stream.str();
}

Eigen::VectorXd LMB::result()
{
  OPENANN_CHECK(opt);
  opt->setParameters(optimum);
  return optimum;
}

}