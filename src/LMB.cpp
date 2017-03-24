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

LMB::LMB(double l, double s)
: lambda(l), lscale(s), currerr(0), iteration(0)
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

  StoppingInterrupt interrupt;
  while(step() && !interrupt.isSignaled())
  {
    ++iteration;
    OPENANN_DEBUG << "Iteration #" << iteration
                  << ", training error = "
                  << currerr
                  << ", lambda = "
                  << lambda
                  << ", ||grad|| = "
                  << JE.norm();
  }
  OPENANN_DEBUG << "Terminated:";
  OPENANN_DEBUG << iteration << " iterations";
  OPENANN_DEBUG << "Error = " << currerr;
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
  double newerr = 0;
  lambda /= 10.0;
  do
  {
    lambda *= 10.0;
    opt->setParameters(Wb);
    opt->errorJacobian(currerr, E, J);
    JE = J.transpose()*E/opt->examples(); 
    JJ = J.transpose()*J/opt->examples();
    optimum = Wb - (JJ + lambda*I).ldlt().solve(JE);
    opt->setParameters(optimum);
    newerr = opt->error();
  }
  while(newerr > currerr && lambda < 1e10);
  
  // Scale lambda lower since it did a good job.
  lambda = std::max(lambda/10.0, 1.0e-20);

  const bool run = (stop.maximalIterations == // Maximum iterations reached?
                    StoppingCriteria::defaultValue.maximalIterations ||
                    iteration < stop.maximalIterations) &&
                   (stop.minimalSearchSpaceStep == // Gradient too small?
                    StoppingCriteria::defaultValue.minimalSearchSpaceStep ||
                    JE.norm() >= stop.minimalSearchSpaceStep) &&
                  (stop.minimalValueDifferences == // No function improvement?
                   StoppingCriteria::defaultValue.minimalValueDifferences ||
                   std::abs(newerr - preverr) >= stop.minimalValueDifferences);
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