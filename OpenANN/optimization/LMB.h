#ifndef OPENAN_OPTIMIZATION_LMB_H_
#define OPENAN_OPTIMIZATION_LMB_H_

#include <OpenANN/optimization/Optimizer.h>
#include <OpenANN/optimization/StoppingCriteria.h>
#include <OpenANN/util/Random.h>
#include <Eigen/Core>
#include <vector>
#include <list>

namespace OpenANN
{

class LMB : public Optimizer
{
  //! Stopping criteria
  StoppingCriteria stop;
  //! Optimizable problem
  Optimizable* opt; // do not delete
  
  // Error values.
  Eigen::MatrixXd E;
  // Jacobian. 
  Eigen::MatrixXd J;
  // Identity matrix. 
  Eigen::MatrixXd I; 
  // Storage of J'J.
  Eigen::MatrixXd JJ;
  // Storage of J'E (gradient). 
  Eigen::VectorXd JE; 
  // Storage of parameters at the start of a step.
  Eigen::VectorXd Wb;
  // Storage of best parameters.
  Eigen::VectorXd optimum;

  //! Marquardt parameter. 
  double lambda, lambda0; 
  //! Marquardt scaling parameter.
  double lscale;
  //! Maximum Marquardt parameter. 
  double lmax;

  //! Regularization weights. 
  double alpha, beta; 

  //! Effective number of parameters.
  double gamma;

  // Current error and SSE.
  double currerr, sse, swe;

  int iteration; 

public:
  LMB(double lambda = 0.005, double lscale = 10.0, double lmax = 1e10);
  ~LMB(); 

  virtual void setOptimizable(Optimizable& opt);
  virtual void setStopCriteria(const StoppingCriteria& stop);
  virtual void optimize();
  virtual bool step();
  virtual Eigen::VectorXd result();
  virtual std::string name();
};

}
#endif