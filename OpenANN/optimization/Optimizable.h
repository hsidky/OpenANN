#ifndef OPENANN_OPTIMIZATION_OPTIMIZABLE_H_
#define OPENANN_OPTIMIZATION_OPTIMIZABLE_H_
#if __GNUC__ >= 4
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <Eigen/Core>
#include <vector>

namespace OpenANN
{

/**
 * @class Optimizable
 *
 * Represents an optimizable object.
 *
 * An optimizable object can be arbitrary as long as it provides some
 * objective function. The objective function in this context is called error
 * function and will be minimized. This could be e.g. the sum of squared error
 * between predictions and targets for a neural network. But the idea behind
 * this is more general. We can e.g. optimize the reward in a reinforcement
 * learning problem or the energy of an unsupervised model like an RBM.
 */
class Optimizable
{
public:
  virtual ~Optimizable() {}

  /**
   * @name Batch Methods
   * Functions that must be implemented in every Optimizable.
   */
  ///@{
  /**
   * Check if the object knows how to initialize its parameters.
   * @return does the optimizable object provide a parameter initialization?
   */
  virtual bool providesInitialization() = 0;
  /**
   * Initialize the optimizable parameters.
   */
  virtual void initialize() = 0;
  /**
   * Request the number of optimizable parameters.
   * @return number of optimizable parameters
   */
  virtual unsigned dimension() = 0;
  /**
   * Request the current parameters.
   * @return current parameters
   */
  virtual const Eigen::VectorXd& currentParameters() = 0;
  /**
   * Set new parameters.
   * @param parameters new parameters
   */
  virtual void setParameters(const Eigen::VectorXd& parameters) = 0;
  /**
   * Compute error on training set.
   * @return current error on training set or objective function value
   */
  virtual double error() = 0;
  /**
   * Check if the object provides a gradient of the error function with
   * respect to its parameters.
   * @return does the optimizable provide a gradient?
   */
  virtual bool providesGradient() = 0;
  /**
   * Compute gradient of the error function with respect to the parameters.
   * @return gradient
   */
  virtual Eigen::VectorXd gradient() = 0;
  ///@}

  /**
   * @name Mini-batch Methods
   * Functions that should be implemented to speed up optimization and are
   * required by some optimization algorithms.
   */
  ///@{
  /**
   * Request number of training examples.
   * @return number of training examples
   */
  virtual unsigned examples() { return 1; }
  /**
   * Compute error of a given training example.
   * @param n index of the training example in the dataset
   * @return error of the n-th training example
   */
  virtual double error(unsigned n) { return error(); }
  /**
   * Compute gradient of a given training example.
   * @param n index of the training example in the dataset
   * @return gradient of the n-th training example
   */
  virtual Eigen::VectorXd gradient(unsigned n) { return gradient(); }
  /**
   * Calculates the function value and gradient of a training example.
   * @param n index of training example
   * @param value function value
   * @param grad gradient of the function, lenght must be dimension()
   */
  virtual void errorGradient(int n, double& value, Eigen::VectorXd& grad);
  /**
   * Calculates the function value and gradient of all training examples.
   * @param value function value
   * @param grad gradient of the function, lenght must be dimension()
   */
  virtual void errorGradient(double& value, Eigen::VectorXd& grad);

  /**
   * Calculates the function value and the Jacobian of all training examples. 
   * This varies specifically from errorGradient as it returns F(x) - T, where 
   * F(X) is the NN output and T is the target data, and dF_i/dP_j where P_j 
   * is the jth parameter. 
   * @param toterr Value of the error function. 
   * @param value Error matrix of size (examples, outputs). 
   * @param jac Jacobian matrix of size (examples, parameters). 
   */
  virtual void errorJacobian(double& value, Eigen::MatrixXd& error, Eigen::MatrixXd& jac){}

  /**
   * Calculates the errors of given training examples.
   * @param startN iterator over index vector
   * @param endN iterator over index vector
   * @return each row contains the error for one training example
   */
  virtual Eigen::VectorXd error(std::vector<int>::const_iterator startN,
                                std::vector<int>::const_iterator endN);
  /**
   * Calculates the accumulated gradient of given training examples.
   * @param startN iterator over index vector
   * @param endN iterator over index vector
   * @return each row contains the gradient for one training example
   */
  virtual Eigen::VectorXd gradient(std::vector<int>::const_iterator startN,
                                   std::vector<int>::const_iterator endN);
  /**
   * Calculates the accumulated gradient and error of given training examples.
   * @param startN iterator over index vector
   * @param endN iterator over index vector
   * @param value function value
   * @param grad gradient of the function, lenght must be dimension()
   * @return each row contains the gradient for one training example
   */
  virtual void errorGradient(std::vector<int>::const_iterator startN,
                             std::vector<int>::const_iterator endN,
                             double& value, Eigen::VectorXd& grad);
  ///@}

  /**
   * This callback is called after each optimization algorithm iteration.
   */
  virtual void finishedIteration() {}
};

} // namespace OpenANN

#endif // OPENANN_OPTIMIZATION_OPTIMIZABLE_H_
