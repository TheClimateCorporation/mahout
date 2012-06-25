/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.regression.sgd;

import com.google.common.base.Preconditions;
import org.apache.mahout.classifier.sgd.SGDStrategy;
import org.apache.mahout.regression.AbstractVectorLinearPredictor;
import org.apache.mahout.regression.OnlineLinearPredictorLearner;
import org.apache.mahout.classifier.sgd.PriorFunction;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.classifier.sgd.SGDLearner;

import java.util.Iterator;

/**
 * Generic definition of a linear predictor function.
 * <p/>
 * Provides the SGD based algorithm for learning a regression solution, but omits all
 * annealing of learning rates.  Any extension of this abstract class must define the overall
 * and per-term annealing for themselves.
 */
public abstract class AbstractOnlineLinearPredictor extends AbstractVectorLinearPredictor implements SGDLearner, OnlineLinearPredictorLearner {
  // coefficients for the prediction.  This is a dense Vector.
  protected Vector beta;

  protected int step;

  protected SGDStrategy strategy;

  // information about how long since coefficient rows were updated.  This allows lazy regularization.
  protected Vector updateSteps;

  // information about how many updates we have had on a location.  This allows per-term
  // annealing a la confidence weighted learning.
  protected Vector updateCounts;

  // weight of the prior on beta
  private double lambda = 1.0e-5;
  protected PriorFunction prior;

  // can we ignore any further regularization when doing prediction?
  private boolean sealed;

  // by default we don't do any fancy training
  private LinearPredictorGradient gradient = new DefaultLinerPredictorGradient();

  /**
   * Chainable configuration option.
   *
   * @param lambda New value of lambda, the weighting factor for the prior distribution.
   * @return This, so other configurations can be chained.
   */
  public AbstractOnlineLinearPredictor lambda(double lambda) {
    this.lambda = lambda;
    return this;
  }

  public double linearCombination(Vector instance) {
    return beta.dot(instance);
  }

  /**
   * Returns a single scalar value of the prediction for instance.
   *
   * @param instance The vector of features to be classified.
   * @return The linear prediction.
   * @throws IllegalArgumentException If the the instance dimensionality does not match the coefficient dimensionality.
   */
  @Override
  public double predict(Vector instance) {
    Preconditions.checkArgument(instance.size() == beta.size(),
        "Can only call predict with instance of same dimension as beta");

    // apply pending regularization to whichever coefficients matter
    regularize(instance);

    return linearCombination(instance);
  }

  @Override
  public void train(long trackingKey, String groupKey, double actual, Vector instance) {
    unseal();

    double learningRate = currentLearningRate();
    Preconditions.checkState(learningRate > 0.0 && learningRate < 1.0 && learningRate != Double.NaN,
            "learning rate must be in (0,1");
    // push coefficients back to zero based on the prior
    regularize(instance);

    // update each row of coefficients according to result
    double gradient = this.gradient.apply(groupKey, actual, instance, this);

    strategy.applyGradient(this, instance, beta, gradient, updateSteps, updateCounts);
  }

  @Override
  public void train(long trackingKey, double actual, Vector instance) {
    train(trackingKey, null, actual, instance);
  }

  @Override
  public void train(double actual, Vector instance) {
    Preconditions.checkArgument(instance.size() == beta.size());
    train(0, null, actual, instance);
  }

  public void regularize(Vector instance) {
    if (updateSteps == null || isSealed()) {
      return;
    }
    strategy.applyPrior(this, updateSteps, instance, beta);
  }

  // these two abstract methods are how extensions can modify the basic learning behavior of this object.

  public abstract double perTermLearningRate(int j);

  public abstract double currentLearningRate();

  public void setPrior(PriorFunction prior) {
    this.prior = prior;
  }

  public void setGradient(LinearPredictorGradient gradient) {
    this.gradient = gradient;
  }

  public PriorFunction getPrior() {
    return prior;
  }

  public Vector getBeta() {
    close();
    return beta;
  }

  public void setBeta(int j, double betaIJ) {
    beta.set(j, betaIJ);
  }

  public double getLambda() {
    return lambda;
  }

  public int getStep() {
    return step;
  }

  public void nextStep() {
    step++;
  }

  public boolean isSealed() {
    return sealed;
  }

  protected void unseal() {
    sealed = false;
  }

  private void regularizeAll() {
    Vector all = new DenseVector(beta.like());
    all.assign(1);
    regularize(all);
  }

  @Override
  public void close() {
    if (!sealed) {
      step++;
      regularizeAll();
      sealed = true;
    }
  }

  public void copyFrom(AbstractOnlineLinearPredictor other) {
    beta.assign(other.beta);

    step = other.step;

    updateSteps.assign(other.updateSteps);
    updateCounts.assign(other.updateCounts);
  }

  public boolean validModel() {
    double k = beta.aggregate(Functions.PLUS, new DoubleFunction() {
      @Override
      public double apply(double v) {
        return Double.isNaN(v) || Double.isInfinite(v) ? 1 : 0;
      }
    });
    return k < 1;
  }

  public PriorFunction prior() {
    return prior;
  }
}

