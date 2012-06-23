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

package org.apache.mahout.classifier.sgd;

import com.google.common.base.Preconditions;
import org.apache.mahout.classifier.AbstractVectorClassifier;
import org.apache.mahout.classifier.AbstractVectorRegression;
import org.apache.mahout.classifier.OnlineLearner;
import org.apache.mahout.classifier.OnlineRegressionLearner;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.Functions;

import java.util.Iterator;

/**
 * Generic definition of a 1 of n logistic regression classifier that returns probabilities in
 * response to a feature vector.  This classifier uses 1 of n-1 coding where the 0-th category
 * is not stored explicitly.
 * <p/>
 * Provides the SGD based algorithm for learning a logistic regression, but omits all
 * annealing of learning rates.  Any extension of this abstract class must define the overall
 * and per-term annealing for themselves.
 */
public abstract class AbstractOnlineRegression extends AbstractVectorRegression implements OnlineRegressionLearner {
    // coefficients for the classification.  This is a dense matrix
    protected Vector beta;

    protected int step;

    // information about how long since coefficient rows were updated.  This allows lazy regularization.
    protected Vector updateSteps;

    // information about how many updates we have had on a location.  This allows per-term
    // annealing a la confidence weighted learning.
    protected Vector updateCounts;

    // weight of the prior on beta
    private double lambda = 1.0e-5;
    protected PriorFunction prior;

    // can we ignore any further regularization when doing classification?
    private boolean sealed;

    // by default we don't do any fancy training
    private RegressionGradient gradient = new DefaultRegressionGradient();

    /**
     * Chainable configuration option.
     *
     * @param lambda New value of lambda, the weighting factor for the prior distribution.
     * @return This, so other configurations can be chained.
     */
    public AbstractOnlineRegression lambda(double lambda) {
        this.lambda = lambda;
        return this;
    }

    /**
     * Computes the inverse link function, by default the logistic link function.
     *
     * @param v The output of the linear combination in a GLM.  Note that the value
     *          of v is disturbed.
     * @return A version of v with the link function applied.
     */
    public Vector link(Vector v) {
        double max = v.maxValue();
        if (max >= 40) {
            // if max > 40, we subtract the large offset first
            // the size of the max means that 1+sum(exp(v)) = sum(exp(v)) to within round-off
            v.assign(Functions.minus(max)).assign(Functions.EXP);
            return v.divide(v.norm(1));
        } else {
            v.assign(Functions.EXP);
            return v.divide(1 + v.norm(1));
        }
    }

    /**
     * Computes the binomial logistic inverse link function.
     *
     * @param r The value to transform.
     * @return The logit of r.
     */
    public double link(double r) {
        if (r < 0.0) {
            double s = Math.exp(r);
            return s / (1.0 + s);
        } else {
            double s = Math.exp(-r);
            return 1.0 / (1.0 + s);
        }
    }

    public double linearCombination(Vector instance) {
        return beta.dot(instance);
    }

    /**
     * Returns a single scalar probability in the case where we have two categories.  Using this
     * method avoids an extra vector allocation as opposed to calling classify() or an extra two
     * vector allocations relative to classifyFull().
     *
     * @param instance The vector of features to be classified.
     * @return The probability of the first of two categories.
     * @throws IllegalArgumentException If the classifier doesn't have two categories.
     */
    @Override
    public double predict(Vector instance) {
        Preconditions.checkArgument(instance.size() == beta.size(), "Can only call predict with instance of same dimension as beta");

        // apply pending regularization to whichever coefficients matter
        regularize(instance);

        return linearCombination(instance);
    }

    @Override
    public void train(long trackingKey, String groupKey, double actual, Vector instance) {
        unseal();

        double learningRate = currentLearningRate();

        // push coefficients back to zero based on the prior
        regularize(instance);

        // update each row of coefficients according to result
        double gradient = this.gradient.apply(groupKey, actual, instance, this);

            // then we apply the gradientBase to the resulting element.
            Iterator<Vector.Element> nonZeros = instance.iterateNonZero();
            while (nonZeros.hasNext()) {
                Vector.Element updateLocation = nonZeros.next();
                int j = updateLocation.index();

                double newValue = beta.getQuick(j) + gradient * learningRate * perTermLearningRate(j) * instance.get(j);
                beta.setQuick(j, newValue);
            }

        // remember that these elements got updated
        Iterator<Vector.Element> i = instance.iterateNonZero();
        while (i.hasNext()) {
            Vector.Element element = i.next();
            int j = element.index();
            updateSteps.setQuick(j, getStep());
            updateCounts.setQuick(j, updateCounts.getQuick(j) + 1);
        }
        nextStep();

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

        // anneal learning rate
        double learningRate = currentLearningRate();

        // here we lazily apply the prior to make up for our neglect
        Iterator<Vector.Element> nonZeros = instance.iterateNonZero();
        while (nonZeros.hasNext()) {
            Vector.Element updateLocation = nonZeros.next();
            int j = updateLocation.index();
            double missingUpdates = getStep() - updateSteps.get(j);
            if (missingUpdates > 0) {
                double rate = getLambda() * learningRate * perTermLearningRate(j);
                double newValue = prior.age(beta.get(j), missingUpdates, rate);
                beta.set(j, newValue);
                updateSteps.set(j, getStep());
            }
        }
    }

    // these two abstract methods are how extensions can modify the basic learning behavior of this object.

    public abstract double perTermLearningRate(int j);

    public abstract double currentLearningRate();

    public void setPrior(PriorFunction prior) {
        this.prior = prior;
    }

    public void setGradient(RegressionGradient gradient) {
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

    protected void nextStep() {
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

    public void copyFrom(AbstractOnlineRegression other) {
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

}

