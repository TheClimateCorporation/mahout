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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.classifier.sgd.PolymorphicWritable;
import org.apache.mahout.classifier.sgd.SGDStrategy;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.VectorWritable;

import com.google.common.base.Preconditions;

/**
 * Extends the basic on-line logistic regression learner with a specific set of learning
 * rate annealing schedules.
 */
public class OnlineLinearPredictor extends AbstractOnlineLinearPredictor implements Writable {
  public static final int WRITABLE_VERSION = 1;

  // these next two control decayFactor^steps exponential type of annealing
  // learning rate and decay factor
  private double mu0 = 0.01;
  private double decayFactor = 1 - 1.0e-3;

  // these next two control 1/steps^forget type annealing
  private int stepOffset = 10;
  // -1 equals even weighting of all examples, 0 means only use exponential annealing
  private double forgettingExponent = -0.5;

  // controls how per term annealing works
  private int perTermAnnealingOffset = 20;

  public int numFeatures() {
    return numFeatures;
  }

  private int numFeatures;

  public OnlineLinearPredictor() {
    // private constructor available for serialization, but not normal use
  }

  public OnlineLinearPredictor(int numFeatures, SGDStrategy strategy) {
    this.strategy = strategy;
    this.numFeatures = numFeatures;

    updateSteps = new DenseVector(numFeatures);
    updateCounts = new DenseVector(numFeatures).assign(perTermAnnealingOffset);
    beta = new DenseVector(numFeatures);
  }

  /**
   * Chainable configuration option.
   *
   * @param alpha New value of decayFactor, the exponential decay rate for the learning rate.
   * @return This, so other configurations can be chained.
   */
  public OnlineLinearPredictor alpha(double alpha) {
    this.decayFactor = alpha;
    return this;
  }

  /**
   * Chainable configuration option.
   *
   * @param learningRate New value of initial learning rate.
   * @return This, so other configurations can be chained.
   */
  public OnlineLinearPredictor learningRate(double learningRate) {
    Preconditions.checkArgument(learningRate < 1.0 && learningRate > 0.0 && Double.NaN != learningRate,
            "learning rate must be in (0,1)");
    this.mu0 = learningRate;
    return this;
  }

  public OnlineLinearPredictor stepOffset(int stepOffset) {
    this.stepOffset = stepOffset;
    return this;
  }

  public OnlineLinearPredictor decayExponent(double decayExponent) {
    if (decayExponent > 0) {
      decayExponent = -decayExponent;
    }
    this.forgettingExponent = decayExponent;
    return this;
  }


  @Override
  public double perTermLearningRate(int j) {
    return Math.sqrt(perTermAnnealingOffset / updateCounts.get(j));
  }

  @Override
  public double currentLearningRate() {
    return mu0 * Math.pow(decayFactor, getStep()) * Math.pow(getStep() + stepOffset, forgettingExponent);
  }

  public void copyFrom(OnlineLinearPredictor other) {
    super.copyFrom(other);
    mu0 = other.mu0;
    decayFactor = other.decayFactor;
    strategy = other.strategy.copy();

    stepOffset = other.stepOffset;
    forgettingExponent = other.forgettingExponent;

    perTermAnnealingOffset = other.perTermAnnealingOffset;
  }

  public OnlineLinearPredictor copy() {
    close();
    OnlineLinearPredictor r = new OnlineLinearPredictor(beta.size(), strategy.copy());
    r.copyFrom(this);
    return r;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(WRITABLE_VERSION);
    out.writeDouble(mu0);
    out.writeDouble(decayFactor);
    out.writeInt(stepOffset);
    out.writeInt(step);
    out.writeDouble(forgettingExponent);
    out.writeInt(perTermAnnealingOffset);
    VectorWritable.writeVector(out, beta);
    PolymorphicWritable.write(out, strategy);
    VectorWritable.writeVector(out, updateCounts);
    VectorWritable.writeVector(out, updateSteps);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    int version = in.readInt();
    if (version == WRITABLE_VERSION) {
      mu0 = in.readDouble();
      decayFactor = in.readDouble();
      stepOffset = in.readInt();
      step = in.readInt();
      forgettingExponent = in.readDouble();
      perTermAnnealingOffset = in.readInt();
      beta = VectorWritable.readVector(in);
      strategy = PolymorphicWritable.read(in, SGDStrategy.class);
      updateCounts = VectorWritable.readVector(in);
      updateSteps = VectorWritable.readVector(in);
    } else {
      throw new IOException("Incorrect object version, wanted " + WRITABLE_VERSION + " got " + version);
    }
  }
}
