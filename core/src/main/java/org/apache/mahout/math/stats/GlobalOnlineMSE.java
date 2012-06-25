/**
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

package org.apache.mahout.math.stats;

import com.google.common.base.Preconditions;
import org.apache.commons.math.stat.descriptive.summary.Sum;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Random;

/**
 * Computes a running estimate of AUC (see http://en.wikipedia.org/wiki/Receiver_operating_characteristic).
 * <p/>
 * Since AUC is normally a global property of labeled scores, it is almost always computed in a
 * batch fashion.  The probabilistic definition (the probability that a random element of one set
 * has a higher score than a random element of another set) gives us a way to estimate this
 * on-line.
 *
 * @see GroupedOnlineAuc
 */
public class GlobalOnlineMSE implements OnlineMSE {
  enum ReplacementPolicy {
    FIFO, FAIR, RANDOM
  }

  public static final int HISTORY = 50;

  // defines the exponential averaging window for results
  private int windowSize = Integer.MAX_VALUE;

  // FIFO has distinctly the best properties as a policy.  See OnlineAucTest for details
  private ReplacementPolicy policy = ReplacementPolicy.FIFO;
  private final Random random = RandomUtils.getRandom();
  private Vector errors;
  private long samples;

  public GlobalOnlineMSE() {
    errors = new DenseVector(HISTORY);
    errors.assign(Double.NaN);
    samples = 0;
  }

  private double loss(double value, double prediction){
    return (value - prediction)*(value - prediction);
  }

  @Override
  public double addSample(double value, double prediction) {
      switch (policy) {
        case FIFO:
          errors.set((int) samples % HISTORY, loss(value,prediction));
          break;
        case FAIR:
          int j1 = random.nextInt((int) samples + 1);
          if (j1 < HISTORY) {
            errors.set(j1, loss(value,prediction));
          }
          break;
        case RANDOM:
          int j2 = random.nextInt(HISTORY);
          errors.set(j2, loss(value,prediction));
          break;
        default:
          throw new IllegalStateException("Unknown policy: " + policy);
      }

    samples++;

    return mse();
  }

  @Override
  public double mse() {
    double sum = 0.0;
    int count = 0;
    for(Vector.Element element : errors){
      if(!(Double.isNaN(element.get()))){
        sum = sum + element.get();
        count++;
      }
    }
    return sum / count;
  }

  public double value() {
    return mse();
  }

  @Override
  public void setPolicy(ReplacementPolicy policy) {
    this.policy = policy;
  }

  @Override
  public void setWindowSize(int windowSize) {
    this.windowSize = windowSize;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(windowSize);
    out.writeInt(policy.ordinal());
    out.writeLong(samples);
    VectorWritable.writeVector(out,errors);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    windowSize = in.readInt();
    policy = ReplacementPolicy.values()[in.readInt()];
    samples = in.readLong();
    errors = VectorWritable.readVector(in);
  }

}
