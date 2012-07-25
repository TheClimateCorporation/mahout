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
package org.apache.mahout.regression.sgd;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.List;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.classifier.sgd.PolymorphicWritable;
import org.apache.mahout.classifier.sgd.SGDStrategy;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.stats.GlobalOnlineMSE;
import org.apache.mahout.math.stats.OnlineMSE;
import org.apache.mahout.regression.AbstractVectorLinearPredictor;
import org.apache.mahout.regression.OnlineLinearPredictorLearner;

import com.google.common.collect.Lists;

/**
 * Does cross-fold validation of MSE on several online linear regression models. Each record is passed
 * to all but one of the models for training and to the remaining model for evaluation.  In order to
 * maintain proper segregation between the different folds across training data iterations, data should
 * either be passed to this learner in the same order each time the training data is traversed or a
 * tracking key such as the file offset of the training record should be passed with each training example.
 */
public class CrossFoldRegressionLearner extends AbstractVectorLinearPredictor implements OnlineLinearPredictorLearner, Writable {
  private int record;
  private OnlineMSE mse = new GlobalOnlineMSE();
  private final List<OnlineLinearPredictor> models = Lists.newArrayList();

  // lambda, learningRate, perTermOffset, perTermExponent
  private double[] parameters = new double[4];
  private int numFeatures;
  private SGDStrategy strategy;
  private int windowSize = Integer.MAX_VALUE;

  public CrossFoldRegressionLearner() {
  }

  public CrossFoldRegressionLearner(int folds, int numFeatures, SGDStrategy strategy) {
    this.numFeatures = numFeatures;
    this.strategy = strategy;
    for (int i = 0; i < folds; i++) {
      OnlineLinearPredictor model = new OnlineLinearPredictor(numFeatures, strategy);
      model.alpha(1).stepOffset(0).decayExponent(0);
      models.add(model);
    }
  }

  // -------- builder-like configuration methods
  public CrossFoldRegressionLearner learningRate(double x) {
    for (OnlineLinearPredictor model : models) {
      model.learningRate(x);
    }
    return this;
  }

  public CrossFoldRegressionLearner stepOffset(int x) {
    for (OnlineLinearPredictor model : models) {
      model.stepOffset(x);
    }
    return this;
  }

  public CrossFoldRegressionLearner decayExponent(double x) {
    for (OnlineLinearPredictor model : models) {
      model.decayExponent(x);
    }
    return this;
  }

  public CrossFoldRegressionLearner alpha(double alpha) {
    for (OnlineLinearPredictor model : models) {
      model.alpha(alpha);
    }
    return this;
  }

  // -------- training methods
  @Override
  public void train(double actual, Vector instance) {
    train(record, null, actual, instance);
  }

  @Override
  public void train(long trackingKey, double actual, Vector instance) {
    train(trackingKey, null, actual, instance);
  }

  @Override
  public void train(long trackingKey, String groupKey, double actual, Vector instance) {
    record++;
    int k = 0;
    for (OnlineLinearPredictor model : models) {
      if (k == trackingKey % models.size()) {
        double v = model.predict(instance);
        mse.addSample(actual,v);
      } else {
        model.train(trackingKey, groupKey, actual, instance);
      }
      k++;
    }
  }

  @Override
  public void close() {
    for (OnlineLinearPredictor m : models) {
      m.close();
    }
  }

  public void resetLineCounter() {
    record = 0;
  }

  public boolean validModel() {
    boolean r = true;
    for (OnlineLinearPredictor model : models) {
      r &= model.validModel();
    }
    return r;
  }

  // -------- classification methods


  @Override
  public double predict(Vector instance) {
    Vector r = new DenseVector(models.size());
    for (OnlineLinearPredictor model : models){
      r.assign(model.predict(instance));
    }
    return r.aggregate(Functions.PLUS,Functions.IDENTITY) / r.size();
  }


  // -------- status reporting methods

  public double mse() {
    return mse.mse();
  }

  // -------- evolutionary optimization

  public CrossFoldRegressionLearner copy() {
    CrossFoldRegressionLearner r = new CrossFoldRegressionLearner(models.size(), numFeatures, strategy.copy());
    r.models.clear();
    for (OnlineLinearPredictor model : models) {
      model.close();
      OnlineLinearPredictor newModel =
          new OnlineLinearPredictor(model.numFeatures(), model.strategy.copy());
      newModel.copyFrom(model);
      r.models.add(newModel);
    }
    return r;
  }

  public int getRecord() {
    return record;
  }

  public void setRecord(int record) {
    this.record = record;
  }
  
  public SGDStrategy getStrategy() {
	  return strategy;
  }
  
  public void setStrategy(SGDStrategy strategy) {
	  this.strategy = strategy;
  }

  public List<OnlineLinearPredictor> getModels() {
    return models;
  }

  public void addModel(OnlineLinearPredictor model) {
    models.add(model);
  }

  public double[] getParameters() {
    return parameters;
  }

  public void setParameters(double[] parameters) {
    this.parameters = parameters;
  }

  public int getNumFeatures() {
    return numFeatures;
  }

  public void setNumFeatures(int numFeatures) {
    this.numFeatures = numFeatures;
  }

  public void setWindowSize(int windowSize) {
    this.windowSize = windowSize;
    mse.setWindowSize(windowSize);
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(record);
    PolymorphicWritable.write(out, mse);
    out.writeInt(models.size());
    for (OnlineLinearPredictor model : models) {
      model.write(out);
    }

    for (double x : parameters) {
      out.writeDouble(x);
    }
    out.writeInt(numFeatures);
    PolymorphicWritable.write(out, strategy);
    out.writeInt(windowSize);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    record = in.readInt();
    mse = PolymorphicWritable.read(in, OnlineMSE.class);
    int n = in.readInt();
    for (int i = 0; i < n; i++) {
      OnlineLinearPredictor olr = new OnlineLinearPredictor();
      olr.readFields(in);
      models.add(olr);
    }
    parameters = new double[4];
    for (int i = 0; i < 4; i++) {
      parameters[i] = in.readDouble();
    }
    numFeatures = in.readInt();
    strategy = PolymorphicWritable.read(in, SGDStrategy.class);
    windowSize = in.readInt();
  }


  public void setMSEEvaluator(OnlineMSE mse) {
    this.mse = mse;
  }
}
