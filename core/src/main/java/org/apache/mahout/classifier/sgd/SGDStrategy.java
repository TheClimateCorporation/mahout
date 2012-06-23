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

import org.apache.mahout.math.Vector;

import java.util.Iterator;

public class SGDStrategy {

  private PriorFunction prior;

  public SGDStrategy(PriorFunction prior) {
    this.prior = prior;
  }

  public void applyPrior(SGDLearner learner, Vector updateSteps, Vector instance, Vector beta) {
    Iterator<Vector.Element> nonZeros = instance.iterateNonZero();
    while (nonZeros.hasNext()) {
      Vector.Element updateLocation = nonZeros.next();
      int j = updateLocation.index();
      double missingUpdates = learner.getStep() - updateSteps.get(j);
      if (missingUpdates > 0) {
        double rate = learner.getLambda() * learner.currentLearningRate() * learner.perTermLearningRate(j);
        double newValue = prior.age(beta.get(j), missingUpdates, rate);
        beta.set(j, newValue);
        updateSteps.set(j, learner.getStep());
      }
    }
  }

  public void applyGradient(SGDLearner learner, Vector instance, Vector beta, double gradientBase, Vector updateSteps, Vector updateCounts) {
    // apply the gradientBase to beta
    Iterator<Vector.Element> nonZeros = instance.iterateNonZero();
    while (nonZeros.hasNext()) {
      Vector.Element updateLocation = nonZeros.next();
      int j = updateLocation.index();

      double newValue = beta.getQuick(j) + gradientBase * learner.currentLearningRate() * learner.perTermLearningRate(j) * instance.get(j);
      beta.setQuick(j, newValue);
    }
    // remember that these elements got updated
    Iterator<Vector.Element> i = instance.iterateNonZero();
    while (i.hasNext()) {
      Vector.Element element = i.next();
      int j = element.index();
      updateSteps.setQuick(j, learner.getStep());
      updateCounts.setQuick(j, updateCounts.getQuick(j) + 1);
    }
    learner.nextStep();
  }
}
