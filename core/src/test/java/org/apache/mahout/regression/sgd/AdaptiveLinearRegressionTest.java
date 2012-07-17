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

import java.util.Random;

import org.apache.mahout.classifier.sgd.L1;
import org.apache.mahout.classifier.sgd.PriorSGDStrategy;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.junit.Test;

public final class AdaptiveLinearRegressionTest extends MahoutTestCase {

  @Test
  public void testTrain() {

    Random gen = RandomUtils.getRandom();

    AdaptiveLinearRegression.Wrapper cl = new AdaptiveLinearRegression.Wrapper(200, new PriorSGDStrategy(new L1()));
    cl.update(new double[]{1.0e-2, 0.01});

    for (int i = 0; i < 10000; i++) {
      AdaptiveLinearRegression.TrainingExample r = getExample(i, gen);
      cl.train(r);
      if (i % 1000 == 0) {
        System.out.printf("%10d %10.3f\n", i, cl.getLearner().mse());
      }
    }
    assertEquals(0.0, cl.getLearner().mse(), 0.1);

    // we expect AdaptiveLinearRegression to learn the similar parameters to above
    AdaptiveLinearRegression x = new AdaptiveLinearRegression(200, new PriorSGDStrategy(new L1()));
    x.setInterval(1000);

    for (int i = 0; i < 20000; i++) {
      AdaptiveLinearRegression.TrainingExample r = getExample(i, gen);
      x.train(r.getKey(), r.getActual(), r.getInstance());
      if (i % 1000 == 0 && x.getBest() != null) {
        System.out.printf("%10d %10.4f %10.8f %.8f\n",
            i, x.mse(),
            Math.log10(x.getBest().getMappedParams()[0]), Math.log10(x.getBest().getMappedParams()[1]));
      }
    }
    assertEquals(0.0, x.mse(), 0.1);
  }

  private static AdaptiveLinearRegression.TrainingExample getExample(int i, Random gen) {
    Vector data = new DenseVector(200);
    double target = 1.0;
    data.assign(0.0);
    data.set(0,1.0);
    for(int j=1; j < data.size(); j++){
      double epsilon = gen.nextGaussian();
      data.set(j,epsilon);
    }

    return new AdaptiveLinearRegression.TrainingExample(i, null, target, data);
  }

  @Test
  public void stepSize() {
    assertEquals(500, AdaptiveLinearRegression.stepSize(15000, 2));
    assertEquals(2000, AdaptiveLinearRegression.stepSize(15000, 2.6));
    assertEquals(5000, AdaptiveLinearRegression.stepSize(24000, 2.6));
    assertEquals(10000, AdaptiveLinearRegression.stepSize(15000, 3));
  }

  @Test
  public void constantStep() {
    AdaptiveLinearRegression lr = new AdaptiveLinearRegression(1000, new PriorSGDStrategy(new L1()));
    lr.setInterval(5000);
    assertEquals(20000, lr.nextStep(15000));
    assertEquals(20000, lr.nextStep(15001));
    assertEquals(20000, lr.nextStep(16500));
    assertEquals(20000, lr.nextStep(19999));
  }


  @Test
  public void growingStep() {
    AdaptiveLinearRegression lr = new AdaptiveLinearRegression(1000, new PriorSGDStrategy(new L1()));
    lr.setInterval(2000, 10000);

    // start with minimum step size
    for (int i = 2000; i < 20000; i+=2000) {
      assertEquals(i + 2000, lr.nextStep(i));
    }

    // then level up a bit
    for (int i = 20000; i < 50000; i += 5000) {
      assertEquals(i + 5000, lr.nextStep(i));
    }

    // and more, but we top out with this step size
    for (int i = 50000; i < 500000; i += 10000) {
      assertEquals(i + 10000, lr.nextStep(i));
    }
  }
}
