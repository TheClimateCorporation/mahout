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

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.regression.sgd.OnlineLinearPredictor;

import java.io.IOException;
import java.util.Random;

public abstract class OnlineLinearPredictorBaseTest extends SGDTestCase {

    Vector readStandardData() throws IOException {
        // 500 test samples.  First column is constant value of 1.0.  Second is normally distributed from
        // N(0, 0.1). The remaining columns are normally distributed from N(0,1).
        input = readCsv("sgd-regression.csv");

        // regenerate the target variable
        Vector target = new DenseVector(input.numRows());
        target.assign(1);
        return target;
    }

    static void train(Matrix input, Vector target, OnlineLinearPredictor regression) {
        RandomUtils.useTestSeed();
        Random gen = RandomUtils.getRandom();

        // train on samples in random order (but only one pass)
        for (int row : permute(gen, input.numRows())) {
            regression.train((int) target.get(row), input.viewRow(row));
        }
        regression.close();
    }

    static void test(Matrix input, Vector target, OnlineLinearPredictor regression,
                     double expected_mean_error, double expected_absolute_error) {
        // now test the accuracy
        Vector tmp = regression.predict(input);
        // mean(abs(tmp - target))
        double meanAbsoluteError = tmp.minus(target).aggregate(Functions.PLUS, Functions.ABS) / input.numRows();

        // max(abs(tmp - target)
        double maxAbsoluteError = tmp.minus(target).aggregate(Functions.MAX, Functions.ABS);

        System.out.printf("mAE = %.4f, maxAE = %.4f\n", meanAbsoluteError, maxAbsoluteError);
        assertEquals(0, meanAbsoluteError, expected_mean_error);
        assertEquals(0, maxAbsoluteError, expected_absolute_error);

        // expected solution is <0.5,0.5,0.0,0.0>
        Vector beta = regression.getBeta();
        assertEquals(beta.get(0), 0.5, 1.0e-2);
        assertEquals(beta.get(1),0.5,1.0e-2);
        assertEquals(beta.get(2),0.0,1.0e-1);
        assertEquals(beta.get(3),0.0,1.0e-1);

        // convenience methods should give the same results
        Vector v = regression.predict(input);
        assertEquals(0, v.minus(tmp).norm(1), 1.0e-5);
    }

}
