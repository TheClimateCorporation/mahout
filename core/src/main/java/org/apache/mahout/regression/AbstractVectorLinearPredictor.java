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

package org.apache.mahout.regression;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

/**
 * Defines the interface for regression solutions that take input as a vector.  This is implemented
 * as an abstract class so that it can implement a number of handy convenience methods
 * related to regression of vectors.
 */
public abstract class AbstractVectorLinearPredictor {
  // ------ These are all that are necessary to define a vector classifier.

  /**
   * Predict value of a vector returning a predicted value.
   *
   * @param instance A feature vector to be classified.
   * @return A double of the predicted value.
   */
  public abstract double predict(Vector instance);

  // ------- From here on, we have convenience methods that provide an easier API to use.

  /**
   * @param data The matrix whose rows are vectors to predict
   * @return A vector of predictions, one value per row of the input matrix.
   */
  public Vector predict(Matrix data) {
    Vector r = new DenseVector(data.numRows());
    for (int row = 0; row < data.numRows(); row++) {
      r.set(row, predict(data.viewRow(row)));
    }
    return r;
  }

}