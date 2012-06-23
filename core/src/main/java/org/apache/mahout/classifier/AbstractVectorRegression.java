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

package org.apache.mahout.classifier;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

/**
 * Defines the interface for regression solutions that take input as a vector.  This is implemented
 * as an abstract class so that it can implement a number of handy convenience methods
 * related to regression of vectors.
 */
public abstract class AbstractVectorRegression {
    // ------ These are all that are necessary to define a vector classifier.

    /**
     * TODO: update
     * Classify a vector returning a vector of numCategories-1 scores.  It is assumed that
     * the score for the missing category is one minus the sum of the scores that are returned.
     *
     * Note that the missing score is the 0-th score.
     * @param instance  A feature vector to be classified.
     * @return  A vector of probabilities in 1 of n-1 encoding.
     */
    public abstract double predict(Vector instance);

    // ------- From here on, we have convenience methods that provide an easier API to use.

    /**
     * Returns n-1 probabilities, one for each category but the last, for each row of a matrix. The
     * probability of the missing 0-th category is 1 - rowSum(this result).
     *
     * @param data The matrix whose rows are vectors to classify
     * @return A matrix of scores, one row per row of the input matrix, one column for each but the
     *         last category.
     */
    public Vector predict(Matrix data) {
        Vector r = new DenseVector(data.numRows());
        for (int row = 0; row < data.numRows(); row++) {
            r.set(row, predict(data.viewRow(row)));
        }
        return r;
    }


}