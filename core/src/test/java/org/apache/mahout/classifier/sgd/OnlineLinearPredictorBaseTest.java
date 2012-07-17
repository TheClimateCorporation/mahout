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

import java.io.IOException;
import java.util.Random;

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.regression.sgd.OnlineLinearPredictor;

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

	Vector readEllipsoidData() {
		input = new DenseMatrix(new double[][] {{1.5, 0}, {0, 1}});
		return new DenseVector(new double[] {1.5, 1});
	}

	Vector readRandomData() {
		input = new DenseMatrix(new double[][] {
				{ 0.13436424,  0.84743374,  0.76377462,  0.25506903},
				{ 0.49543509,  0.44949106,  0.65159297,  0.78872335},
				{ 0.09385959,  0.02834748,  0.8357651,   0.43276707},
				{ 0.76228008,  0.00210605,  0.44538719,  0.72154003},
				{ 0.22876222,  0.9452707,   0.90142746,  0.03058998},
				{ 0.02544586,  0.54141247,  0.93914916,  0.38120424},
				{ 0.2165994,   0.42211658,  0.02904079,  0.22169167},
				{ 0.43788759,  0.49581224,  0.23308445,  0.23086654},
				{ 0.21878104,  0.45960347,  0.28978161,  0.02148971},
				{ 0.83757798,  0.55645432,  0.64229436,  0.18590627},
				{ 0.99254341,  0.85994653,  0.12088996,  0.33269519},
				{ 0.72148441,  0.71119177,  0.93644059,  0.422107  },
				{ 0.83003569,  0.67030557,  0.30336851,  0.58758061},
				{ 0.882479,    0.84619742,  0.50528382,  0.58900226},
				{ 0.03452583,  0.24273997,  0.79740425,  0.414314  },
				{ 0.1730074,   0.54879876,  0.70304076,  0.67448583},
				{ 0.37470302,  0.43896163,  0.50842649,  0.77844262},
				{ 0.52093842,  0.39325509,  0.48969352,  0.02957496},
				{ 0.04348729,  0.70338209,  0.98318772,  0.59318373},
				{ 0.39359969,  0.1703492,   0.50223856,  0.98207664},
				{ 0.77052314,  0.53961745,  0.86028978,  0.23217613},
				{ 0.51377166,  0.95246739,  0.57779481,  0.45913173},
				{ 0.26927948,  0.54799631,  0.95711628,  0.00570913},
				{ 0.78365523,  0.82048591,  0.88617958,  0.74050341},
				{ 0.8091399,   0.51867828,  0.56135786,  0.42609068}});
		return new DenseVector(new double[] {
				0.0561233, 0.87001016, 0.56999933, 0.19983942, 0.50472047,
				0.48492511, 0.35678996, 0.34607792, 0.5384788, 0.62348945,
				0.61245246, 0.4581468, 0.02797498, 0.22960503, 0.17721126,
				0.58446087, 0.86100886, 0.79843894, 0.79709756, 0.81643737,
				0.25529404, 0.84174483, 0.67311353, 0.08323414, 0.01669063});
	}
	
	static void train(Matrix input, Vector target, OnlineLinearPredictor regression) {
		train(input, target, regression, 1);
	}

	static void train(Matrix input, Vector target, OnlineLinearPredictor regression, int numPasses) {
		RandomUtils.useTestSeed();
		Random gen = RandomUtils.getRandom();

		// train on samples in random order
		for(int i = 0; i < numPasses; i++) {
			for (int row : permute(gen, input.numRows())) {
				regression.train(target.get(row), input.viewRow(row));
			}
		}
		regression.close();
	}

	static void testAccuracy(Matrix input, Vector target, OnlineLinearPredictor regression,
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
	}

	static void testBetas(Matrix input, Vector targetBeta, OnlineLinearPredictor regression) {
		Vector tmp = regression.predict(input);
		Vector beta = regression.getBeta();
		System.out.println("beta: " + beta);
		for(int i = 0; i < beta.size(); i++) {
			assertEquals(beta.get(i), targetBeta.get(i), 2.0e-2);
		}
		// convenience methods should give the same results
		Vector v = regression.predict(input);
		assertEquals(0, v.minus(tmp).norm(1), 1.0e-5);
	}

}
