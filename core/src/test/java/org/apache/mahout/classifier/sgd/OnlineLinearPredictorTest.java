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

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.projection.LinearInequalityProjection;
import org.apache.mahout.math.projection.POCS;
import org.apache.mahout.math.projection.Projection;
import org.apache.mahout.regression.sgd.OnlineLinearPredictor;
import org.junit.Test;

public final class OnlineLinearPredictorTest extends OnlineLinearPredictorBaseTest {

	@Test
	public void testTrainWithPriorSGDStrategy() throws Exception {
		Vector target = readStandardData();
		// expected solution is <0.5,0.5,0.0,0.0>
		Vector targetBeta = new DenseVector(new double[]{0.5, 0.5, 0.0, 0.0});
		// lambda here needs to be relatively small to avoid swamping the actual signal, but can be
		// larger than usual because the data are dense.  The learning rate doesn't matter too much
		// for this example, but should generally be < 1
		// --passes 1 --rate 50 --lambda 0.001 --input sgd-y.csv --features 21 --output model --noBias
		//   --target y --categories 2 --predictors  V2 V3 V4 V5 V6 V7 --types n
		OnlineLinearPredictor regression = new OnlineLinearPredictor(4, new PriorSGDStrategy(new L1())).learningRate(0.2);
		((PriorSGDStrategy) (regression.getStrategy())).setLambda(1 * 1.0e-3);
		train(getInput(), target, regression, 10);
		testAccuracy(getInput(), target, regression, 0.05, 0.3);
		testBetas(getInput(), targetBeta, regression);
	}

	@Test
	public void testTrainWithProjectionSGDStrategy() throws Exception {
		Vector target = readStandardData();
		Vector targetBeta = new DenseVector(new double[]{0.0, 1.0, 0.0, 0.0});
		POCS pocs = new POCS(new Projection[] { new LinearInequalityProjection(new DenseVector(new double[]{1, 0, 0, 0}), 0.0), 
				new LinearInequalityProjection(new DenseVector(new double[]{0, 1, 0, 0}), 1.0)});
		OnlineLinearPredictor regression = new OnlineLinearPredictor(4, new ProjectionSGDStrategy(pocs)).learningRate(0.4);
		train(getInput(), target, regression, 1000);
		testBetas(getInput(), targetBeta, regression);
	}

	@Test
	public void testTrainWithProjectionSGDStrategyEllipsoid() throws Exception {
		Vector target = readEllipsoidData();
		Vector targetBeta = new DenseVector(new double[]{0.0, 0.0});
		POCS pocs = new POCS(new Projection[] { new LinearInequalityProjection(new DenseVector(new double[]{1, 1}), 0.0),
				new LinearInequalityProjection(new DenseVector(new double[]{1, -1}), 0.0)});
		OnlineLinearPredictor regression = new OnlineLinearPredictor(2, new ProjectionSGDStrategy(pocs)).learningRate(0.2);
		regression.setBeta(new DenseVector(new double[] {-2.0, -1.0}));
		train(getInput(), target, regression, 1000);
		testBetas(getInput(), targetBeta, regression);
	}

	@Test
	public void testTrainWithProjectionSGDStrategyEllipsoid2() throws Exception {
		Vector target = readEllipsoidData();
		Vector targetBeta = new DenseVector(new double[]{0.0, 1.0});
		POCS pocs = new POCS(new Projection[] { new LinearInequalityProjection(new DenseVector(new double[]{1, 0}), 0.0)});
		OnlineLinearPredictor regression = new OnlineLinearPredictor(2, new ProjectionSGDStrategy(pocs)).learningRate(0.5);
		regression.setBeta(new DenseVector(new double[] {-2.0, -1.0}));
		train(getInput(), target, regression, 1000);
		testBetas(getInput(), targetBeta, regression);
	}

	@Test
	public void testTrainWithProjectionSGDStrategyRandom() throws Exception {
		Vector target = readRandomData();
		Vector targetBeta = new DenseVector(new double[] {-0.31265007, 0.43969544, 0.09641295, 0.5791211});
		POCS pocs = new POCS(new Projection[] {
				new LinearInequalityProjection(new DenseVector(new double[] { 0.01455997,  0.75558678,  0.24955923,  0.10948863}), 0.57584596),
				new LinearInequalityProjection(new DenseVector(new double[] { 0.62480208,  0.34442286,  0.06951538,  0.15962552}), 0.32124581),
				new LinearInequalityProjection(new DenseVector(new double[] { 0.5273804,   0.16814495,  0.27291444,  0.71158993}), 0.63094786),
				new LinearInequalityProjection(new DenseVector(new double[] { 0.45470163,  0.32200177,  0.47377101,  0.02363458}), 0.05878512),
				new LinearInequalityProjection(new DenseVector(new double[] { 0.3865571,   0.42091868,  0.1880393,   0.10876169}), 0.29860595),
				new LinearInequalityProjection(new DenseVector(new double[] { 0.8998185,   0.51011598,  0.20909099,  0.60564864}), 0.96790331),
				new LinearInequalityProjection(new DenseVector(new double[] { 0.81703967,  0.02081811,  0.01786452,  0.14646174}), 0.87553424),
				new LinearInequalityProjection(new DenseVector(new double[] { 0.71883547,  0.16022759,  0.70460563,  0.6781758 }), 0.30638662),
				new LinearInequalityProjection(new DenseVector(new double[] { 0.54470216,  0.22059975,  0.97559452,  0.79781086}), 0.85851441),
				new LinearInequalityProjection(new DenseVector(new double[] { 0.51659952,  0.22319578,  0.64850642,  0.39489801}), 0.31036363)});
		OnlineLinearPredictor regression = new OnlineLinearPredictor(4, new ProjectionSGDStrategy(pocs)).learningRate(0.9).decayExponent(0.01);
		regression.setBeta(new DenseVector(new double[] {-2.0, -1.0, 5, 3}));
		train(getInput(), target, regression, 2000);
		testBetas(getInput(), targetBeta, regression);		
		
		
					
				
	}
}