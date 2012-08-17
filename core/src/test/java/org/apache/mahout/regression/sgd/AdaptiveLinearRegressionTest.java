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
import org.apache.mahout.classifier.sgd.ProjectionSGDStrategy;
import org.apache.mahout.classifier.sgd.SGDStrategy;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.projection.LinearInequalityProjection;
import org.apache.mahout.math.projection.POCS;
import org.apache.mahout.math.projection.Projection;
import org.junit.Test;

import org.apache.mahout.classifier.sgd.SGDTestCase;

public final class AdaptiveLinearRegressionTest extends MahoutTestCase {

	private static AdaptiveLinearRegression.TrainingExample getGaussianExample(int i, Random gen, int cardinality) {
		Vector data = new DenseVector(cardinality);
		double target = 1.0;
		data.assign(0.0);
		data.set(0,1.0);
		for(int j=1; j < data.size(); j++){
			double epsilon = gen.nextGaussian();
			data.set(j,epsilon);
		}
		return new AdaptiveLinearRegression.TrainingExample(i, null, target, data);
	}

	private static AdaptiveLinearRegression.TrainingExample getRandomExample(int i, Random gen, Vector targets, Matrix instances ) {
		int j = gen.nextInt(targets.size());
		return new AdaptiveLinearRegression.TrainingExample(i, null, targets.getQuick(j), instances.viewRow(j));
	}

	@Test
	public void testTrainPrior() {
		Random gen = RandomUtils.getRandom();
		int cardinality = 200;
		AdaptiveLinearRegression.Wrapper cl = new AdaptiveLinearRegression.Wrapper(cardinality, new PriorSGDStrategy(new L1()));
		cl.update(new double[]{1.0e-2, 0.01});

		for (int i = 0; i < 10000; i++) {
			AdaptiveLinearRegression.TrainingExample r = getGaussianExample(i, gen, cardinality);
			cl.train(r);
			if (i % 1000 == 0) {
				System.out.printf("%10d %10.3f\n", i, cl.getLearner().mse());
			}
		}
		assertEquals(0.0, cl.getLearner().mse(), 0.1);

		// we expect AdaptiveLinearRegression to learn the similar parameters to above
		AdaptiveLinearRegression x = new AdaptiveLinearRegression(cardinality, new PriorSGDStrategy(new L1()));
		x.setInterval(1000);

		for (int i = 0; i < 20000; i++) {
			AdaptiveLinearRegression.TrainingExample r = getGaussianExample(i, gen, cardinality);
			x.train(r.getKey(), r.getActual(), r.getInstance());
			if (i % 1000 == 0 && x.getBest() != null) {
				System.out.printf("%10d %10.4f %10.8f %.8f\n",
						i, x.mse(),
						Math.log10(x.getBest().getMappedParams()[0]), Math.log10(x.getBest().getMappedParams()[1]));
			}
		}
		assertEquals(0.0, x.mse(), 0.1);
		x.close();
	}

	private void testTrainProjection(ProjectionSGDStrategy strategy, Vector targets, Matrix instances, Vector targetBeta, int numReps) {
		Random gen = RandomUtils.getRandom();
		int cardinality = instances.viewRow(0).size();

		AdaptiveLinearRegression x = new AdaptiveLinearRegression(cardinality, strategy);
		x.setInterval(1000);

		for (int i = 0; i < numReps; i++) {
			AdaptiveLinearRegression.TrainingExample r = getRandomExample(i, gen, targets, instances);
			x.train(r.getKey(), r.getActual(), r.getInstance());
			if (i % 1000 == 0 && x.getBest() != null) {
				System.out.printf("%10d %10.4f %10.8f %.8f\n",
						i, x.mse(),
						Math.log10(x.getBest().getMappedParams()[0]), Math.log10(x.getBest().getMappedParams()[1]));
			}
		}
		
		Vector beta = x.getBest().getPayload().getLearner().getModels().get(0).getBeta();
		x.close();
		
		System.out.printf("beta   : %s\n", beta);
		System.out.printf("target : %s\n", targetBeta);
		
		for(int i = 0; i < beta.size(); i++) {
			assertEquals(beta.getQuick(i), targetBeta.getQuick(i), 0.05);
		}
	}

	@Test
	public void testTrainWithProjectionSGDStrategy() throws Exception {
		Matrix instances = SGDTestCase.readCsv("sgd-regression.csv");
		Vector targets = new DenseVector(instances.numRows());
		targets.assign(1);
		Vector targetBeta = new DenseVector(new double[]{0.0, 1.0, 0.0, 0.0});
		POCS pocs = new POCS(new Projection[] { new LinearInequalityProjection(new DenseVector(new double[]{1, 0, 0, 0}), 0.0), 
				new LinearInequalityProjection(new DenseVector(new double[]{0, 1, 0, 0}), 1.0)});
		testTrainProjection(new ProjectionSGDStrategy(pocs), targets, instances, targetBeta, 10000);
	}

	private void testTrainWithProjectionSGDStrategyEllipsoidBase(POCS pocs, Vector targetBeta) {
		Vector targets = new DenseVector(new double[] {1.5, 1});
		Matrix instances = new DenseMatrix(new double[][] {{1.5, 0}, {0, 1}});
		testTrainProjection(new ProjectionSGDStrategy(pocs), targets, instances, targetBeta, 10000);
	}

	@Test
	public void testTrainWithProjectionSGDStrategyEllipsoid() throws Exception {
		Vector targetBeta = new DenseVector(new double[]{0.0, 0.0});
		POCS pocs = new POCS(new Projection[] { new LinearInequalityProjection(new DenseVector(new double[]{1, 1}), 0.0),
				new LinearInequalityProjection(new DenseVector(new double[]{1, -1}), 0.0)});
		testTrainWithProjectionSGDStrategyEllipsoidBase(pocs, targetBeta);
	}

	@Test
	public void testTrainWithProjectionSGDStrategyEllipsoid2() throws Exception {
		Vector targetBeta = new DenseVector(new double[]{0.0, 1.0});
		POCS pocs = new POCS(new Projection[] { new LinearInequalityProjection(new DenseVector(new double[]{1, 0}), 0.0)});
		testTrainWithProjectionSGDStrategyEllipsoidBase(pocs, targetBeta);
	}

	@Test
	public void testTrainWithProjectionSGDStrategyRandom() throws Exception {
		Matrix instances = new DenseMatrix(new double[][] {
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
		Vector targets =  new DenseVector(new double[] {
				0.0561233, 0.87001016, 0.56999933, 0.19983942, 0.50472047,
				0.48492511, 0.35678996, 0.34607792, 0.5384788, 0.62348945,
				0.61245246, 0.4581468, 0.02797498, 0.22960503, 0.17721126,
				0.58446087, 0.86100886, 0.79843894, 0.79709756, 0.81643737,
				0.25529404, 0.84174483, 0.67311353, 0.08323414, 0.01669063});
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
		testTrainProjection(new ProjectionSGDStrategy(pocs), targets, instances, targetBeta, 10000);
	}

	@Test
	public void stepSize() {
		assertEquals(500, AdaptiveLinearRegression.stepSize(15000, 2));
		assertEquals(2000, AdaptiveLinearRegression.stepSize(15000, 2.6));
		assertEquals(5000, AdaptiveLinearRegression.stepSize(24000, 2.6));
		assertEquals(10000, AdaptiveLinearRegression.stepSize(15000, 3));
	}

	public void constantStep(int cardinality, SGDStrategy strategy) {
		AdaptiveLinearRegression lr = new AdaptiveLinearRegression(cardinality, strategy);
		lr.setInterval(5000);
		assertEquals(20000, lr.nextStep(15000));
		assertEquals(20000, lr.nextStep(15001));
		assertEquals(20000, lr.nextStep(16500));
		assertEquals(20000, lr.nextStep(19999));
		lr.close();
	}

	@Test
	public void constantStepPrior() {
		constantStep(1000, new PriorSGDStrategy(new L1()));
	}

	@Test
	public void constantStepPOCS() {
		POCS pocs = new POCS(new Projection[] { new LinearInequalityProjection(new DenseVector(new double[]{1, 0, 0, 0}), 0.0), 
				new LinearInequalityProjection(new DenseVector(new double[]{0, 1, 0, 0}), 1.0)});
		constantStep(4, new ProjectionSGDStrategy(pocs));
	}

	private void growingStep(int cardinality, SGDStrategy strategy) {
		AdaptiveLinearRegression lr = new AdaptiveLinearRegression(cardinality, new PriorSGDStrategy(new L1()));
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
		lr.close();
	}

	@Test
	public void growingStepPrior() {
		growingStep(1000, new PriorSGDStrategy(new L1()));
	}

	@Test
	public void growingStepPOCS() {
		POCS pocs = new POCS(new Projection[] { new LinearInequalityProjection(new DenseVector(new double[]{1, 0, 0, 0}), 0.0), 
				new LinearInequalityProjection(new DenseVector(new double[]{0, 1, 0, 0}), 1.0)});
		growingStep(4, new ProjectionSGDStrategy(pocs));
	}
}
