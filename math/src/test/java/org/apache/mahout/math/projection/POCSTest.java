package org.apache.mahout.math.projection;

import static org.junit.Assert.*;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.junit.Test;

public class POCSTest {

	@Test
	public void testSingleProjection() {
		Vector v = new DenseVector(new double[] {1, -1});
		Vector g = new DenseVector(new double[] {0, -1});
		double h = 0;
		Projection p0 = new LinearInequalityProjection(g, h);

		POCS p = new POCS(new Projection[]{p0}, 0.0);
		Vector v1 = p.apply(v);
		assertEquals(v1, new DenseVector(new double[] {1, 0}));
	}

	@Test
	public void testMultipleProjectionsSimple() {
		Vector v = new DenseVector(new double[] {-1, -0});
		POCS pocs = new POCS(new Projection[]
				{new LinearInequalityProjection(new DenseVector(new double[] {-1, 1}), 0.0),
				new LinearInequalityProjection(new DenseVector(new double[] {-1, -1}), 0.0)},
				1e-5);
		Vector v1 = pocs.apply(v);
		assertEquals(v1, new DenseVector(new double[] {0, 0}));
	}

	@Test
	public void testMultipleProjections() {
		double rEpsilon = 1e-10;
		Vector v = new DenseVector(new double[] {-1, -0});
		POCS pocs = new POCS(new Projection[]
				{new LinearInequalityProjection(new DenseVector(new double[] {-0.5, 1}), 0.0),
				new LinearInequalityProjection(new DenseVector(new double[] {-0.5, -1}), 0.0)},
				rEpsilon);
		Vector v1 = pocs.apply(v);
		assertTrue(v1.minus(new DenseVector(new double[] {0, 0})).norm(2.0) <= rEpsilon);
	}
}
