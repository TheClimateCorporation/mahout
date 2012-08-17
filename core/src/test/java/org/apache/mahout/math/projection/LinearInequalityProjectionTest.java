package org.apache.mahout.math.projection;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.junit.Test;

public class LinearInequalityProjectionTest {

	@Test
	public void testSimpleProjection() {
		Vector v = new DenseVector(new double[] {1, -1});
		Vector g = new DenseVector(new double[] {0, -1});
		double h = 0;
		Projection p = new LinearInequalityProjection(g, h);
		assertFalse(p.contains(v));

		Vector v1 = p.project(v);
		assertTrue(p.contains(v1));

		assertEquals(v1, new DenseVector(new double[] {1, 0}));
	}

	@Test
	public void testIdentityProjection() {
		Vector v = new DenseVector(new double[] {1, 1});
		Vector g = new DenseVector(new double[] {0, -1});
		double h = 0;
		Projection p = new LinearInequalityProjection(g, h);
		assertTrue(p.contains(v));

		Vector v1 = p.project(v);
		assertTrue(p.contains(v1));

		assertEquals(v1, v);
	}

	@Test
	public void testProjection() {
		Vector v = new DenseVector(new double[] {0, 0});
		Vector g = new DenseVector(new double[] {-1, -1});
		double h = -1;
		Projection p = new LinearInequalityProjection(g, h);
		assertFalse(p.contains(v));

		Vector v1 = p.project(v);
		assertTrue(p.contains(v1));

		assertEquals(v1, new DenseVector(new double[] {0.5, 0.5}));
	}

}
