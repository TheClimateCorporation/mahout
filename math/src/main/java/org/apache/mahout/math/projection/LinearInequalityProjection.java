package org.apache.mahout.math.projection;

import org.apache.mahout.math.Vector;

public class LinearInequalityProjection implements Projection {

	private Vector g;
	private double h;
	private double gTg;

	@SuppressWarnings("unused")
	private LinearInequalityProjection() {}

	public LinearInequalityProjection(Vector g, double h) {
		if(!(g.iterateNonZero().hasNext()))
			throw new IllegalArgumentException("Vector g must contain non-zero elements.");
		this.g = g.clone();
		this.h = h;
		this.gTg = g.dot(g);
	}

	@Override
	public boolean contains(Vector v) {
		return g.dot(v) <= h;
	}

	@Override
	public Vector project(Vector v) {
		double gTv = g.dot(v);
		// same as a call to isInside(v), but we cache gTv for performance.
		if(gTv <= h)
			return v.clone();
		return v.plus(g.times((h - gTv) / gTg));
	}

	@Override
	public Projection clone() {
		return new LinearInequalityProjection(g, h);
	}
}
