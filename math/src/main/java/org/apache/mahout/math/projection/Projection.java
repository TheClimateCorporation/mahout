package org.apache.mahout.math.projection;

import org.apache.mahout.math.Vector;

public interface Projection {
	public boolean contains(Vector v);
	public Vector project(Vector v);
	public Projection clone();
}
