package org.apache.mahout.math.projection;

import org.apache.mahout.math.Vector;

import org.apache.hadoop.io.Writable;

public interface Projection extends Writable {
	public boolean contains(Vector v);
	public Vector project(Vector v);
	public Projection clone();
}
