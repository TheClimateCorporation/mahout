package org.apache.mahout.math.projection;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.mahout.classifier.sgd.PolymorphicWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class LinearInequalityProjection implements Projection {

	private final static int WRITABLE_VERSION = 1;
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
		// same as a call to contains(v), but we cache gTv for performance.
		if(gTv <= h)
			return v.clone();
		return v.plus(g.times((h - gTv) / gTg));
	}

	@Override
	public Projection clone() {
		return new LinearInequalityProjection(g, h);
	}
	
	@Override
	public void write(DataOutput out) throws IOException {
		out.writeInt(WRITABLE_VERSION);
		PolymorphicWritable.write(out, new VectorWritable(g));
		out.writeDouble(h);
		out.writeDouble(gTg);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		int version = in.readInt();
		if (version == WRITABLE_VERSION) {
			g = PolymorphicWritable.read(in, VectorWritable.class).get();
			h = in.readDouble();
			gTg = in.readDouble();
		} else {
			throw new IOException("Incorrect object version, wanted " + WRITABLE_VERSION + " got " + version);
		}
	}
}
