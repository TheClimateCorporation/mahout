package org.apache.mahout.math.projection;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.mahout.classifier.sgd.PolymorphicWritable;
import org.apache.mahout.math.Vector;

/**
 * Projection Onto Convex Sets (POCS)
 *
 * Implementation of iterative POCS, useful for constraining a vector v
 * to a convex set specified by an ordered list of projections.
 * This algorithm successively applies the entire list of projections until
 * the distance between subsequently projected vectors is below rEpsilon.
 * @author ebrevdo
 *
 */
public class POCS implements Projection {
	
	private final static int WRITABLE_VERSION = 1;
	private int projectionsLength; // exists for serialization
	private Projection[] projections;
	private double rEpsilon;

	private final static double DEFAULT_REPSILON = 1e-6;

	// exists for serialization
	public POCS() {
		projectionsLength = 0;
		rEpsilon = DEFAULT_REPSILON;
	}

	public POCS(Projection[] projections) {
		this(projections, DEFAULT_REPSILON);
	}

	public POCS(Projection[] projections, double reps) {
		this.projections = new Projection[projections.length];
		this.projectionsLength = projections.length;
		this.rEpsilon = reps;
		for (int pi=0; pi<projections.length; pi++) {
			this.projections[pi] = projections[pi].clone();
		}
	}

	private Vector applyProjections(Vector v) {
		for (Projection p : projections) {
			v = p.project(v);
		}
		return v;
	}

	public POCS clone() {
		return new POCS(projections, rEpsilon);
	}

	@Override
	public boolean contains(Vector v) {
		Vector v1 = project(v);
		return v1.minus(v).norm(2.0) <= rEpsilon;
	}

	@Override
	public Vector project(Vector v) {
		double err = Double.POSITIVE_INFINITY;
		while (err > rEpsilon) {
			Vector nv = applyProjections(v);
			err = nv.minus(v).norm(2.0);
			v = nv;
		}
		return v;
	}
	
	@Override
	public void write(DataOutput out) throws IOException {
		out.writeInt(WRITABLE_VERSION);
		out.writeInt(projectionsLength);
		for(int pi = 0; pi < projectionsLength; pi++) {
			PolymorphicWritable.write(out, projections[pi]);
		}
		out.writeDouble(rEpsilon);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		int version = in.readInt();
		if (version == WRITABLE_VERSION) {
			projectionsLength = in.readInt();
			for(int pi = 0; pi < projectionsLength; pi++) {
				PolymorphicWritable.read(in, Projection.class);
			}
			rEpsilon = in.readDouble();
		} else {
			throw new IOException("Incorrect object version, wanted " + WRITABLE_VERSION + " got " + version);
		}
	}
}
