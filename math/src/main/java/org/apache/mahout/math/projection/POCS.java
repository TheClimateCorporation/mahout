package org.apache.mahout.math.projection;
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
public class POCS {
	private Projection[] projections;
	private double rEpsilon;

	public final static double DEFAULT_REPS = 1e-6;

	@SuppressWarnings("unused")
	private POCS() {}

	public POCS(Projection[] projections) {
		this(projections, DEFAULT_REPS);
	}

	public POCS(Projection[] projections, double reps) {
		this.projections = new Projection[projections.length];
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

	public Vector apply(Vector v) {
		double err = Double.POSITIVE_INFINITY;
		while (err > rEpsilon) {
			Vector nv = applyProjections(v);
			err = nv.minus(v).norm(2.0);
			v = nv;
		}
		return v;
	}

	public POCS clone() {
		return new POCS(projections, rEpsilon);
	}
}
