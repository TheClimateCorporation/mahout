package org.apache.mahout.classifier.sgd;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Iterator;

import org.apache.mahout.math.Vector;
import org.apache.mahout.math.projection.Projection;

public class ProjectionSGDStrategy extends AbstractSGDStrategy implements SGDStrategy {

	private static final int WRITABLE_VERSION = 1;
	private Projection projection;
	
	public ProjectionSGDStrategy() {}
	
	public ProjectionSGDStrategy(Projection projection) {
		this.projection = projection.clone();
	}
	
	public ProjectionSGDStrategy(ProjectionSGDStrategy other) {
		this.projection = other.projection.clone();
	}
	
	@Override
	public SGDStrategy copy() {
		return new ProjectionSGDStrategy(this);
	}

	@Override
	public void regularize(SGDLearner learner, Vector updateSteps,
			Vector instance, Vector beta) {
		Vector betaNew = projection.project(beta);
		Vector diff = beta.minus(betaNew);
		
		Iterator<Vector.Element> iterator = diff.iterateNonZero();
		while(iterator.hasNext()) {
			int index = iterator.next().index();
			updateSteps.setQuick(index, learner.getStep());
			beta.setQuick(index, betaNew.getQuick(index));
		}
	}

	@Override
	public void update(double[] params) {
		// no-op
	}
	
	@Override
	public void write(DataOutput out) throws IOException {
		out.writeInt(WRITABLE_VERSION);
		PolymorphicWritable.write(out, projection);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		int version = in.readInt();
		if (version == WRITABLE_VERSION) {
			projection = PolymorphicWritable.read(in, Projection.class);
		} else {
			throw new IOException("Incorrect object version, wanted " + WRITABLE_VERSION + " got " + version);
		}
	}
}
