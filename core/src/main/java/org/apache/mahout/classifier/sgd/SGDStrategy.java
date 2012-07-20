package org.apache.mahout.classifier.sgd;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.Vector;

public interface SGDStrategy extends Writable {
	public SGDStrategy copy();
	public void regularize(SGDLearner learner, Vector updateSteps, Vector instance, Vector beta);
	public void applyGradient(SGDLearner learner, Vector instance, Vector beta, double gradientBase, Vector updateSteps, Vector updateCounts);
	public void update(double[] params);
}
