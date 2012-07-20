package org.apache.mahout.classifier.sgd;

import java.util.Iterator;

import org.apache.mahout.math.Vector;

public class AbstractSGDStrategy {

	public AbstractSGDStrategy() {
		super();
	}

	public void applyGradient(SGDLearner learner, Vector instance, Vector beta,
			double gradientBase, Vector updateSteps, Vector updateCounts) {
		// apply the gradientBase to beta
		Iterator<Vector.Element> nonZeros = instance.iterateNonZero();
		while (nonZeros.hasNext()) {
			Vector.Element updateLocation = nonZeros.next();
			int j = updateLocation.index();
			double gradient = gradientBase * learner.currentLearningRate() * learner.perTermLearningRate(j) * instance.get(j);
			double newValue = beta.getQuick(j) + gradient;
			beta.setQuick(j, newValue);
		}

		// remember that these elements got updated
		Iterator<Vector.Element> i = instance.iterateNonZero();
		while (i.hasNext()) {
			Vector.Element element = i.next();
			int j = element.index();
			updateSteps.setQuick(j, learner.getStep());
			updateCounts.setQuick(j, updateCounts.getQuick(j) + 1);
		}
		learner.nextStep();
	}

}