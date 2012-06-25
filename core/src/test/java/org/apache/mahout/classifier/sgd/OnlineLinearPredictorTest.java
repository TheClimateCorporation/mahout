/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.classifier.sgd;

import org.apache.mahout.math.Vector;
import org.apache.mahout.regression.sgd.OnlineLinearPredictor;
import org.junit.Test;

import java.io.IOException;

public final class OnlineLinearPredictorTest extends OnlineLinearPredictorBaseTest {

    @Test
    public void testTrain() throws Exception {
        Vector target = readStandardData();


        // lambda here needs to be relatively small to avoid swamping the actual signal, but can be
        // larger than usual because the data are dense.  The learning rate doesn't matter too much
        // for this example, but should generally be < 1
        // --passes 1 --rate 50 --lambda 0.001 --input sgd-y.csv --features 21 --output model --noBias
        //   --target y --categories 2 --predictors  V2 V3 V4 V5 V6 V7 --types n
        OnlineLinearPredictor regression = new OnlineLinearPredictor(4, new L1())
                .lambda(1 * 1.0e-10)
                .learningRate(0.2);

        train(getInput(), target, regression);
        test(getInput(), target, regression, 0.05, 0.3);
    }

}