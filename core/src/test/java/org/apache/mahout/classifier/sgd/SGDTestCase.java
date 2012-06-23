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

import com.google.common.base.CharMatcher;
import com.google.common.base.Charsets;
import com.google.common.base.Splitter;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.io.CharStreams;
import com.google.common.io.Resources;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;

import java.io.IOException;
import java.io.InputStreamReader;
import java.util.List;
import java.util.Map;
import java.util.Random;

public abstract class SGDTestCase extends MahoutTestCase {
    /**
     * Permute the integers from 0 ... max-1
     *
     * @param gen The random number generator to use.
     * @param max The number of integers to permute
     * @return An array of jumbled integer values
     */
    static int[] permute(Random gen, int max) {
        int[] permutation = new int[max];
        permutation[0] = 0;
        for (int i = 1; i < max; i++) {
            int n = gen.nextInt(i + 1);
            if (n == i) {
                permutation[i] = i;
            } else {
                permutation[i] = permutation[n];
                permutation[n] = i;
            }
        }
        return permutation;
    }


    /**
     * Reads a file containing CSV data.  This isn't implemented quite the way you might like for a
     * real program, but does the job for reading test data.  Most notably, it will only read numbers,
     * not quoted strings.
     *
     * @param resourceName Where to get the data.
     * @return A matrix of the results.
     * @throws java.io.IOException If there is an error reading the data
     */
    static Matrix readCsv(String resourceName) throws IOException {
        Splitter onCommas = Splitter.on(',').trimResults(CharMatcher.anyOf(" \""));

        Readable isr = new InputStreamReader(Resources.getResource(resourceName).openStream(), Charsets.UTF_8);
        List<String> data = CharStreams.readLines(isr);
        String first = data.get(0);
        data = data.subList(1, data.size());

        List<String> values = Lists.newArrayList(onCommas.split(first));
        Matrix r = new DenseMatrix(data.size(), values.size());

        int column = 0;
        Map<String, Integer> labels = Maps.newHashMap();
        for (String value : values) {
            labels.put(value, column);
            column++;
        }
        r.setColumnLabelBindings(labels);

        int row = 0;
        for (String line : data) {
            column = 0;
            values = Lists.newArrayList(onCommas.split(line));
            for (String value : values) {
                r.set(row, column, Double.parseDouble(value));
                column++;
            }
            row++;
        }

        return r;
    }

}
