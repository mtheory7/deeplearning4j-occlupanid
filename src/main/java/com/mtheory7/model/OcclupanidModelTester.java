package com.mtheory7.model;

import com.mtheory7.utils.PadToSquare;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.PipelineImageTransform;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Arrays;
import java.util.List;

public class OcclupanidModelTester {
    protected static final Logger log = LoggerFactory.getLogger(OcclupanidModelTester.class);

    public void predictImage(String modelPath, String imagePath) throws Exception {
        // Set loadUpdater to false in order to prevent Adam state from loading saving memory
        ComputationGraph model = ComputationGraph.load(new File(modelPath), false);

        log.info("Preparing image loader and transforms...");
        // Same image padding as in training (a must!)
        PipelineImageTransform inferenceTransform = new PipelineImageTransform(Arrays.asList(
                new Pair<>(new PadToSquare(), 1.0)
        ), true);

        NativeImageLoader loader = new NativeImageLoader(224, 224, 3, inferenceTransform);
        INDArray imageMatrix = loader.asMatrix(new File(imagePath));
        DataNormalization scaler = new VGG16ImagePreProcessor();
        scaler.transform(imageMatrix);

        log.info("Running inference...");
        // The output() method returns an array of INDArrays (one for each output layer).
        // SqueezeNet only has one output layer, so we take index 0.
        INDArray[] output = model.output(false, imageMatrix);
        INDArray predictions = output[0];

        // Map the indices back to labels, DL4J sorts folder names alphabetically during training
        List<String> classLabels = Arrays.asList("AdmordorsumFloridensis", "PalpatophoraUtiliformis");
        int predictedClassIndex = predictions.argMax(1).getInt(0);
        double confidence = predictions.getDouble(predictedClassIndex) * 100.0;

        log.info("--- Prediction Results ---");
        log.info("Predicted Class: {}", classLabels.get(predictedClassIndex));
        log.info("Confidence: {}%", String.format("%.2f", confidence));
        log.info("Raw Probabilities: A. Floridensis: {}, P. Utiliformis: {}",
                String.format("%.4f", predictions.getDouble(0)),
                String.format("%.4f", predictions.getDouble(1)));
    }
}
