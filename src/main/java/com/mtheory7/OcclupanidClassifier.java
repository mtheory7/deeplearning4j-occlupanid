package com.mtheory7;

import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.core.config.Configurator;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jetbrains.annotations.NotNull;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class OcclupanidClassifier {
    protected static final Logger log = LoggerFactory.getLogger(OcclupanidClassifier.class);

    private static int height = 300;
    private static int width = 300;
    private static int channels = 3; // Keep this as 3 it means RGB
    private static int numOfClasses = 2; // For now just using A. Floridensis and P. Utiliformis
    private static int batchSize = 16; // Not sure if this is right? Previously was 96
    private static long seed = 12345L; // LOL What is this
    private static int epochs = 20; // Sure? Not sure what this is really...

    private static final String modelParentPath = "C:/Users/thesp/Code/deeplearning4j-occlupanid/models";
    private static final String modelFileName = "deeplearning4jOcclupanidModel.zip";
    private static final String modelPath = modelParentPath + "/" + modelFileName;

    private static final String trainingDataPath = "C:/Users/thesp/Code/deeplearning4j-occlupanid/trainingData/data";

    private static final String[] LABELS = new String[] {
            "AdmordorsumFloridensis", "PalpatophoraUtiliformis"
    };

    public static void main(String[] args) throws Exception {
        // --- Dataset Paths ---
        // Expected directory structure:
        // /path/to/dataset/type_A/image1.jpg
        // /path/to/dataset/type_B/image2.jpg
        File trainingDataDirectory = new File(trainingDataPath);
        if (!trainingDataDirectory.exists()) {
            log.info("Please update the rootDir path to your local image folder.");
        }

        DataSetIterator dataSetIterator = getDataSetIterator(trainingDataDirectory);

        // --- Model Architecture (Custom CNN from Scratch) ---
        MultiLayerNetwork model = new MultiLayerNetwork(getMLConfiguration(numOfClasses));
        model.init();
        model.setListeners(new ScoreIterationListener(10)); // Track loss metrics every 10 iterations

        // --- Training Loop ---
        log.info("Starting network training...");
        for (int i = 0; i < epochs; i++) {
            log.info("Epoch " + (i + 1) + " / " + epochs);
            model.fit(dataSetIterator);
            dataSetIterator.reset();
        }

        // --- Evaluation ---
        log.info("Evaluating model accuracy...");
        Evaluation eval = model.evaluate(dataSetIterator);
        log.info(eval.stats());

        // --- Saving the Model ---
        File modelSaveFile = new File(modelPath);
        model.save(modelSaveFile, true);
        log.info("Model successfully exported to: " + modelSaveFile.getAbsolutePath());

        System.exit(0);
    }

    private static DataSetIterator getDataSetIterator(File rootDir) throws IOException {
        FileSplit fileSplit = new FileSplit(rootDir, BaseImageLoader.ALLOWED_FORMATS, new Random(seed));
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        // Initialize ImageRecordReader which automatically resizes/crops inputs to specified height/width
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        recordReader.initialize(fileSplit);

        // Build Iterator (Label index is at position 1 because image pixels are index 0)
        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numOfClasses);

        // Scale RGB values from [0, 255] down to [0, 1] for gradient stability
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        dataSetIterator.setPreProcessor(scaler);
        return dataSetIterator;
    }

    public static MultiLayerConfiguration getMLConfiguration(int numOfClasses) {
        return new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(1e-3)) // Fast convergence optimization
                .l2(5e-4) // Regularization to prevent overfitting on small datasets
                .list()
                .layer(0, new ConvolutionLayer.Builder(3, 3)
                        .nIn(channels)
                        .nOut(32)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .nOut(64)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(4, new ConvolutionLayer.Builder(3, 3)
                        .nOut(128)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(6, new DenseLayer.Builder()
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())
                .layer(7, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numOfClasses)
                        .activation(Activation.SOFTMAX) // Binary categorical probability distribution
                        .build())
                .setInputType(InputType.convolutional(height, width, channels))
                .build();
    }
}