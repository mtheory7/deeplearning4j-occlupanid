package com.mtheory7;

import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.core.config.Configurator;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.*;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.jetbrains.annotations.NotNull;
import org.nd4j.common.primitives.Pair;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.ExponentialSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class OcclupanidClassifier {
    protected static final Logger log = LoggerFactory.getLogger(OcclupanidClassifier.class);

    private static int height = 224;
    private static int width = 224;
    private static int channels = 3; // Keep this as 3 it means RGB
    private static int numOfClasses = 2; // For now just using A. Floridensis and P. Utiliformis
    private static int batchSize = 16;
    private static long seed = 12345L;
    private static int epochs = 150;

    private static final String modelSavePath = "C:/Users/thesp/Code/deeplearning4j-occlupanid/models/deeplearning4jOcclupanidModel1.0.0.zip";
    private static final String trainingDataPath = "C:/Users/thesp/Code/deeplearning4j-occlupanid/trainingData/data/train";
    private static final String testingDataPath = "C:/Users/thesp/Code/deeplearning4j-occlupanid/trainingData/data/test";

    public static void main(String[] args) throws Exception {
        File trainingDataDirectory = new File(trainingDataPath);
        File testingDataDirectory = new File(testingDataPath);

        // 1. Define individual transforms
        ImageTransform padToSquare = new PadToSquare();
        //ImageTransform flip = new FlipImageTransform(1); // 1 = horizontal flip
        //ImageTransform rotate = new RotateImageTransform(new Random(seed), 15); // Random rotation up to 15 degrees
        ImageTransform crop = new CropImageTransform(10); // Crop 10 pixels off borders
        // 2. Combine them into a Pipeline
        // The boolean 'true' flags shuffle the order of transformations for randomness
        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(
                new Pair<>(padToSquare, 1.0),
                //new Pair<>(flip, 0.5),   // 50% chance to flip
                //new Pair<>(rotate, 0.8), // 80% chance to rotate
                new Pair<>(crop, 0.3)    // 30% chance to crop
        );
        PipelineImageTransform transformPipeline = new PipelineImageTransform(pipeline, true);
        DataSetIterator trainingDataSetIterator = getDataSetIterator(trainingDataDirectory, transformPipeline);
        log.info("Detected training labels: " + trainingDataSetIterator.getLabels());
        DataSetIterator testingDataSetIterator = getDataSetIterator(testingDataDirectory, null);
        log.info("Detected testing labels: " + trainingDataSetIterator.getLabels());
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        trainingDataSetIterator.setPreProcessor(scaler);
        testingDataSetIterator.setPreProcessor(scaler);

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);

        MultiLayerNetwork model = new MultiLayerNetwork(getMLConfiguration(numOfClasses));
        model.init();
        model.setListeners(
                new ScoreIterationListener(10),
                new StatsListener(statsStorage),
                new EvaluativeListener(testingDataSetIterator, 1, InvocationType.EPOCH_END)
        );

        log.info("Starting network training...");
        for (int i = 0; i < epochs; i++) {
            log.info("Epoch " + (i + 1) + " / " + epochs);
            model.fit(trainingDataSetIterator);
            trainingDataSetIterator.reset();
        }

        log.info("Evaluating model accuracy using testing data directory...");
        Evaluation eval = model.evaluate(testingDataSetIterator);
        log.info(eval.stats());

        File modelSaveFile = new File(modelSavePath);
        model.save(modelSaveFile, true);
        log.info("Model successfully exported to: " + modelSaveFile.getAbsolutePath());
    }

    private static DataSetIterator getDataSetIterator(File rootDir, PipelineImageTransform pipelineImageTransform) throws IOException {
        FileSplit fileSplit = new FileSplit(rootDir, BaseImageLoader.ALLOWED_FORMATS, new Random(seed));
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        if (pipelineImageTransform == null) {
            recordReader.initialize(fileSplit);
        } else {
            recordReader.initialize(fileSplit, pipelineImageTransform);
        }
        return new RecordReaderDataSetIterator(recordReader, batchSize, 1, numOfClasses);
    }

    public static MultiLayerConfiguration getMLConfiguration(int numOfClasses) {
        return new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                // Replaced static learning rate with an exponential decay schedule
                .updater(new Adam(new ExponentialSchedule(ScheduleType.EPOCH, 1e-3, 0.95)))
                //.updater(new Adam(1e-4))
                .l2(1e-4) // Retained L2 regularization, removed L1
                .weightInit(WeightInit.RELU)
                .list()
                // Block 1
                .layer(0, new ConvolutionLayer.Builder(3, 3)
                        .nIn(channels)
                        .nOut(32)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new BatchNormalization.Builder().build())
                .layer(2, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                // Block 2
                .layer(3, new ConvolutionLayer.Builder(3, 3)
                        .nOut(64)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(4, new BatchNormalization.Builder().build())
                .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                // Block 3
                .layer(6, new ConvolutionLayer.Builder(3, 3)
                        .nOut(128)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(7, new BatchNormalization.Builder().build())
                .layer(8, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                // Global Average Pooling
                .layer(9, new GlobalPoolingLayer.Builder(PoolingType.AVG).build())
                // Dense Layer
                .layer(10, new DenseLayer.Builder()
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())
                // Dropout Layer
                .layer(11, new DropoutLayer(0.5))
                // Output Layer
                .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numOfClasses)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(height, width, channels))
                .build();
    }
}