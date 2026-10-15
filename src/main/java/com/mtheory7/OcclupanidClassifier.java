package com.mtheory7;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
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
import org.deeplearning4j.ui.model.storage.FileStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaDelta;
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
    private static int width = 400;
    private static int channels = 3; // Keep this as 3 it means RGB
    private static int numOfClasses = 2; // For now just using A. Floridensis and P. Utiliformis
    private static int batchSize = 76; // Not sure if this is right? Previously was 96
    private static long seed = 123L; // LOL What is this
    private static int epochs = 4; // Sure? Not sure what this is really...
    
    private static final String modelParentPath = "C:/Users/thesp/Code/deeplearning4j-occlupanid/models";
    private static final String imageToTestWithPath = "C:/Users/thesp/Code/deeplearning4j-occlupanid/testImages";
    private static final String testImagesPath = "C:/Users/thesp/Code/deeplearning4j-occlupanid/trainingData/data/test";
    private static final String trainImagesPath = "C:/Users/thesp/Code/deeplearning4j-occlupanid/trainingData/data/train";
    private static final String dl4jStatsFileName = "ui-stats.dl4j";
    private static final String modelFileName = "occlupanidModel.dl4j.zip";
    private static final String modelPath = modelParentPath + "/" + modelFileName;
    private static final String imageToTestWith = imageToTestWithPath + "/pu_test_01.jpg";

    private static final String[] LABELS = new String[] {
            "AdmordorsumFloridensis", "PalpatophoraUtiliformis"
    };

    public static void main(String[] args) throws Exception {
 /*       log.error("Starting...");
        OcclupanidClassifier occlupanidClassifier = new OcclupanidClassifier();
        log.error("Finished creating occlupanidClassifier object.");

        File trainParentDir = new File(trainImagesPath);
        FileSplit trainFileSplit = new FileSplit(trainParentDir, NativeImageLoader.ALLOWED_FORMATS);
        ParentPathLabelGenerator trainLabelMaker = new ParentPathLabelGenerator();
        ImageRecordReader trainRecordReader = new ImageRecordReader(height, width, channels, trainLabelMaker);
        trainRecordReader.initialize(trainFileSplit);
        DataSetIterator occTrainIter = new RecordReaderDataSetIterator.Builder(trainRecordReader, batchSize)
                .classification(1, numOfClasses)
                .build();
        log.error("Done creating occTrainIter object.");

        File testParentDir = new File(testImagesPath);
        FileSplit testFileSplit = new FileSplit(testParentDir, NativeImageLoader.ALLOWED_FORMATS);
        ParentPathLabelGenerator testLabelMaker = new ParentPathLabelGenerator();
        ImageRecordReader testRecordReader = new ImageRecordReader(height, width, channels, testLabelMaker);
        testRecordReader.initialize(testFileSplit);
        DataSetIterator occTestIter = new RecordReaderDataSetIterator.Builder(testRecordReader, batchSize)
                .classification(1, numOfClasses)
                .build();
        log.error("Done creating occTestIter object.");

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(occTrainIter);
        scaler.fit(occTestIter);
        occTrainIter.setPreProcessor(scaler);
        occTestIter.setPreProcessor(scaler);
        log.error("Scaling preProcessor set.");

        MultiLayerNetwork model = occlupanidClassifier.getModel();
        log.error("Done making model object.");
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new FileStatsStorage(new File(modelParentPath, dl4jStatsFileName));
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener( statsStorage), new ScoreIterationListener(50), new EvaluativeListener(occTestIter, 1, InvocationType.EPOCH_END));

        log.error("Beginning .fit command");
        model.fit(occTrainIter, epochs);

        log.error("Saving model...");
        model.save(new File(modelParentPath, modelFileName), true);

        *//* END OF MODEL TRAINING/GENERATION CODE --- BEGIN OF CODE TO TEST TRAINED/SAVED MODEL *//*

        try {
            log.error("Loading model...");
            File modelFile = new File(modelPath);
            MultiLayerNetwork loadedModel = ModelSerializer.restoreMultiLayerNetwork(modelFile);

            log.error("Loading and preprocessing image...");
            File imgFile = new File(imageToTestWith);

            NativeImageLoader loader = new NativeImageLoader(height, width, channels);
            INDArray imageTensor = loader.asMatrix(imgFile);

            ImagePreProcessingScaler testScaler = new ImagePreProcessingScaler(0, 1);
            testScaler.transform(imageTensor);

            log.error("Running inference...");
            INDArray output = loadedModel.output(imageTensor, false);

            int predictedClassIndex = Nd4j.argMax(output, 1).getInt(0);
            double confidence = output.getDouble(predictedClassIndex) * 100;

            log.error("\n--- Prediction Result ---");
            System.out.printf("Predicted Class: %s (Index: %d)%n", LABELS[predictedClassIndex], predictedClassIndex);
            System.out.printf("Confidence: %.2f%%%n", confidence);
            log.error("Full probabilities: " + output);

        } catch (IOException e) {
            System.err.println("Error reading model or image file: " + e.getMessage());
            e.printStackTrace();
        }

        System.exit(0);*/



        // --- Pipeline Dimensions & Setup ---
        // Resize all mixed inputs to a standard square size to fix spatial variance
        int height = 300;
        int width = 300;
        int channels = 3;
        int batchSize = 16;
        int numClasses = 2;
        int epochs = 20;
        long seed = 12345;
        Random rng = new Random(seed);

        // --- Dataset Paths ---
        // Expected directory structure:
        // /path/to/dataset/type_A/image1.jpg
        // /path/to/dataset/type_B/image2.jpg
        File rootDir = new File("C:/Users/thesp/Code/deeplearning4j-occlupanid/trainingData/data");
        if (!rootDir.exists()) {
            log.error("Please update the rootDir path to your local image folder.");
        }

        FileSplit fileSplit = new FileSplit(rootDir, BaseImageLoader.ALLOWED_FORMATS, rng);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        // Initialize ImageRecordReader which automatically resizes/crops inputs to specified height/width
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        recordReader.initialize(fileSplit);

        // Build Iterator (Label index is at position 1 because image pixels are index 0)
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numClasses);

        // Scale RGB values from [0, 255] down to [0, 1] for gradient stability
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        dataIter.setPreProcessor(scaler);

        // --- Model Architecture (Custom CNN from Scratch) ---
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
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
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX) // Binary categorical probability distribution
                        .build())
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10)); // Track loss metrics every 10 iterations

        // --- Training Loop ---
        log.error("Starting network training...");
        for (int i = 0; i < epochs; i++) {
            log.error("Epoch " + (i + 1) + " / " + epochs);
            model.fit(dataIter);
            dataIter.reset();
        }

        // --- Evaluation ---
        log.error("Evaluating model accuracy...");
        Evaluation eval = model.evaluate(dataIter);
        log.error(eval.stats());

        // --- Saving the Model ---
        File modelSaveFile = new File("BreadClipClassifierModel.zip");
        model.save(modelSaveFile, true);
        log.error("Model successfully exported to: " + modelSaveFile.getAbsolutePath());



    }

    public MultiLayerNetwork getModel()  {
        log.error("Building simple convolutional network...");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new AdaDelta())
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                        .nIn(channels).nOut(32).build())
                .layer(new BatchNormalization())
                .layer(new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).poolingType(SubsamplingLayer.PoolingType.MAX).build())
                .layer(new ConvolutionLayer.Builder().kernelSize(1,1).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                        .nOut(16).build())
                .layer(new BatchNormalization())
                .layer(new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                        .nOut(64).build())
                .layer(new BatchNormalization())
                .layer(new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).poolingType(SubsamplingLayer.PoolingType.MAX).build())
                .layer(new ConvolutionLayer.Builder().kernelSize(1,1).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                        .nOut(32).build())
                .layer(new BatchNormalization())
                .layer(new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                        .nOut(128).build())
                .layer(new BatchNormalization())
                .layer(new ConvolutionLayer.Builder().kernelSize(1,1).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                        .nOut(64).build())
                .layer(new BatchNormalization())
                .layer(new ConvolutionLayer.Builder().kernelSize(1,1).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                        .nOut(numOfClasses).build())
                .layer(new BatchNormalization())
                .layer(new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).poolingType(SubsamplingLayer.PoolingType.AVG).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(numOfClasses)
                        .dropOut(0.8)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        return model;
    }
}