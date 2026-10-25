package com.mtheory7.model;

import com.mtheory7.utils.PadToSquare;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.CropImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileGraphSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.FileStatsStorage;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.model.ResNet50;
import org.nd4j.common.primitives.Pair;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

public class OcclupanidModelTrainer {
    protected static final Logger log = LoggerFactory.getLogger(OcclupanidModelTrainer.class);

    private static final int height = 224;
    private static final int width = 224;
    private static final int channels = 3; // Keep this as 3 it means RGB
    private static final int numOfClasses = 2; // For now just using A.Floridensis and P.Utiliformis
    private static final int batchSize = 4;
    private static final long seed = 12345L;
    private static final int epochs = 100;

    public void trainModelAndSave(String modelPath, String dataPath) throws IOException {
        String modelCheckpointPath = modelPath + "/checkpoints";
        String modelSavePath = modelPath + "/deeplearning4jOcclupanidModel1.0.0.zip";
        String trainingDataPath = dataPath + "/train";
        String testingDataPath = dataPath + "/test";

        File trainingDataDirectory = new File(trainingDataPath);
        File testingDataDirectory = new File(testingDataPath);

        PipelineImageTransform transformPipeline = new PipelineImageTransform(
                Arrays.asList(new Pair<>(new PadToSquare(), 1.0), new Pair<>(new CropImageTransform(10), 0.3)), true);

        DataSetIterator trainingDataSetIterator = getDataSetIterator(trainingDataDirectory, transformPipeline);
        log.info("Detected training labels: {}", trainingDataSetIterator.getLabels());
        DataSetIterator testingDataSetIterator = getDataSetIterator(testingDataDirectory, null);

        VGG16ImagePreProcessor scaler = new VGG16ImagePreProcessor();
        trainingDataSetIterator.setPreProcessor(scaler);
        testingDataSetIterator.setPreProcessor(scaler);

        UIServer uiServer = UIServer.getInstance();
        File statsFile = new File(modelPath, "training_stats.dl4j");
        StatsStorage statsStorage = new FileStatsStorage(statsFile);
        uiServer.attach(statsStorage);

        log.info("Loading pre-trained ResNet50 weights from ImageNet...");
        ResNet50 zooModel = ResNet50.builder().build();
        ComputationGraph pretrainedNet = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);

        log.info("Configuring Transfer Learning...");
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .updater(new Adam(1e-4))
                .seed(seed)
                .build();

        log.info("Building ComputationGraph...");
        ComputationGraph model = new TransferLearning.GraphBuilder(pretrainedNet)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor("flatten_1")
                .removeVertexAndConnections("fc1000")
                .addLayer("predictions",
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(2048)
                                .nOut(numOfClasses)
                                .activation(Activation.SOFTMAX)
                                .build(),
                        "flatten_1")
                .setOutputs("predictions")
                .build();

        model.setListeners(
                new ScoreIterationListener(5),
                new StatsListener(statsStorage)
        );

        log.info("Configuring Early Stopping...");
        EarlyStoppingConfiguration<ComputationGraph> esConf = new EarlyStoppingConfiguration.Builder<ComputationGraph>()
                .epochTerminationConditions(
                        new MaxEpochsTerminationCondition(epochs),
                        new ScoreImprovementEpochTerminationCondition(5) // Wait 5 epochs before stopping if no improvement
                )
                .scoreCalculator(new DataSetLossCalculator(testingDataSetIterator, true))
                .evaluateEveryNEpochs(1)
                .modelSaver(new LocalFileGraphSaver(modelCheckpointPath))
                .build();

        EarlyStoppingGraphTrainer trainer = new EarlyStoppingGraphTrainer(esConf, model, trainingDataSetIterator);

        log.info("Starting network training with Early Stopping...");
        EarlyStoppingResult<ComputationGraph> result = trainer.fit();

        log.info("Training complete.");
        log.info("Termination reason: {}", result.getTerminationReason());
        log.info("Total epochs: {}", result.getTotalEpochs());
        log.info("Best epoch number: {}", result.getBestModelEpoch());
        log.info("Score at best epoch: {}", result.getBestModelScore());

        // Retrieve the best model found before it started overfitting
        ComputationGraph bestModel = result.getBestModel();

        log.info("Evaluating best model accuracy using testing data directory...");
        testingDataSetIterator.reset(); // Reset before final evaluation
        Evaluation eval = bestModel.evaluate(testingDataSetIterator);
        log.info(eval.stats());

        File modelSaveFile = new File(modelSavePath);
        bestModel.save(modelSaveFile, true);
        log.info("Model successfully exported to: {}", modelSaveFile.getAbsolutePath());
    }

    private DataSetIterator getDataSetIterator(File rootDir, PipelineImageTransform pipelineImageTransform) throws IOException {
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
}