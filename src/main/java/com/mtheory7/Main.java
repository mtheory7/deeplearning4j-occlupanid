package com.mtheory7;

import com.mtheory7.model.OcclupanidModelTester;
import com.mtheory7.model.OcclupanidModelTrainer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Main {
    protected static final Logger log = LoggerFactory.getLogger(Main.class);
    private static final String modelPath = "C:/Users/thesp/Code/deeplearning4j-occlupanid/models";
    private static final String dataPath = "C:/Users/thesp/Code/deeplearning4j-occlupanid/trainingData/data";

    public static void main(String[] args) throws Exception {
        log.info("Starting trainModelAndSave function...");
        OcclupanidModelTrainer occlupanidModelTrainer = new OcclupanidModelTrainer();
        occlupanidModelTrainer.trainModelAndSave(modelPath, dataPath);
        log.info("Finished trainModelAndSave function...");

        log.info("Starting model tests...");
        OcclupanidModelTester occlupanidModelTester = new OcclupanidModelTester();
        occlupanidModelTester.predictImages((modelPath + "/deeplearning4jOcclupanidModel1.1.0-SNAPSHOT.zip"), dataPath);
        log.info("Finished testing model");

        System.exit(0);
    }
}