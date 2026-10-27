package com.mtheory7;

import com.mtheory7.model.OcclupanidModelTester;
import com.mtheory7.model.OcclupanidModelTrainer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Main {
    protected static final Logger log = LoggerFactory.getLogger(Main.class);
    private static final String modelPath = "C:/Users/thesp/Code/deeplearning4j-occlupanid/models";
    private static final String dataPath = "C:/Users/thesp/Code/deeplearning4j-occlupanid/trainingData/data";
    private static final String testImage01_AF = "C:/Users/thesp/Code/deeplearning4j-occlupanid/trainingData/data/AdmordorsumFloridensis/PXL_20261011_174130130.jpg";
    private static final String testImage02_AF = "C:/Users/thesp/Code/deeplearning4j-occlupanid/trainingData/data/AdmordorsumFloridensis/PXL_20261011_174249261.jpg";
    private static final String testImage01_PU = "C:/Users/thesp/Code/deeplearning4j-occlupanid/trainingData/data/PalpatophoraUtiliformis/PXL_20261011_181610016.jpg";
    private static final String testImage02_PU = "C:/Users/thesp/Code/deeplearning4j-occlupanid/trainingData/data/PalpatophoraUtiliformis/PXL_20261011_181405016.jpg";

    public static void main(String[] args) throws Exception {
        log.info("Starting trainModelAndSave function...");
        OcclupanidModelTrainer occlupanidModelTrainer = new OcclupanidModelTrainer();
        occlupanidModelTrainer.trainModelAndSave(modelPath, dataPath);
        log.info("Finished trainModelAndSave function...");

        log.info("Starting model tests...");
        OcclupanidModelTester occlupanidModelTester = new OcclupanidModelTester();
        occlupanidModelTester.predictImage((modelPath + "/deeplearning4jOcclupanidModel1.1.0-SNAPSHOT.zip"), testImage01_AF);
        occlupanidModelTester.predictImage((modelPath + "/deeplearning4jOcclupanidModel1.1.0-SNAPSHOT.zip"), testImage02_AF);
        occlupanidModelTester.predictImage((modelPath + "/deeplearning4jOcclupanidModel1.1.0-SNAPSHOT.zip"), testImage01_PU);
        occlupanidModelTester.predictImage((modelPath + "/deeplearning4jOcclupanidModel1.1.0-SNAPSHOT.zip"), testImage02_PU);
        log.info("Finished testing model");

        System.exit(0);
    }
}