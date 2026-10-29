package com.mtheory7;

import com.mtheory7.model.OcclupanidModelTester;
import com.mtheory7.model.OcclupanidModelTrainer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;

public class Main {
    protected static final Logger log = LoggerFactory.getLogger(Main.class);
    public static final List<String> CLASS_LABELS = Arrays.asList("AdmordorsumFloridensis", "PalpatophoraUtiliformis", "PorrectofrontusMechadeus");
    public static final String MODEL_NAME = "deeplearning4jOcclupanidModel1.2.0.zip";
    private static final String modelPath = "C:/Users/thesp/Code/deeplearning4j-occlupanid/models";
    private static final String dataPath = "C:/Users/thesp/Code/deeplearning4j-occlupanid/trainingData/data";

    public static void main(String[] args) throws Exception {
        log.info("Starting trainModelAndSave function...");
        OcclupanidModelTrainer occlupanidModelTrainer = new OcclupanidModelTrainer();
        occlupanidModelTrainer.trainModelAndSave(modelPath, dataPath);
        log.info("Finished trainModelAndSave function...");

        log.info("Starting model tests...");
        OcclupanidModelTester occlupanidModelTester = new OcclupanidModelTester();
        occlupanidModelTester.predictImages((modelPath + "/" + MODEL_NAME), dataPath);
        log.info("Finished testing model");

        System.exit(0);
    }
}