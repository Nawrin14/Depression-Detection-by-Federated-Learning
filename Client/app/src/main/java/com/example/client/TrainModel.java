package com.example.client;

import android.os.Environment;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.io.File;

public class TrainModel  {

    public static final File downloadDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS);
    public static String baseDir =  Environment.getExternalStorageDirectory().getAbsolutePath();
    public static final File datasetLocation = new File(baseDir, "local_dataset.csv");
    public static String id = null;

    private static final int numHiddenNodes = 1000;
    private static final int numOutputs = 5;
    private static final double learningRate = 0.001;

    public static boolean isTransferred = false;

    public static FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
            .updater(new Sgd(learningRate))
            .seed(100)
            .build();

    public static MultiLayerNetwork model = null;

    public static MultiLayerNetwork TrainingModel(File file) {

        MultiLayerNetwork transferredModel = model;

        //Freeze the initial layers and configure the last layer
        if (!isTransferred) {
            transferredModel = new TransferLearning.Builder(model)
                        .fineTuneConfiguration(fineTuneConf)
                        .setFeatureExtractor(1).removeOutputLayer()
                        .addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .activation(Activation.SOFTMAX)
                                .nIn(numHiddenNodes).nOut(numOutputs)
                                .weightInit(WeightInit.XAVIER)
                                .build())
                        .build();

            isTransferred = true;
        }

        RecordReader rr = new CSVRecordReader();
        try {
            rr.initialize(new FileSplit(file));
        } catch (Exception e) {
            e.printStackTrace();
        }
        DataSetIterator iterator = new RecordReaderDataSetIterator(rr, 8, 0, 5);

        int numIteration = 1;

        //Train the last layer using local data
        while(iterator.hasNext()) {
            transferredModel.fit(iterator.next());

            double score = transferredModel.score();
            MainActivity.score += " Loss at iteration " + numIteration + ": " + score + "\n";
            numIteration++;
        }

        return transferredModel;
    }
}