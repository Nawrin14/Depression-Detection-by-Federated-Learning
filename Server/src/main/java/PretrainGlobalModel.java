import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

public class PretrainGlobalModel {

    public static void main(String[] args) throws Exception {
        int seed = 100;
        int batchSize = 8;
        int numEpochs = 20;
        int numInputs = 15;
        int numOutputs = 5;
        double learningRate = 0.001;

        System.out.println(new ClassPathResource("").getPath());
        final String filenameTrain  = "src/main/resources/dataset/TrainData.csv";
        final String filenameTest  = "src/main/resources/dataset/TestData.csv";

        //Load the train data
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(filenameTrain)));
        DataSetIterator trainIterator = new RecordReaderDataSetIterator(rr,batchSize,0,5);

        //Load the test data
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File(filenameTest)));
        DataSetIterator testIterator = new RecordReaderDataSetIterator(rrTest,batchSize,0,5);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(learningRate, 0.9))
                .list()
                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(500)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder().nIn(500).nOut(1000)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(1000).nOut(numOutputs).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));    //Print loss after 100 parameter updates
        model.fit(trainIterator,numEpochs);

        //Evaluate model using test data
        Evaluation eval = new Evaluation(numOutputs);
        while(testIterator.hasNext()){
            DataSet ds = testIterator.next();
            INDArray features = ds.getFeatures();
            INDArray labels = ds.getLabels();
            INDArray prediction = model.output(features,false);

            eval.eval(labels,prediction);
        }

        //Print the evaluation statistics
        System.out.println(eval.stats());

        //Save model
        File saveLocation = new File("src/main/resources/globalmodel/global_model.zip");
        ModelSerializer.writeModel(model,saveLocation,true);
    }
}