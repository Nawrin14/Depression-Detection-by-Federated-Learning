import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;

public class TestGlobalModel {

    public static void main(String[] args) throws Exception {

        int batchSize = 8;
        int numOutputs = 5;

        final String filenameTest  = "src/main/resources/dataset/TestData.csv";

        //Load the test data
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File(filenameTest)));
        DataSetIterator iterator = new RecordReaderDataSetIterator(rrTest,batchSize,0,5);

        //Load model
        File loadLocation = new File("src/main/resources/globalmodel/global_model.zip");
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(loadLocation,false);

        //Evaluate model using test data
        Evaluation eval = new Evaluation(numOutputs);
        while(iterator.hasNext()){
            DataSet ds = iterator.next();
            INDArray features = ds.getFeatures();
            INDArray labels = ds.getLabels();
            INDArray prediction = model.output(features,false);

            eval.eval(labels,prediction);
        }

        //Print the evaluation statistics
        System.out.println(eval.stats());
    }
}