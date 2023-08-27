import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class AverageLocalModelWeights {

    private static final String globalModel = "src/main/resources/globalmodel/global_model.zip";
    private static final String localModels = "src/main/resources/localmodel";

    public static void AverageWeights(List<File>files,File originalModel,int layer,double alpha) {

        //Load original global model
        MultiLayerNetwork model = null;
        try {
            model = ModelSerializer.restoreMultiLayerNetwork(originalModel,false);
        } catch (IOException e) {
            e.printStackTrace();
        }

        Map<String,INDArray> paramTable = model.paramTable();
        INDArray weight = paramTable.get(String.format("%d_W",layer));
        INDArray bias = paramTable.get(String.format("%d_b",layer));
        INDArray averageWeight = weight.mul(alpha);
        INDArray averageBias = bias.mul(alpha);

        //Average local model parameters
        int len = files.size();
        for(int i = 0; i < len; i++) {
            try {
                model = ModelSerializer.restoreMultiLayerNetwork(files.get(i),false);
            } catch (IOException e) {
                e.printStackTrace();
            }

            paramTable = model.paramTable();
            weight = paramTable.get(String.format("%d_W",layer));
            averageWeight = averageWeight.add(weight.mul(1.0-alpha).div(len));
            bias = paramTable.get(String.format("%d_b",layer));
            averageBias = averageBias.add(bias.mul(1.0-alpha).div(len));
        }

        //Update global model
        model.setParam(String.format("%d_W",layer),averageWeight);
        model.setParam(String.format("%d_b",layer),averageBias);

        //Save updated global model
        try {
            ModelSerializer.writeModel(model,globalModel,false);
        } catch (IOException e){
            e.printStackTrace();
        }
    }

    public static void main(String[] args) throws Exception {
        File dir = new File(localModels);
        File[] listOfFiles = dir.listFiles();
        List<File>models = new ArrayList<>();

        for(int i = 0; i < listOfFiles.length; i++) {
            if (listOfFiles[i].isFile()) {
                models.add(listOfFiles[i]);
            }
        }

        File originalModel = new File(globalModel);

        //layer is the layer number that is trained or averaged
        //alpha is the weight of the original model that is retained in the updated model
        AverageWeights(models,originalModel,2,0.3);
    }
}