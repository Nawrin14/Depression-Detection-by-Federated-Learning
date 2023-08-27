package com.example.client;

import android.Manifest;
import android.app.Activity;
import android.app.DownloadManager;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PackageManager;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.BatteryManager;
import android.os.Bundle;
import android.os.Environment;
import android.provider.Settings;
import android.view.View;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.fragment.app.DialogFragment;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.storage.FirebaseStorage;
import com.google.firebase.storage.OnProgressListener;
import com.google.firebase.storage.StorageReference;
import com.google.firebase.storage.UploadTask;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class MainActivity extends AppCompatActivity implements SensorEventListener,
        AdapterView.OnItemSelectedListener, ActivityCompat.OnRequestPermissionsResultCallback, GenderChoiceDialog.GenderChoiceListener {

    //File read and write permissions
    private static final int REQUEST_EXTERNAL_STORAGE = 1;
    private static String[] PERMISSIONS_STORAGE = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
    };

    public static void verifyStoragePermission(Activity activity) {
        //Get permission status
        int permission = ActivityCompat.checkSelfPermission(activity, Manifest.permission.WRITE_EXTERNAL_STORAGE);
        if (permission != PackageManager.PERMISSION_GRANTED) {
            //Request permission
            ActivityCompat.requestPermissions(
                    activity,
                    PERMISSIONS_STORAGE,
                    REQUEST_EXTERNAL_STORAGE
            );
        }
    }

    //Get battery level
    private BroadcastReceiver mBatInfoReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context ctxt, Intent intent) {
            int level = intent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1);
            int scale = intent.getIntExtra(BatteryManager.EXTRA_SCALE, -1);
            batteryPct = level * 100 / (float) scale;
        }
    };

    //Sensor variables
    private static SensorManager mSensorManager;
    private static Sensor mAccelerometer;
    private static Sensor mGravity;
    private static Sensor mGyroscope;
    private static Sensor mMagnetic;

    public static String score = "";
    private static String currentActivity = "None";
    private double[][] dataSets = new double[1000][15];  //Local dataset contains less than 1000 samples and 15 features
    private static final int mCounts = 1000;
    private static final int[] sampleShape = {1, 15};

    int i = 0;
    double batteryPct = 0;
    int genderMale;
    int genderFemale;
    private Thread thread;

    private static boolean isTraining = false;
    private static boolean isLoading = false;
    private static boolean isLabeling = false;

    TextView message;
    TextView result;
    ProgressBar progressBar;

    private static File globalModel;
    private static File localModel;

    Uri fileURI;
    StorageReference storageReference;
    DatabaseReference databaseReference;
    String database_URL = "https://fl-client-89ccb-default-rtdb.firebaseio.com/";

    //Start thread
    private void feedMultiple() {
        if (thread != null) {
            thread.interrupt();
        }

        thread = new Thread(new Runnable() {
            @Override
            public void run() {
                while (true) {
                    try {
                        Thread.sleep(10);
                    } catch (Exception e) {
                        // TODO Auto-generated catch block
                        e.printStackTrace();
                    }
                }
            }
        });

        thread.start();
    }

    public void initDataset(){
        i=0;
        dataSets = new double[1000][15];
    }

    //Generate samples from collected sensor data
    public double[] groupSensorData() {

        double[] sample = new double[15];
        double accX_sum = 0;
        double accY_sum = 0;
        double accZ_sum = 0;

        double gravX_sum = 0;
        double gravY_sum = 0;
        double gravZ_sum = 0;

        double gyroX_sum = 0;
        double gyroY_sum = 0;
        double gyroZ_sum = 0;

        double magX_sum = 0;
        double magY_sum = 0;
        double magZ_sum = 0;

        double battery_sum = 0;

        //Perform average of sensor data to generate one sample
        for (int j = 0; j < i; j++) {

            accX_sum += dataSets[j][0];
            accY_sum += dataSets[j][1];
            accZ_sum += dataSets[j][2];

            battery_sum += dataSets[j][3];

            gravX_sum += dataSets[j][4];
            gravY_sum += dataSets[j][5];
            gravZ_sum += dataSets[j][6];

            gyroX_sum += dataSets[j][7];
            gyroY_sum += dataSets[j][8];
            gyroZ_sum += dataSets[j][9];

            magX_sum += dataSets[j][10];
            magY_sum += dataSets[j][11];
            magZ_sum += dataSets[j][12];
        }

        sample[0] = accX_sum / i;
        sample[1] = accY_sum / i;
        sample[2] = accZ_sum / i;

        sample[3] = battery_sum / i;

        sample[4] = gravX_sum / i;
        sample[5] = gravY_sum / i;
        sample[6] = gravZ_sum / i;

        sample[7] = gyroX_sum / i;
        sample[8] = gyroY_sum / i;
        sample[9] = gyroZ_sum / i;

        sample[10] = magX_sum / i;
        sample[11] = magY_sum / i;
        sample[12] = magZ_sum / i;

        sample[13] = genderFemale;
        sample[14] = genderMale;

        return sample;
    }

    //Get sensor data
    @Override
    public final void onSensorChanged(SensorEvent sensorEvent) {

        Sensor sensor = sensorEvent.sensor;

        dataSets[i][3] = batteryPct;
        dataSets[i][13] = genderFemale;
        dataSets[i][14] = genderMale;

        if (sensor.getType() == Sensor.TYPE_ACCELEROMETER && mAccelerometer != null) {
            dataSets[i][0] = sensorEvent.values[0];
            dataSets[i][1] = sensorEvent.values[1];
            dataSets[i][2] = sensorEvent.values[2];
        } else {
            dataSets[i][0] = 0.0;
            dataSets[i][1] = 0.0;
            dataSets[i][2] = 0.0;
        }

        if (sensor.getType() == Sensor.TYPE_GRAVITY && mGravity != null) {
            dataSets[i][4] = sensorEvent.values[0];
            dataSets[i][5] = sensorEvent.values[1];
            dataSets[i][6] = sensorEvent.values[2];
        } else {
            dataSets[i][4] = 0.0;
            dataSets[i][5] = 0.0;
            dataSets[i][6] = 0.0;
        }

        if (sensor.getType() == Sensor.TYPE_GYROSCOPE && mGyroscope != null) {
            dataSets[i][7] = sensorEvent.values[0];
            dataSets[i][8] = sensorEvent.values[1];
            dataSets[i][9] = sensorEvent.values[2];
        } else {
            dataSets[i][7] = 0.0;
            dataSets[i][8] = 0.0;
            dataSets[i][9] = 0.0;
        }

        if (sensor.getType() == Sensor.TYPE_MAGNETIC_FIELD && mMagnetic != null) {
            dataSets[i][10] = sensorEvent.values[0];
            dataSets[i][11] = sensorEvent.values[1];
            dataSets[i][12] = sensorEvent.values[2];
        } else {
            dataSets[i][10] = 0.0;
            dataSets[i][11] = 0.0;
            dataSets[i][12] = 0.0;

            i++;
        }

        //Re-initialize if local dataset exceeds 1000 samples
        if (i >= mCounts) {
            initDataset();
        }
    }

    @Override
    public final void onAccuracyChanged(Sensor sensor, int accuracy) {

    }

    //Spinner
    @Override
    public void onItemSelected(AdapterView<?> parent, View view,
                               int pos, long id) {
        currentActivity = parent.getItemAtPosition(pos).toString();
    }

    @Override
    public void onNothingSelected(AdapterView<?> parent) {

    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        verifyStoragePermission(MainActivity.this);

        message = (TextView)findViewById(R.id.textView_message);
        result = (TextView)findViewById(R.id.textView_result);
        progressBar = (ProgressBar) findViewById(R.id.progressBar);

        storageReference = FirebaseStorage.getInstance().getReference();
        databaseReference = FirebaseDatabase.getInstance(database_URL).getReference();

        DialogFragment genderChoiceDialog = new GenderChoiceDialog();
        genderChoiceDialog.setCancelable(false);
        genderChoiceDialog.show(getSupportFragmentManager(), "Gender choice dialog");

        mSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        mAccelerometer = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        mGravity = mSensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY);
        mGyroscope = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        mMagnetic = mSensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);

        this.registerReceiver(this.mBatInfoReceiver, new IntentFilter(Intent.ACTION_BATTERY_CHANGED));

        if (mAccelerometer != null) {
            mSensorManager.registerListener(this, mAccelerometer, SensorManager.SENSOR_DELAY_NORMAL);
        }

        if (mGravity != null) {
            mSensorManager.registerListener(this, mGravity, SensorManager.SENSOR_DELAY_NORMAL);
        }

        if (mGyroscope != null) {
            mSensorManager.registerListener(this, mGyroscope, SensorManager.SENSOR_DELAY_NORMAL);
        }

        if (mMagnetic != null) {
            mSensorManager.registerListener(this, mMagnetic, SensorManager.SENSOR_DELAY_NORMAL);
        }

        Spinner activities = findViewById(R.id.spinner);
        activities.setOnItemSelectedListener(this);

        score = "";
        feedMultiple();

        //Get unique device ID
        TrainModel.id = Settings.Secure.getString(getContentResolver(), Settings.Secure.ANDROID_ID);
        globalModel = new File(TrainModel.downloadDir, "global_model.zip");
        localModel = new File(TrainModel.baseDir, TrainModel.id + "_local_model.zip");

        Button button = (Button) findViewById(R.id.button_get_model);
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (isLoading || isTraining || isLabeling) {
                    return;
                }

                result.setText("");
                isLoading = true;
                AsyncTaskRunner runner = new AsyncTaskLoadModel();
                runner.execute();
                progressBar.setVisibility(View.VISIBLE);
            }
        });

        Button button_train = (Button) findViewById(R.id.button_train);
        button_train.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (isTraining || isLoading || isLabeling) {
                    return;
                }

                result.setText("");

                if(!globalModel.exists()){
                    message.setText(R.string.warning_load_model);
                    return;
                }

                try {
                    TrainModel.model = ModelSerializer.restoreMultiLayerNetwork(globalModel, false);
                } catch (IOException e) {
                    e.printStackTrace();
                }

                isTraining = true;
                AsyncTaskRunner runner = new AsyncTaskTrainModel();
                runner.execute();
                progressBar.setVisibility(View.VISIBLE);
            }
        });

        Button button_label = (Button) findViewById(R.id.button_label);
        button_label.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (isTraining || isLoading || isLabeling) {
                    return;
                }

                result.setText("");
                isLabeling = true;
                AsyncTaskRunner runner = new AsyncTaskLabelling();
                runner.execute();
                progressBar.setVisibility(View.VISIBLE);
            }
        });

        Button button_infer = (Button) findViewById(R.id.button_inference);
        button_infer.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (isTraining || isLoading || isLabeling) {
                    return;
                }

                result.setText("");

                //Perform prediction using global model
                try {
                    TrainModel.model = ModelSerializer.restoreMultiLayerNetwork(globalModel, false);
                } catch (IOException e) {
                    e.printStackTrace();
                }

                //Perform prediction using local model
                try {
                    TrainModel.model = ModelSerializer.restoreMultiLayerNetwork(localModel, false);
                } catch (IOException e) {
                    e.printStackTrace();
                }

                if (TrainModel.model == null) {
                    message.setText(R.string.warning_load_model);
                    return;
                }

                double[] sample = groupSensorData();
                if (sample == null) {
                    return;
                }

                //Predict label from new data
                INDArray sampleToInfer = Nd4j.create(ArrayUtil.flattenDoubleArray(sample), sampleShape);
                INDArray predicted = TrainModel.model.output(sampleToInfer, false);
                INDArray index = predicted.argMax();
                int[] pl = index.toIntVector();
                String predictedLabel = "Predicted label: " + LabelIndex.labels[pl[0]];
                message.setText(predictedLabel);
            }
        });

        Button button_result = (Button) findViewById(R.id.button_result);
        button_result.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (isTraining || isLoading || isLabeling) {
                    return;
                }

                //Check if local model is trained
                if (TrainModel.model == null || score.isEmpty()) {
                    message.setText(R.string.warning_train_model);
                    return;
                }
                result.setText(score);
            }
        });
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (thread != null) {
            thread.interrupt();
        }
        mSensorManager.unregisterListener(this);
    }

    @Override
    protected void onResume() {
        super.onResume();
        mSensorManager.registerListener(this, mAccelerometer, SensorManager.SENSOR_DELAY_NORMAL);
        mSensorManager.registerListener(this, mGravity, SensorManager.SENSOR_DELAY_NORMAL);
        mSensorManager.registerListener(this, mGyroscope, SensorManager.SENSOR_DELAY_NORMAL);
        mSensorManager.registerListener(this, mMagnetic, SensorManager.SENSOR_DELAY_NORMAL);
    }

    @Override
    protected void onPostResume() {
        super.onPostResume();
    }

    @Override
    protected void onDestroy() {
        mSensorManager.unregisterListener(MainActivity.this);
        thread.interrupt();
        super.onDestroy();
    }

    @Override
    public void onBackPressed() {
        super.onBackPressed();

        if(TrainModel.datasetLocation.exists()){
            TrainModel.datasetLocation.delete();
        }
    }

    @Override
    public void onPositiveButtonClicked(String[] list, int position) {
        if (position == 0) {
            genderMale = 1;
            genderFemale = 0;
        } else {
            genderMale = 0;
            genderFemale = 1;
        }
    }

    private class AsyncTaskRunner extends AsyncTask<Void, Integer, Integer> {

        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            progressBar.setVisibility(View.INVISIBLE);
        }

        @Override
        protected Integer doInBackground(Void... params) {
            return 0;
        }

        @Override
        protected void onProgressUpdate(Integer... values) {
            super.onProgressUpdate(values);
        }

        @Override
        protected void onPostExecute(Integer result) {
            super.onPostExecute(result);
            progressBar.setVisibility(View.INVISIBLE);
        }
    }

    //Download global model
    private class AsyncTaskLoadModel extends AsyncTaskRunner {

        @Override
        protected void onPreExecute() {
            super.onPreExecute();

            String content = "Loading model...";
            message.setText(content);
        }

        @Override
        protected Integer doInBackground(Void... params) {
            try {
                downloadModel();
                Thread.sleep(10000);
            } catch (Exception e) {
                e.printStackTrace();
            }
            return 0;
        }

        @Override
        protected void onPostExecute(Integer result) {
            super.onPostExecute(result);

            globalModel = new File(TrainModel.downloadDir, "global_model.zip");
            localModel = new File(TrainModel.baseDir, TrainModel.id + "_local_model.zip");

            if(globalModel.exists()) {
                message.setText(getString(R.string.model_loaded));

                if(localModel.exists()) {
                    localModel.delete();
                }
            }
            else {
                message.setText(getString(R.string.load_error));
            }

            TrainModel.isTransferred = false;
            isLoading = false;
        }
    }

    private void downloadModel() {
        final String fileName = "global_model.zip";

        FirebaseStorage storage = FirebaseStorage.getInstance();
        StorageReference storageRef = storage.getReference();
        storageRef.child("Global Model").child(fileName).getDownloadUrl()
                .addOnSuccessListener(new OnSuccessListener<Uri>() {
                    @Override
                    public void onSuccess(Uri uri) {
                        DownloadManager.Request r = new DownloadManager.Request(uri);
                        r.setDestinationInExternalPublicDir(Environment.DIRECTORY_DOWNLOADS, fileName);
                        r.allowScanningByMediaScanner();
                        DownloadManager dm = (DownloadManager) getSystemService(DOWNLOAD_SERVICE);
                        dm.enqueue(r);
                    }
                }).addOnFailureListener(new OnFailureListener() {
                    @Override
                    public void onFailure(@NonNull Exception exception) {
                        Toast.makeText(MainActivity.this, "Model load failed", Toast.LENGTH_SHORT).show();
                    }
                });
    }

    //Train local model
    private class AsyncTaskTrainModel extends AsyncTaskRunner {

        Boolean modelTrained = false;

        @Override
        protected void onPreExecute() {
            super.onPreExecute();

            String content = "Training model...";
            message.setText(content);
        }

        @Override
        protected Integer doInBackground(Void... params) {
            try {
                if(!TrainModel.datasetLocation.exists()){
                    message.setText(R.string.warning_label_data);
                    return 0;
                }

                score = "";
                localModel = new File(TrainModel.baseDir, TrainModel.id + "_local_model.zip");
                MultiLayerNetwork trainedModel = TrainModel.TrainingModel(TrainModel.datasetLocation);
                ModelSerializer.writeModel(trainedModel, localModel, false);
                TrainModel.model = trainedModel;
                modelTrained = true;

            } catch (Exception e) {
                e.printStackTrace();
            }
            return 0;
        }

        @Override
        protected void onPostExecute(Integer result) {
            super.onPostExecute(result);

            globalModel = new File(TrainModel.downloadDir, "global_model.zip");
            localModel = new File(TrainModel.baseDir, TrainModel.id + "_local_model.zip");

            if(modelTrained){
                message.setText(R.string.train_finish);

                fileURI = Uri.fromFile(localModel);
                uploadModel(fileURI);

                TrainModel.datasetLocation.delete();
                globalModel.delete();
                initDataset();
            }
            isTraining = false;
        }
    }

    private void uploadModel(Uri fileURI) {
        StorageReference reference = storageReference.child("Local Models").child(TrainModel.id + "_local_model.zip");
        reference.putFile(fileURI)
                .addOnSuccessListener(new OnSuccessListener<UploadTask.TaskSnapshot>() {
                    @Override
                    public void onSuccess(UploadTask.TaskSnapshot taskSnapshot) {

                    }
                }).addOnProgressListener(new OnProgressListener<UploadTask.TaskSnapshot>() {
                    @Override
                    public void onProgress(@NonNull UploadTask.TaskSnapshot taskSnapshot) {

                    }
                });
    }

    //Label local dataset
    private class AsyncTaskLabelling extends  AsyncTaskRunner {

        @Override
        protected void onPreExecute() {
            super.onPreExecute();

            String content = "Labeling sample...";
            message.setText(content);
        }

        @Override
        protected Integer doInBackground(Void... params) {
            try {
                for(int j = 0; j < i; j++){

                    double[] sample = dataSets[j];
                    if (sample == null) {
                        return 0;
                    }

                    int index = LabelIndex.labelToNum(currentActivity);
                    if (index == -1) {
                        return 0;
                    }

                    try {
                        BufferedWriter br = new BufferedWriter(new FileWriter(
                                TrainModel.datasetLocation.toString(), true));
                        StringBuilder sb = new StringBuilder();
                        sb.append(String.format("%d", index));

                        for (double element : sample) {
                            sb.append(String.format(",%.4f", element));
                        }

                        br.write(sb.toString());
                        br.newLine();
                        br.close();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            } catch (RuntimeException e) {
                e.printStackTrace();
            }
            return 0;
        }

        @Override
        protected void onPostExecute(Integer result) {
            super.onPostExecute(result);

            message.setText(i + " data collected.");
            isLabeling = false;
            initDataset();
        }
    }
}