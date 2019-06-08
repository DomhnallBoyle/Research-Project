/*
 * Filename: VideoCapture.java
 * Description: Used to capture angles and video footage
 * Author: Domhnall Boyle
 * Maintained by: Domhnall Boyle
 */

package com.example.domhnall.avdatacapture;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import android.app.Activity;
import android.content.Context;
import android.hardware.Camera;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.media.CamcorderProfile;
import android.media.MediaRecorder;
import android.media.MediaScannerConnection;
import android.os.Bundle;
import android.os.Handler;
import android.os.SystemClock;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.TextView;
import android.widget.Toast;


public class VideoCapture extends Activity implements SensorEventListener {
    /**
     * Class for capturing video and capturing angles from the game rotation sensor
     */

    // constants
    private static final int APS = 25;
    private static final String OUTPUT_DIRECTORY = "/storage/emulated/0/";

    // camera stuff
    private Camera myCamera;
    private MyCameraSurfaceView myCameraSurfaceView;
    private MediaRecorder mediaRecorder;
    private TextView angle, time;

    // UI stuff
    private Button recordButton;
    private SurfaceHolder surfaceHolder;
    private boolean recording;

    // sensor stuff
    private SensorManager sensorManager;
    private Sensor grvSensor;
    private float[] rotationMatrix = new float[9];
    private float[] readings = new float[3];
    private float offset, reading;
    private boolean offsetCalculated = false;

    // handler stuff
    private ArrayList<Float> angles, outputAngles;
    private Handler timeHandler, angleHandler;
    private Runnable angleRunnable;

    // time stuff
    private long msTime, startTime, timeBuff, updateTime = 0L ;
    private int seconds, minutes, ms;

    /**
     * Called when the activity is first created
     * @param savedInstanceState
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // not recording initially
        recording = false;

        // set the view
        setContentView(R.layout.activity_video_capture);

        // get camera for preview
        myCamera = getCameraInstance();
        if(myCamera == null){
            Toast.makeText(VideoCapture.this,
                    "Fail to get Camera",
                    Toast.LENGTH_LONG).show();
        }

        // set the camera view as the view
        myCameraSurfaceView = new MyCameraSurfaceView(this, myCamera);
        FrameLayout myCameraPreview = (FrameLayout)findViewById(R.id.videoview);
        myCameraPreview.addView(myCameraSurfaceView);

        // setup the record button event listener
        recordButton = (Button)findViewById(R.id.button_record);
        recordButton.setOnClickListener(myButtonOnClickListener);

        // setup the text labels - make invisible initially
        angle = (TextView)findViewById(R.id.textview_angle);
        angle.setVisibility(View.INVISIBLE);
        time = (TextView)findViewById(R.id.textview_time);
        time.setVisibility(View.INVISIBLE);

        // setup the sensors
        sensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        grvSensor = (Sensor) sensorManager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR);
        sensorManager.registerListener(this, grvSensor, SensorManager.SENSOR_DELAY_NORMAL);
        angles = new ArrayList<Float>();
        outputAngles = new ArrayList<Float>();

        // setup the angle handler - ran every 1 second
        angleHandler = new Handler();
        angleRunnable = new Runnable() {
            public void run() {

                // if there is enough angles
                if (angles.size() >= APS) {

                    // set the displacement
                    int displacement = angles.size() / APS;

                    // loop over the angles, going up in displacements
                    for (int i = 0; i < angles.size(); i += displacement) {
                        outputAngles.add(angles.get(i));
                    }

                    // clear the array list of angles
                    angles.clear();
                }
                angleHandler.postDelayed(angleRunnable, 1000);
            }
        };

        // set the label handler for displaying the elapsed time - ran every 0 seconds
        timeHandler = new Handler();
        timeHandler.postDelayed(new Runnable() {
            public void run() {
                // get the elapsed time
                msTime = SystemClock.uptimeMillis() - startTime;
                updateTime = timeBuff + msTime;
                seconds = (int) (updateTime / 1000);
                minutes = seconds / 60;
                seconds = seconds % 60;
                ms = (int) (updateTime % 1000);

                // update the text label
                time.setText("Time: " + minutes + ":"
                        + String.format("%02d", seconds) + ":"
                        + String.format("%03d", ms));

                // run the method again straight away
                timeHandler.postDelayed(this, 0);
            }
        }, 0);

        // setup the output directories
        File outputDirectory = new File(OUTPUT_DIRECTORY);
        if (!outputDirectory.exists()) {
            boolean success = outputDirectory.mkdir();
            if (success)
                Toast.makeText(this, "Created AVDataCapture dir", Toast.LENGTH_SHORT);
            else
                Toast.makeText(this, "Failed to create AVDataCapture dir", Toast.LENGTH_SHORT);
        }
    }

    // new button listener for the record button
    Button.OnClickListener myButtonOnClickListener
            = new Button.OnClickListener(){

        @Override
        public void onClick(View v) {
            // TODO Auto-generated method stub

            try{
                if(recording){
                    // if already in recording mode - stop recording

                    // remove callbacks
                    angleHandler.removeCallbacks(angleRunnable);

                    // stop recording and release camera
                    mediaRecorder.stop();  // stop the recording
                    releaseMediaRecorder(); // release the MediaRecorder object

                    // reset every back
                    recordButton.setText("REC");
                    recording = false;
                    angle.setVisibility(View.INVISIBLE);
                    time.setVisibility(View.INVISIBLE);

                    // save the angles to the CSV file
                    CSVUtility.writeToCSV(OUTPUT_DIRECTORY + "AVDataCapture_data.csv", outputAngles);
                    Toast.makeText(VideoCapture.this, "Saved", Toast.LENGTH_LONG).show();
                    outputAngles.clear();

                    // scan the directory - bug in android
                    MediaScannerConnection.scanFile(VideoCapture.this,
                        new String[] {
                                OUTPUT_DIRECTORY + "AVDataCapture_video.mp4",
                                OUTPUT_DIRECTORY + "AVDataCapture_data.csv"
                        }, null, null);

                    offsetCalculated = false;
                }
                else{
                    // begin recording

                    // release Camera before MediaRecorder start
                    releaseCamera();

                    // failed to prepare video recorder
                    if(!prepareMediaRecorder()){
                        Toast.makeText(VideoCapture.this,
                                "Fail in prepareMediaRecorder()!\n - Ended -",
                                Toast.LENGTH_LONG).show();
                        finish();
                    }

                    // start the video recording
                    mediaRecorder.start();
                    recording = true;
                    recordButton.setText("STOP");
                    angle.setVisibility(View.VISIBLE);
                    startTime = SystemClock.uptimeMillis();
                    time.setVisibility(View.VISIBLE);

                    // start the angle handler every 1 second
                    angleHandler.postDelayed(angleRunnable, 1000);
                }
            }catch (Exception ex){
                ex.printStackTrace();
            }
        }};

    /**
     * Get the camera instance
     * @return
     */
    private Camera getCameraInstance(){
        // TODO Auto-generated method stub
        Camera c = null;
        try {
            c = Camera.open(); // attempt to get a Camera instance
        }
        catch (Exception e){
            // Camera is not available (in use or does not exist)
        }
        return c; // returns null if camera is unavailable
    }


    /**
     * Preparing the camera and media recorder
     * @return
     */
    private boolean prepareMediaRecorder(){
        // create instances of the camera and media recorder
        myCamera = getCameraInstance();
        mediaRecorder = new MediaRecorder();

        // setup the media recorder
        myCamera.unlock();
        mediaRecorder.setCamera(myCamera);
        mediaRecorder.setAudioSource(MediaRecorder.AudioSource.CAMCORDER);
        mediaRecorder.setVideoSource(MediaRecorder.VideoSource.CAMERA);
        // QUALITY_720P = 30 FPS
        mediaRecorder.setProfile(CamcorderProfile.get(CamcorderProfile.QUALITY_720P));
        mediaRecorder.setOutputFile(OUTPUT_DIRECTORY + "AVDataCapture_video.mp4");
        mediaRecorder.setPreviewDisplay(myCameraSurfaceView.getHolder().getSurface());
        mediaRecorder.setVideoFrameRate(APS);

        // prepare for recording video
        try {
            mediaRecorder.prepare();
        } catch (IllegalStateException e) {
            releaseMediaRecorder();
            return false;
        } catch (IOException e) {
            releaseMediaRecorder();
            return false;
        }
        return true;

    }

    /**
     * Function ran on pause
     * Release the recorder and camera
     */
    @Override
    protected void onPause() {
        super.onPause();
        releaseMediaRecorder();       // if you are using MediaRecorder, release it first
        releaseCamera();              // release the camera immediately on pause event
    }

    /**
     * Function to release the media recorder
     */
    private void releaseMediaRecorder(){
        if (mediaRecorder != null) {
            mediaRecorder.reset();   // clear recorder configuration
            mediaRecorder.release(); // release the recorder object
            mediaRecorder = new MediaRecorder();
            myCamera.lock();           // lock camera for later use
        }
    }

    /**
     * Function to release the camera
     */
    private void releaseCamera(){
        if (myCamera != null){
            myCamera.release();        // release the camera for other applications
            myCamera = null;
        }
    }

    public class MyCameraSurfaceView extends SurfaceView implements SurfaceHolder.Callback{
        /**
         *
         */

        private SurfaceHolder mHolder;
        private Camera mCamera;

        /**
         *
         * @param context
         * @param camera
         */
        public MyCameraSurfaceView(Context context, Camera camera) {
            super(context);
            mCamera = camera;

            // Install a SurfaceHolder.Callback so we get notified when the
            // underlying surface is created and destroyed.
            mHolder = getHolder();
            mHolder.addCallback(this);
            // deprecated setting, but required on Android versions prior to 3.0
            mHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);
        }

        /**
         *
         * @param holder
         * @param format
         * @param weight
         * @param height
         */
        @Override
        public void surfaceChanged(SurfaceHolder holder, int format, int weight,
                                   int height) {
            // If your preview can change or rotate, take care of those events here.
            // Make sure to stop the preview before resizing or reformatting it.

            if (mHolder.getSurface() == null){
                // preview surface does not exist
                return;
            }

            // stop preview before making changes
            try {
                mCamera.stopPreview();
            } catch (Exception e){
                // ignore: tried to stop a non-existent preview
            }

            // make any resize, rotate or reformatting changes here

            // start preview with new settings
            try {
                mCamera.setPreviewDisplay(mHolder);
                mCamera.startPreview();

            } catch (Exception e){
            }
        }

        /**
         *
         * @param holder
         */
        @Override
        public void surfaceCreated(SurfaceHolder holder) {
            // TODO Auto-generated method stub
            // The Surface has been created, now tell the camera where to draw the preview.
            try {
                mCamera.setPreviewDisplay(holder);
                mCamera.startPreview();
            } catch (IOException e) {
            }
        }

        /**
         *
         * @param holder
         */
        @Override
        public void surfaceDestroyed(SurfaceHolder holder) {
            // TODO Auto-generated method stub

        }
    }

    /**
     * Function ran when there is a change in the sensor i.e. new reading
     * @param event game rotation sensor event
     */
    @Override
    public void onSensorChanged(SensorEvent event) {
        // if in capture mode
        if (recording) {

            // if the angle offset has not been calculated
            if (!offsetCalculated) {
                // calculate the offset
                getReadings(event);

                offset = readings[0];
                offsetCalculated = true;
                Toast.makeText(this, "Offset: " + offset, Toast.LENGTH_LONG).show();
            }

            // get the readings from the sensor
            getReadings(event);

            // get the angle
            reading = readings[0];

            // make adjustments to the angle using the calculated offset
            if (reading - offset < -180) {
                reading = (180 - offset) + (180 - Math.abs(reading));
            }
            else if (reading - offset > 180) {
                reading = (-180 - offset) - (180 - reading);
            }
            else {
                reading = reading - offset;
            }

            // set the angle text
            angle.setText(String.format(getResources().getString(R.string.value_format), reading));

            // add the read angle to the array list
            angles.add(reading);
        }
    }

    /**
     * Function to run if the accuracy of the sensor changed
     * @param sensor
     * @param accuracy
     */
    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }

    /**
     * Function that gets readings from the sensor and converts them to degrees
     * @param event an event from the sensor
     */
    private void getReadings(SensorEvent event) {
        SensorManager.getRotationMatrixFromVector(rotationMatrix, event.values);
        SensorManager.getOrientation(rotationMatrix, readings);
        convertToDegrees(readings);
    }

    /**
     * Converts readings to degrees from radians
     * @param readings array of floats
     */
    private void convertToDegrees(float[] readings) {
        for (int i = 0; i < readings.length; i++) {
            readings[i] = (float) Math.toDegrees(readings[i]);
        }
    }
}