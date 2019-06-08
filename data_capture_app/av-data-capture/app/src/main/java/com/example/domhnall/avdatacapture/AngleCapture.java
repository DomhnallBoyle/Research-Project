/*
 * Filename: AngleCapture.java
 * Description: Used to capture angles from the game rotation sensors
 * Author: Domhnall Boyle
 * Maintained by: Domhnall Boyle
 */

package com.example.domhnall.avdatacapture;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Handler;
import android.os.SystemClock;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.RelativeLayout;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import java.util.ArrayList;


public class AngleCapture extends AppCompatActivity implements SensorEventListener {
    /**
     * Activity class that captures angles using the game rotation sensor and saves them to .CSV file
     */

    // constants
    private static final int DEFAULT_APS = 10;
    private static final String OUTPUT_DIRECTORY = "/storage/emulated/0/";

    // view stuff
    private RelativeLayout relativeLayout;
    private TextView angle, time;
    private Button startButton;
    private Spinner aps_selector;

    // sensor stuff
    private SensorManager sensorManager;
    private Sensor grvSensor;
    private float[] rotationMatrix = new float[9];
    private float[] readings = new float[3];
    private float offset, reading;
    private boolean offsetCalculated = false, capturing = false;

    // handler stuff
    private ArrayList<Float> angles;
    private Handler timeHandler, angleHandler;
    private Runnable angleRunnable;

    // time stuff
    private String currentTime;
    private long msTime, startTime, timeBuff, updateTime = 0L ;
    private int seconds, minutes, ms, angles_per_second;

    private CSVUtility csvUtility;

    /**
     * Function ran during on create of the activity
     * @param savedInstanceState
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_angle_capture);

        // linking the layout to the view
        relativeLayout = (RelativeLayout)findViewById(R.id.angles_capture_layout);

        // setting up the sensors
        sensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        grvSensor = (Sensor) sensorManager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR);
        sensorManager.registerListener(this, grvSensor, SensorManager.SENSOR_DELAY_NORMAL);

        // holds the list of angles
        angles = new ArrayList<Float>();

        // setting up a drop down list for the captured Angles Per Second
        // default is 10
        aps_selector = (Spinner) findViewById(R.id.aps_selector);
        String[] items = new String[]{"10", "15", "20", "25", "30"};
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, items);
        aps_selector.setAdapter(adapter);
        aps_selector.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener(){
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                angles_per_second = Integer.parseInt((String) parent.getItemAtPosition(position));
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                // TODO Auto-generated method stub
            }
        });
        angles_per_second = DEFAULT_APS;

        // instantiate the CSV utility for saving the outupt
        csvUtility = new CSVUtility(OUTPUT_DIRECTORY + "AVDataCapture_data.csv", new String[]{"Angle_degrees", "Angle_radians", "Time"});

        // linking the angle label to the view and adding a handler
        angle = (TextView)findViewById(R.id.textview_capture_angles_angle);
        angleHandler = new Handler();
        angleRunnable = new Runnable() {
            public void run() {
                // handler ran every 1 second
                // if there are enough collected angles
                if (angles.size() >= angles_per_second) {
                    // jumps in the angles
                    int displacement = angles.size() / angles_per_second;
                    // loop over the angles, using the displacement to jump
                    for (int i = 0; i < angles.size() - displacement; i += displacement) {
                        if (capturing) {
                            // write the angles to csv if in capturing mode
                            csvUtility.writeToCSV(
                                    new String[]{
                                            String.format("%.2f", angles.get(i)),
                                            String.format("%.2f", Math.toRadians(angles.get(i))),
                                            String.format("%d:%02d", minutes, seconds)
                                    }
                            );
                        }
                    }

                    angles.clear();
                }
                angleHandler.postDelayed(angleRunnable, 1000);
            }
        };

        // linking the time label to the view
        time = (TextView)findViewById(R.id.textview_capture_angles_time);

        // create a time handler for setting the time on the label
        timeHandler = new Handler();
        // function to be ran constantly
        timeHandler.postDelayed(new Runnable() {
            public void run() {
                // get the elapsed time
                msTime = SystemClock.uptimeMillis() - startTime;
                updateTime = timeBuff + msTime;
                seconds = (int) (updateTime / 1000);
                minutes = seconds / 60;
                seconds = seconds % 60;
                ms = (int) (updateTime % 1000);

                // update the label if in capturing mode
                currentTime = String.format("%d:%02d:%03d", minutes, seconds, ms);

                if (capturing)
                    time.setText("Time: " + currentTime);

                // run the method again
                timeHandler.postDelayed(this, 0);
            }
        }, 0);

        // link the start button to the view and add the click listener
        startButton = (Button)findViewById(R.id.button_capture_angles_start);
        startButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (capturing) {
                    // STOPPING
                    angleHandler.removeCallbacks(angleRunnable);

                    capturing = false;
                    offsetCalculated = false;

                    // UI changes
                    angle.setVisibility(View.INVISIBLE);
                    startButton.setText("START");

                    csvUtility.close();

                    aps_selector.setVisibility(View.VISIBLE);
                }
                else {
                    // STARTING
                    csvUtility.open();

                    capturing = true;

                    aps_selector.setVisibility(View.INVISIBLE);

                    // reset timer
                    startTime = SystemClock.uptimeMillis();

                    // UI changes
                    angle.setVisibility(View.VISIBLE);
                    time.setVisibility(View.VISIBLE);
                    startButton.setText("STOP");

                    // start angle timer
                    angleHandler.postDelayed(angleRunnable, 1000);
                }
            }
        });

        // click listener for the relative layout
        relativeLayout.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (capturing) {
                    // reset the coordinate system
                    offsetCalculated = false;
                }
            }
        });

        // initially set the angle and time labels to invisible before starting
        angle.setVisibility(View.INVISIBLE);
        time.setVisibility(View.INVISIBLE);
    }

    /**
     * Function ran on pause of the app
     * Unregister listeners before pausing the activity
     */
    @Override
    public void onPause() {
        super.onPause();
        Log.i("APP", "Pausing");

        sensorManager.unregisterListener(this);
    }

    /**
     * Function ran on stop of the app
     * Unregister listeners before stopping the activity
     */
    @Override
    public void onStop() {
        super.onStop();
        Log.i("APP", "Stopping");

        sensorManager.unregisterListener(this);
        finish();
    }

    /**
     * Function ran on destroying of the app
     * Unregister listeners before destroying the activity
     */
    @Override
    public void onDestroy() {
        super.onDestroy();
        Log.i("APP", "Destroying");

        sensorManager.unregisterListener(this);
    }

    /**
     * Function ran when there is a change in the sensor i.e. new reading
     * @param event game rotation sensor event
     */
    @Override
    public void onSensorChanged(SensorEvent event) {
        // if in capture mode
        if (capturing) {

            // if the angle offset has not been calculated
            if (!offsetCalculated) {
                // calculate the offset
                getReadings(event);

                offset = readings[0];
                offsetCalculated = true;
                Toast.makeText(this, "Offset: " + offset, Toast.LENGTH_SHORT).show();
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
            System.out.println(reading);
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
