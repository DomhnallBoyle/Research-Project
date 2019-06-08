/*
 * Filename: MainActivity.java
 * Description: The entry point for the application
 * Author: Domhnall Boyle
 * Maintained by: Domhnall Boyle
 */

package com.example.domhnall.avdatacapture;

import android.content.Context;
import android.content.Intent;
import android.hardware.Sensor;
import android.hardware.SensorManager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import java.util.List;


public class MainActivity extends AppCompatActivity {
    /**
     * Activity ran on startup of the application
     */

    // UI elements
    private Button videoAngles, angles;
    private TextView sensorList;

    /**
     * Function to be ran on create of the application
     * @param savedInstanceState
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);


        setContentView(R.layout.activity_main);

        // link up the UI elements to their view elements
        videoAngles = (Button)findViewById(R.id.button_video_angles);
        angles = (Button)findViewById(R.id.button_angles);
        sensorList = (TextView)findViewById(R.id.sensor_list);

        // set a listener to the video & angles button
        videoAngles.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // change activities to collect both angles and video
                Intent myIntent = new Intent(MainActivity.this, VideoCapture.class);
                MainActivity.this.startActivity(myIntent);
            }
        });

        // set a listener to the angles button
        angles.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // change activities to collect the angles
                Intent myIntent = new Intent(MainActivity.this, AngleCapture.class);
                MainActivity.this.startActivity(myIntent);
            }
        });

        // display all the sensors available to the device
        this.displayAllSensors();
    }

    /**
     * Function that displays all the available sensors on the device
     * Needed to check compatibility of the devices
     */
    public void displayAllSensors() {
        // setup the sensor manager and get the sensor list
        SensorManager sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        List<Sensor> sensors = sensorManager.getSensorList(Sensor.TYPE_ALL);

        // set the text view to the sensor list as a string
        String sensor_string = "Sensors Available:\n";

        for (Sensor sensor: sensors) {
            sensor_string += sensor.getName() + "\n";
        }

        sensorList.setText(sensor_string);
    }
}
