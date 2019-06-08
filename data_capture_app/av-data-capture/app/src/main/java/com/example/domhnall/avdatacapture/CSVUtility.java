/*
 * Filename: CSVUtility.java
 * Description: Used to open and save to CSV files on the device
 * Author: Domhnall Boyle
 * Maintained by: Domhnall Boyle
 */

package com.example.domhnall.avdatacapture;

import com.opencsv.CSVWriter;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;

public class CSVUtility {
    /**
     * Utility for opening and writing to CSV files on the device
     */

    // CSV stuff
    private String csvPath;
    private String[] headers;
    private Writer writer;
    private CSVWriter csvWriter;

    /**
     * Constructor of the object
     * @param csvPath path of the CSV file on the device
     * @param headers the headers given on the first row of the device
     */
    public CSVUtility(String csvPath, String[] headers) {
        this.csvPath = csvPath;
        this.headers = headers;
    }

    /**
     * Write the line to CSV
     * @param line array of strings
     */
    public void writeToCSV(String[] line) {
        this.csvWriter.writeNext(line);
    }

    /**
     * Open the file on the device initially writing the CSV headers
     */
    public void open() {
        try {
            this.writer = new BufferedWriter(new FileWriter(csvPath));
            this.csvWriter = new CSVWriter(writer,
                    CSVWriter.DEFAULT_SEPARATOR,
                    CSVWriter.NO_QUOTE_CHARACTER,
                    CSVWriter.DEFAULT_ESCAPE_CHARACTER,
                    CSVWriter.DEFAULT_LINE_END);

            this.csvWriter.writeNext(this.headers);
        }
        catch(IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Close the file when finished
     */
    public void close() {
        try {
            this.writer.close();
        }
        catch(IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Static method to write a line to the CSV file
     * @param csvPath path to the CSV file
     * @param angles An angle for each row
     */
    public static void writeToCSV(String csvPath, ArrayList<Float> angles) {
        try {
            // setup the writer
            Writer writer = new BufferedWriter(new FileWriter(csvPath));

            // setup the CSV writer
            CSVWriter csvWriter = new CSVWriter(writer, CSVWriter.DEFAULT_SEPARATOR,
                    CSVWriter.NO_QUOTE_CHARACTER,
                    CSVWriter.DEFAULT_ESCAPE_CHARACTER,
                    CSVWriter.DEFAULT_LINE_END);

            // create new string array list with a header
            ArrayList<String[]> contents = new ArrayList<String[]>();
            contents.add(new String[]{"Angle_degrees"});

            // add the angle as to this as a string
            for (Float angle: angles) {
                contents.add(new String[]{angle.toString()});
            }

            // write the rows
            csvWriter.writeAll(contents);

            // close the writer when finished
            writer.close();
        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }
}
