package com.example.client;

public class LabelIndex {

        static final String [] labels = {
                "None", "Mild", "Moderate","Moderately Severe", "Severe"
        };

        //Convert categorical label to numerical value
        static int labelToNum(String label) {
            int index;

            switch (label) {
                case "None":
                    index = 0;
                    break;
                case "Mild":
                    index = 1;
                    break;
                case "Moderate":
                    index = 2;
                    break;
                case "Moderately Severe":
                    index = 3;
                    break;
                case "Severe":
                    index = 4;
                    break;
                default:
                    index = -1;
            }

            return index;
        }
    }