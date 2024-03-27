package org.example;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TString;

import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        String modelDirPath = "src/main/resources/model/saved_model";

        // Inicjalizacja skanera do odczytu wejścia od użytkownika

        try (Scanner scanner = new Scanner(System.in); SavedModelBundle model = SavedModelBundle.load(modelDirPath, "serve")) {
            System.out.println("Model loaded successfully.");

            // Tworzenie sesji z załadowanego modelu
            try (Session session = model.session()) {

                System.out.println("Type a movie review (or 'quit' to exit):");

                String review;
                while (!(review = scanner.nextLine()).equalsIgnoreCase("quit")) {

                    // Tworzenie tensora wejściowego z recenzji
                    try (TString inputTensor = TString.tensorOf(TString.vectorOf(review))) {

                        // Wykonanie modelu i pobranie tensora wyjściowego
                        try (Tensor rawOutputTensor = session.runner()
                                .feed("serving_default_keras_layer_1_input:0", inputTensor)
                                .fetch("StatefulPartitionedCall_1:0")
                                .run()
                                .get(0)) {

                            // Rzutowanie tensora do TFloat32, co jest typem tensora zwracanego przez model
                            try (TFloat32 outputTensor = (TFloat32) rawOutputTensor) {
                                // Kopiowanie danych z tensora do tablicy jednowymiarowej, a następnie dostęp do pierwszego elementu
                                FloatNdArray array = NdArrays.ofFloats(Shape.of(1, 1));
                                outputTensor.copyTo(array);

                                float logit = array.getFloat(0, 0);
                                float probability = sigmoid(logit);
                                System.out.println("Probability (positive review): " + probability);
                            }
                        }
                    }
                    System.out.println("Type another movie review (or 'quit' to exit):");
                }
            }
        } catch (Exception e) {
            System.err.println("Failed to execute model: " + e.getMessage());
        }
    }

    private static float sigmoid(float x) {
        return (float) (1 / (1 + Math.exp(-x)));
    }
}