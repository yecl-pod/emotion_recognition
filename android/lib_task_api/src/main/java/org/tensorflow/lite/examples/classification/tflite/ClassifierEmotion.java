package org.tensorflow.lite.examples.classification.tflite;

import android.app.Activity;
import java.io.IOException;

/** This TensorFlowLite classifier works with the Emotion model. */
public class ClassifierEmotion extends Classifier {

    /**
     * Initializes a {@code ClassifierEmotion}.
     *
     * @param device a {@link Device} object to configure the hardware accelerator
     * @param numThreads the number of threads during the inference
     * @throws IOException if the model is not loaded correctly
     */
    public ClassifierEmotion(Activity activity, Device device, int numThreads)
            throws IOException {
        super(activity, device, numThreads);
    }

    @Override
    protected String getModelPath() {
        return "compressed_model.tflite";
    }
}