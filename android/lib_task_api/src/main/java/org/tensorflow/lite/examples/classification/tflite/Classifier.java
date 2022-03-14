/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.classification.tflite;

import static java.lang.Math.min;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.Rect;
import android.graphics.RectF;
import android.os.Trace;
import android.util.Log;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.List;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.label.Category;
import org.tensorflow.lite.support.metadata.MetadataExtractor;
import org.tensorflow.lite.task.core.BaseOptions;
import org.tensorflow.lite.task.core.vision.ImageProcessingOptions.Orientation;
import org.tensorflow.lite.task.vision.classifier.Classifications;

import androidx.annotation.NonNull;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

/** A classifier specialized to label images using TensorFlow Lite. */
public abstract class Classifier {
  public static final String TAG = "ClassifierWithTaskApi";

  /** The model type used for classification. */
  public enum Model {
    FLOAT_MOBILENET,
    QUANTIZED_MOBILENET,
    FLOAT_EFFICIENTNET,
    QUANTIZED_EFFICIENTNET,
    EMOTION
  }

  /** The runtime device type used for executing classification. */
  public enum Device {
    CPU,
    NNAPI,
    GPU
  }

  /** Number of results to show in the UI. */
  private static final int MAX_RESULTS = 3;

  /** Image size along the x axis. */
  private final int imageSizeX;

  /** Image size along the y axis. */
  private final int imageSizeY;
  /** An instance of the driver class to run model inference with Tensorflow Lite. */
  protected Interpreter imageClassifier = null;

  // to detect faces
  protected FaceDetector detector;
  public ArrayList<Recognition> recognitions = new ArrayList<>();

  MainActivity controller = new MainActivity();
  private int count = 0;

  /**
   * Creates a classifier with the provided configuration.
   *
   * @param activity The current Activity.
   * @param model The model to use for classification.
   * @param device The device to use for classification.
   * @param numThreads The number of threads to use for classification.
   * @return A classifier with the desired configuration.
   */
  public static Classifier create(Activity activity, Model model, Device device, int numThreads)
      throws IOException {
    if (model == Model.QUANTIZED_MOBILENET) {
      return new ClassifierQuantizedMobileNet(activity, device, numThreads);
    } else if (model == Model.FLOAT_MOBILENET) {
      return new ClassifierFloatMobileNet(activity, device, numThreads);
    } else if (model == Model.FLOAT_EFFICIENTNET) {
      return new ClassifierFloatEfficientNet(activity, device, numThreads);
    } else if (model == Model.QUANTIZED_EFFICIENTNET) {
      return new ClassifierQuantizedEfficientNet(activity, device, numThreads);
    } else if (model == Model.EMOTION) {
      return new ClassifierEmotion(activity, device, numThreads);
    } else {
      throw new UnsupportedOperationException();
    }
  }

  /** An immutable result returned by a Classifier describing what was recognized. */
  public static class Recognition {
    /**
     * A unique identifier for what has been recognized. Specific to the class, not the instance of
     * the object.
     */
    private final String id;

    /** Display name for the recognition. */
    private final String title;

    /**
     * A sortable score for how good the recognition is relative to others. Higher should be better.
     */
    private final Float confidence;

    /** Optional location within the source image for the location of the recognized object. */
    private RectF location;

    public Recognition(
        final String id, final String title, final Float confidence, final RectF location) {
      this.id = id;
      this.title = title;
      this.confidence = confidence;
      this.location = location;
    }

    public String getId() {
      return id;
    }

    public String getTitle() {
      return title;
    }

    public Float getConfidence() {
      return confidence;
    }

    public RectF getLocation() {
      return new RectF(location);
    }

    public void setLocation(RectF location) {
      this.location = location;
    }

    @Override
    public String toString() {
      String resultString = "";
      if (id != null) {
        resultString += "[" + id + "] ";
      }

      if (title != null) {
        resultString += title + " ";
      }

      if (confidence != null) {
        resultString += String.format("(%.1f%%) ", confidence * 100.0f);
      }

      if (location != null) {
        resultString += location + " ";
      }

      return resultString.trim();
    }
  }

  /** Initializes a {@code Classifier}. */
  protected Classifier(Activity activity, Device device, int numThreads) throws IOException {
    BaseOptions.Builder baseOptionsBuilder = BaseOptions.builder();
    switch (device) {
      case GPU:
        baseOptionsBuilder.useGpu();
        break;
      case NNAPI:
        baseOptionsBuilder.useNnapi();
        break;
      default:
        break;
    }

    if (getModelPath() == "compressed_model.tflite") {
      MappedByteBuffer tfliteModel = FileUtil.loadMappedFile(activity, getModelPath());
      imageClassifier = new Interpreter(tfliteModel);

      // initialize face detector to detect smiles
      FaceDetectorOptions classification_options =
              new FaceDetectorOptions.Builder()
                      .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
                      .build();
      detector = FaceDetection.getClient(classification_options);
    }

    Log.d(TAG, "Created a Tensorflow Lite Image Classifier and detector.");

    // Get the input image size information of the underlying tflite model.
    MappedByteBuffer tfliteModel = FileUtil.loadMappedFile(activity, getModelPath());
    MetadataExtractor metadataExtractor = new MetadataExtractor(tfliteModel);
    // Image shape is in the format of {1, height, width, 3}.
    int[] imageShape = metadataExtractor.getInputTensorShape(/*inputIndex=*/ 0);
    imageSizeY = imageShape[1];
    imageSizeX = imageShape[2];

    controller.init();
  }

  public static Bitmap scaleDown(Bitmap realImage, float maxImageSize,
                                 boolean filter) {
    float ratio = Math.min(
            (float) maxImageSize / realImage.getWidth(),
            (float) maxImageSize / realImage.getHeight());
    int width = Math.round((float) ratio * realImage.getWidth());
    int height = Math.round((float) ratio * realImage.getHeight());

    Bitmap newBitmap = Bitmap.createScaledBitmap(realImage, width,
            height, filter);
    return newBitmap;
  }

  /** Runs inference and returns the classification results. */
  public List<Recognition> recognizeImage(Bitmap bitmap, int sensorOrientation) {
    // Logs this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

//    TensorImage inputImage = TensorImage.fromBitmap(bitmap);
    InputImage image = InputImage.fromBitmap(bitmap, sensorOrientation);

    // run smile detection on input image
    Task<List<Face>> result =
            detector.process(image)
                    .addOnSuccessListener(
                            new OnSuccessListener<List<Face>>() {
                              @Override
                              public void onSuccess(List<Face> faces) {
                                for (Face face : faces) {
                                  Rect bounds = face.getBoundingBox();
                                  float rotY = face.getHeadEulerAngleY();  // Head is rotated to the right rotY degrees
                                  float rotZ = face.getHeadEulerAngleZ();  // Head is tilted sideways rotZ degrees

                                  // If classification was enabled:
                                  if (face.getSmilingProbability() != null) {
                                    float smileProb = face.getSmilingProbability();
                                    reportResults(smileProb);
                                  }
                                }
                              }
                            })
                    .addOnFailureListener(
                            new OnFailureListener() {
                              @Override
                              public void onFailure(@NonNull Exception e) {
                                // Task failed with an exception
                                // ...
                                Log.e(TAG, "Failed to detect face!");
                              }
                            });

//    bitmap = scaleDown(bitmap, 48, true);
//    int width = bitmap.getWidth();
//    int height = bitmap.getHeight();
//    int cropSize = min(width, height);
//    // TODO(b/169379396): investigate the impact of the resize algorithm on accuracy.
//    // Task Library resize the images using bilinear interpolation, which is slightly different from
//    // the nearest neighbor sampling algorithm used in lib_support. See
//    // https://github.com/tensorflow/examples/blob/0ef3d93e2af95d325c70ef3bcbbd6844d0631e07/lite/examples/image_classification/android/lib_support/src/main/java/org/tensorflow/lite/examples/classification/tflite/Classifier.java#L310.
//    ImageProcessingOptions imageOptions =
//        ImageProcessingOptions.builder()
//            .setOrientation(getOrientation(sensorOrientation))
//            // Set the ROI to the center of the image.
//            .setRoi(
//                new Rect(
//                    /*left=*/ (width - cropSize) / 2,
//                    /*top=*/ (height - cropSize) / 2,
//                    /*right=*/ (width + cropSize) / 2,
//                    /*bottom=*/ (height + cropSize) / 2))
//            .build();
//
//    // Runs the inference call.
//    Trace.beginSection("runInference");
//    long startTimeForReference = SystemClock.uptimeMillis();
//
//    float[][] output=new float[1][7];
//    int pixels[] = new int[width * height];
//    float grayscaleImg[][][][] = new float[1][48][48][1];
//
//    bitmap.getPixels(pixels,0, width, 0, 0, width, height);
//
//    for (int i = 0; i < width*height; i++) {
//      int x = i % width;
//      int y = i / width;
////      Log.e(TAG, "x: " + x + " y: " + y + " Red: " + Color.red(pixels[i]) + " Green: " + Color.green(pixels[i]) + " Blue: " + Color.blue(pixels[i]));
//      grayscaleImg[0][x][y][0] = (int) (Color.red(pixels[i]) * 0.299 + Color.green(pixels[i]) * 0.587 + Color.blue(pixels[i]) * 0.114);
//    }
//
//    String classes[] = {"angry", "disgusted", "afraid", "happy", "neutral", "sad", "surprised"};
//
////    Log.e(TAG, "Before image classifier");
//    imageClassifier.run(grayscaleImg, output);
//    String prediction = "";
//    int best_ind = 0;
//    float highest_score = 0;
//
//    for (int i = 0; i < 7; i++) {
//      if (output[0][i] > highest_score) {
//        best_ind = i;
//        highest_score = output[0][i];
//      }
//    }
//    prediction = classes[best_ind];
////    Log.e(TAG, "Pred: "+prediction);
//
//    long endTimeForReference = SystemClock.uptimeMillis();
//    Trace.endSection();
//    Log.v(TAG, "Timecost to run model inference: " + (endTimeForReference - startTimeForReference));
//    Log.v(TAG, "Model Output: " + output);
//
//    Trace.endSection();
//
//    final ArrayList<Recognition> recognitions = new ArrayList<>();
//    recognitions.add(new Recognition(prediction, prediction, highest_score, null));
//
    return recognitions;
  }

  public void reportResults(float smileProb) {
    recognitions = new ArrayList<>();

    if (smileProb > 0.5) {
      recognitions.add(new Recognition("Smiling", "Smiling", smileProb, null));
      if (count % 10 == 0) {
        controller.forward();
      }
    }
    else {
      recognitions.add(new Recognition("Not Smiling", "Not Smiling", 1-smileProb, null));
      if (count % 10 == 0) {
        controller.backward();
      }
    }
    count++;
  }


  /** Closes the interpreter and model to release resources. */
  public void close() {
    if (imageClassifier != null) {
      imageClassifier.close();
    }
  }

  /** Get the image size along the x axis. */
  public int getImageSizeX() {
    return imageSizeX;
  }

  /** Get the image size along the y axis. */
  public int getImageSizeY() {
    return imageSizeY;
  }

  /**
   * Converts a list of {@link Classifications} objects into a list of {@link Recognition} objects
   * to match the interface of other inference method, such as using the <a
   * href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android/lib_support">TFLite
   * Support Library.</a>.
   */
  private static List<Recognition> getRecognitions(List<Classifications> classifications) {

    final ArrayList<Recognition> recognitions = new ArrayList<>();
    // All the demo models are single head models. Get the first Classifications in the results.
    for (Category category : classifications.get(0).getCategories()) {
      recognitions.add(
          new Recognition(
              "" + category.getLabel(), category.getLabel(), category.getScore(), null));
    }
    return recognitions;
  }

  /* Convert the camera orientation in degree into {@link ImageProcessingOptions#Orientation}.*/
  private static Orientation getOrientation(int cameraOrientation) {
    switch (cameraOrientation / 90) {
      case 3:
        return Orientation.BOTTOM_LEFT;
      case 2:
        return Orientation.BOTTOM_RIGHT;
      case 1:
        return Orientation.TOP_RIGHT;
      default:
        return Orientation.TOP_LEFT;
    }
  }

  /** Gets the name of the model file stored in Assets. */
  protected abstract String getModelPath();
}
