/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.classification;

import android.content.Intent;
import android.content.ServiceConnection;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;
import java.io.IOException;
import java.util.List;
import org.tensorflow.lite.examples.classification.env.BorderedText;
import org.tensorflow.lite.examples.classification.env.Logger;
import org.tensorflow.lite.examples.classification.tflite.Classifier;
import org.tensorflow.lite.examples.classification.tflite.Classifier.Device;
import org.tensorflow.lite.examples.classification.tflite.Classifier.Model;
import android.content.*;
import android.os.IBinder;

public class ClassifierActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();
  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
  private static final float TEXT_SIZE_DIP = 10;
  private Bitmap rgbFrameBitmap = null;
  private long lastProcessingTimeMs;
  private Integer sensorOrientation;
  private Classifier classifier;
  private BorderedText borderedText;
  /** Input image size of the model along x axis. */
  private int imageSizeX;
  /** Input image size of the model along y axis. */
  private int imageSizeY;

  private PodUsbSerialService mPodUsbSerialService = null;
  private Boolean mBounded = false;

  // Track throttle and height values for exponential averaging
  private float last_throttle = 0.0F;
  private float last_height = -1.0F;

  // Constants for drone movement
  private final float K_P = 0.1F;
  private final float THROTTLE_SCALE_FORWARD = 1.0F/25.0F;
  private final float THROTTLE_SCALE_BACKWARD = 1.0F;
  private final float THROTTLE_STEP_LIMIT = 0.02F;
  private final float THROTTLE_MIN = 0.1F;
  private final float THROTTLE_MAX = 0.3F;
  private final float THROTTLE_AVERAGE_ALPHA = 0.5F;
  private final float DEFAULT_THROTTLE_FORWARD = 0.1F;
  private final float DEFAULT_THROTTLE_BACKWARD = 0.2F;

  // Constants for distance measurement
  private final int MAX_HEIGHT_LIMIT = 105;
  private final int MIN_HEIGHT_LIMIT = 40;
  private final float HEIGHT_AVERAGE_ALPHA = 0.2F;


  @Override
  protected void onCreate(final Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
  }

  @Override
  public synchronized void onStart() {
    super.onStart();
    // start and bind service
    Intent mIntent = new Intent(this, PodUsbSerialService.class);
    startService(mIntent);
    bindService(mIntent, mConnection, BIND_AUTO_CREATE);

    if (mPodUsbSerialService != null) {
      mPodUsbSerialService.usbStartConnection();
    }
  }

  // get service instance
  private ServiceConnection mConnection = new ServiceConnection() {
    @Override
    public void onServiceConnected(ComponentName name, IBinder service) {
      mBounded = true;
      PodUsbSerialService.UsbBinder mUsbBinder = (PodUsbSerialService.UsbBinder) service;
      mPodUsbSerialService = mUsbBinder.getService();
    }

    @Override
    public void onServiceDisconnected(ComponentName name) {
      mBounded = false;
      mPodUsbSerialService = null;
    }
  };

  @Override
  protected int getLayoutId() {
    return R.layout.tfe_ic_camera_connection_fragment;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    recreateClassifier(getModel(), getDevice(), getNumThreads());
    if (classifier == null) {
      LOGGER.e("No classifier on preview!");
      return;
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
  }

  // Make sure throttle is between min and max allowed values
  private float getCalibratedThrottle(float throttle) {
    if (throttle < THROTTLE_MIN) {
      return THROTTLE_MIN;
    } else if (throttle > THROTTLE_MAX) {
      return THROTTLE_MAX;
    } else {
      return throttle;
    }
  }

  private void droneForward(float throttle) {
    float scaled_throttle = getCalibratedThrottle(throttle * THROTTLE_SCALE_FORWARD);
    moveDrone(scaled_throttle);
  }

  private void droneBackward(float throttle) {
    float scaled_throttle = getCalibratedThrottle(throttle * THROTTLE_SCALE_BACKWARD);
    moveDrone(-1 * scaled_throttle);
  }

  private void droneStop() {
    moveDrone(0.0F);
  }

  private void moveDrone(float target_throttle) {
    // Smooth throttle values using exponential averaging
    float averaged_throttle = THROTTLE_AVERAGE_ALPHA * target_throttle + (1 - THROTTLE_AVERAGE_ALPHA) * last_throttle;

    // Don't send duplicate packet if the throttle isn't sufficiently different from the last one sent
    if (Math.abs(averaged_throttle-last_throttle) < THROTTLE_STEP_LIMIT) return;

    // Send averaged throttle value to drone
    CommanderHoverPacket cp = new CommanderHoverPacket(averaged_throttle, 0F, 0F, 0.6F);
    mPodUsbSerialService.usbSendData(((CrtpPacket) cp).toByteArray());

    // Update last sent throttle value
    last_throttle = averaged_throttle;
  }

  @Override
  protected void processImage() {
    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    final int cropSize = Math.min(previewWidth, previewHeight);

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            if (classifier != null) {
              final long startTime = SystemClock.uptimeMillis();

              // Get emotion recognition results from captured image
              final List<Classifier.Recognition> results =
                  classifier.recognizeImage(rgbFrameBitmap, sensorOrientation);
              lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

              // If done is connected, move based on emotion results
              if (mPodUsbSerialService != null && results.size() != 0) {
                float captured_height = results.get(0).getLocation().height();
                if (Float.isNaN(captured_height)) return;

                // Remove noise from height measurement using exponential averaging
                float smoothed_height = (last_height == -1.0) ?
                        captured_height :
                        (HEIGHT_AVERAGE_ALPHA * captured_height + (1 - HEIGHT_AVERAGE_ALPHA) * last_height);
                last_height = smoothed_height;

                // Send drone motion command based on recognized emotion and current distance
                if (results.size() > 0 && results.get(0).getId() == "Happy" && smoothed_height < MAX_HEIGHT_LIMIT) {
                  float throttle = K_P * (MAX_HEIGHT_LIMIT - smoothed_height);
                  droneForward(throttle);
                } else if (results.size() > 0 && results.get(0).getId() == "Surprised" && smoothed_height > MIN_HEIGHT_LIMIT){
                  float throttle = K_P * (smoothed_height - MIN_HEIGHT_LIMIT);
                  droneBackward(throttle);
                } else if (smoothed_height != 0 && smoothed_height > MAX_HEIGHT_LIMIT) {
                  droneBackward(DEFAULT_THROTTLE_BACKWARD);
                } else if (smoothed_height != 0 && smoothed_height < MIN_HEIGHT_LIMIT) {
                  droneForward(DEFAULT_THROTTLE_FORWARD);
                } else {
                  droneStop();
                }
              }

              runOnUiThread(
                  new Runnable() {
                    @Override
                    public void run() {
                      showResultsInBottomSheet(results);
                      showFrameInfo(previewWidth + "x" + previewHeight);
                      showCropInfo(imageSizeX + "x" + imageSizeY);
                      showCameraResolution(cropSize + "x" + cropSize);
                      showRotationInfo(String.valueOf(sensorOrientation));
                      showInference(lastProcessingTimeMs + "ms");
                    }
                  });
            }
            readyForNextImage();
          }
        });
  }

  @Override
  protected void onInferenceConfigurationChanged() {
    if (rgbFrameBitmap == null) {
      // Defer creation until we're getting camera frames.
      return;
    }
    final Device device = getDevice();
    final Model model = getModel();
    final int numThreads = getNumThreads();
    runInBackground(() -> recreateClassifier(model, device, numThreads));
  }

  private void recreateClassifier(Model model, Device device, int numThreads) {
    if (classifier != null) {
      LOGGER.d("Closing classifier.");
      classifier.close();
      classifier = null;
    }
    if (device == Device.GPU
        && (model == Model.QUANTIZED_MOBILENET || model == Model.QUANTIZED_EFFICIENTNET)) {
      LOGGER.d("Not creating classifier: GPU doesn't support quantized models.");
      runOnUiThread(
          () -> {
            Toast.makeText(this, R.string.tfe_ic_gpu_quant_error, Toast.LENGTH_LONG).show();
          });
      return;
    }
    try {
      LOGGER.d(
          "Creating classifier (model=%s, device=%s, numThreads=%d)", model, device, numThreads);
      classifier = Classifier.create(this, model, device, numThreads);
    } catch (IOException | IllegalArgumentException e) {
      LOGGER.e(e, "Failed to create classifier.");
      runOnUiThread(
          () -> {
            Toast.makeText(this, e.getMessage(), Toast.LENGTH_LONG).show();
          });
      return;
    }

    // Updates the input image size.
    imageSizeX = classifier.getImageSizeX();
    imageSizeY = classifier.getImageSizeY();
  }
}
