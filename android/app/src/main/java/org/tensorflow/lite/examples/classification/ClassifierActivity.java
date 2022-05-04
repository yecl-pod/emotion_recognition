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

  private double k_p = 0.1;
  private double k_d = 0.01;
  private double near_error = 0.0;
  private double far_error = 0.0;
  private double throttle = 0.0;
  private double average_throttle = 0.0;
  private long last_time = SystemClock.uptimeMillis();
  private double average_width = -1.0;
  private int near_width_limit = 95;
  private int far_width_limit = 90;
  private int count = 0;

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

    if (mPodUsbSerialService == null) {
      LOGGER.e("CARTEST Service is null");
    }
    else {
      LOGGER.e("CARTEST Service is NOT null");
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
      LOGGER.e("CARTEST Setting up service");
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

  private void carForward(double throttle) {
    float calibrated_throttle = (float) throttle/30;
    if (calibrated_throttle < 0.1) {
      calibrated_throttle = 0.1F;
    }
    else if (calibrated_throttle > 0.3) {
      calibrated_throttle = 0.3F;
    }

    LOGGER.e("CARTEST forwards: "+calibrated_throttle);
    moveCar(calibrated_throttle);
  }

  private void carBackward(double throttle) {
    float calibrated_throttle = (float) throttle/30;
    if (calibrated_throttle < 0.1) {
      calibrated_throttle = 0.1F;
    }
    else if (calibrated_throttle > 0.3) {
      calibrated_throttle = 0.3F;
    }

    LOGGER.e("CARTEST backwards: "+-1*calibrated_throttle);
    moveCar(-1*calibrated_throttle);
  }

  private void moveCar(double new_throttle) {
    float curr_throttle = (float) (0.2 * new_throttle + 0.8 * average_throttle);
    LOGGER.e("CARTEST FINAL THROTTLE: " + curr_throttle);
    average_throttle = curr_throttle;
    CommanderHoverPacket cp = new CommanderHoverPacket(curr_throttle, 0F, 0F, 0.6F);
    mPodUsbSerialService.usbSendData(((CrtpPacket) cp).toByteArray());
  }

  private void carStop() {
    CommanderHoverPacket cp = new CommanderHoverPacket(0F, 0F, 0F, 0.6F);
    mPodUsbSerialService.usbSendData(((CrtpPacket) cp).toByteArray());
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
              long time_difference = (long) ( (startTime - last_time) / 1000.0);
              final List<Classifier.Recognition> results =
                  classifier.recognizeImage(rgbFrameBitmap, sensorOrientation);
              lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

              // MOVE CAR
              if (mPodUsbSerialService != null && results.size() != 0) {
                float new_width = results.get(0).getLocation().height();
                float width = (float)0.0;
                if (average_width == -1.0) {
                  width = new_width;
                } else {
                  width = (float) (0.8 * average_width + 0.2 * new_width);
                  average_width = width;
                }
                LOGGER.e("CARTEST Width: "+width);


                if (results.get(0).getId() == "Smiling" && width < near_width_limit) {
                  double d_term = ((near_width_limit - width) - near_error) / time_difference;
//                  LOGGER.e("CARTEST d-term: "+d_term);
                  throttle = k_p * (near_width_limit - width) + k_d * d_term;
                  near_error = (near_width_limit - width);
                  far_error = 0.0;

                  carForward(throttle);
                } else if (results.get(0).getId() == "Not Smiling" && width > far_width_limit){
                  double d_term = ((width - far_width_limit) - far_error) / time_difference;
//                  LOGGER.e("CARTEST d-term: "+d_term);
                  throttle = k_p * (width - far_width_limit) + k_d * d_term;
                  far_error = (width - far_width_limit);
                  near_error = 0.0;

                  carBackward(throttle);
                }
                else {
                  carStop();
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
            count++;
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
