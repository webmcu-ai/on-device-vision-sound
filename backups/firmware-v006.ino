// ======================================================
// XIAO ML KIT (OR XIAO ESP32S3 SENSE)
// FULL VISION ML  — v44
//
//
// Small Image collection, training, inference for education and proof of concept
//
// SD card stores: images in class folders
// SD card stores: headers in bin and .h text char array format
// Serial monitor and OLED output
// By Jeremy Ellis
// With free tier assistance from: Claude (code overview), ChatGPT (Critique), Gemini (Research) and Copilot (Alternate)
// Use at your own risk!
// MIT license
//
// Github Profile https://github.com/hpssjellis
// LinkedIn https://www.linkedin.com/in/jeremy-ellis-4237a9bb/
//
// For platformio you need the U8g2 library declared in the platformio.ini file and OPI PSRAM set
// lib_deps =  olikraus/U8g2 @ ^2.35.30
// ; Overriding defaults to enable OPI PSRAM
// build_flags = 
//    -DBOARD_HAS_PSRAM
//    -DARDUINO_USB_CDC_ON_BOOT=1
// board_build.arduino.memory_type = qio_opi
// board_build.flash_mode = qio
// board_upload.flash_size = 8MB
//


// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 0: CORE SYSTEM (ALWAYS INCLUDED)                                   ██
// ██  Headers, Defines, Pins, Globals, Memory, Weights, Setup, Loop           ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████


// optional Uncomment AFTER copying myVisWeights.h from SD to your sketch folder:
// Priority order: SD weights > baked-in weights > random He-init
//////////////////////////////////////IMPORTANT/////////////////////////////////////////////////
//#define USE_BAKED_WEIGHTS

#ifdef USE_BAKED_WEIGHTS
  #include "myVisWeights.h"
#endif

#include "esp_camera.h"
#include "img_converters.h"
#include "FS.h"
#include "SD.h"
#include "SPI.h"
#include <vector>
#include <algorithm>
#include <U8g2lib.h>
#include <Wire.h>

// ======================================================
// SHARED OLED OBJECT — declared ONCE here for both models.
// (Both original sketches declared their own copy of this; only one
// physical display/I2C bus exists, so only one object may exist.)
// ======================================================
U8G2_SSD1306_72X40_ER_1_HW_I2C u8g2(U8G2_R2, U8X8_PIN_NONE);

// ======================================================
// FORWARD DECLARATIONS
// Arduino IDE auto-generates these, but its ctags-based scanner
// can miss or mis-order them in a file this size (multi-line
// signatures, functions used before their textual definition
// across the Vision / Sound / Combined sections below). Declaring
// everything explicitly here removes that dependency entirely.
// ======================================================
float Visclip_value(float v, float mn, float mx);
float Visleaky_relu(float x);
float Visleaky_relu_deriv(float x);
int myVisReadTouch();
void myVisResetTouchState();
void myVisUpdateTouchState();
int myVisCheckTouchInput();
void myVisCheckTouchBackground();
int myVisPeekTouchAction();
void myVisAllocateMemory();
void myVisExportHeader();
bool myVisLoadWeights();
void myVisSaveWeights();
bool myVisLoadImageFromFile(const char* path, float* buf);
void myVisRenderRgbToOLED(int imageCount);
void myVisDisplayImageOnOLED(camera_fb_t* fb, int imageCount);
void myVisActionCollect(int classIdx);
void myVisForwardPass(float* input, float* logits);
void myVisBackwardDense(int label);
void myVisBackwardConv2();
void myVisBackwardPool1();
void myVisBackwardConv1();
void myVisAdamUpdate(float* w, float* g, float* m, float* v, int size, int step);
void myVisUpdateWeights(int step);
void myVisActionTrain();
void myVisActionInfer();
void myVisResetMenuState();
void myVisDrawMenu();
void myVisExecuteMenuItem(int idx);
void myVisHandleMenuNavigation();
float Sndclip_value(float v, float mn, float mx);
float Sndleaky_relu(float x);
float Sndleaky_relu_deriv(float x);
int mySndReadTouch();
void mySndResetTouchState();
void mySndUpdateTouchState();
int mySndCheckTouchInput();
void mySndCheckTouchBackground();
int mySndPeekTouchAction();
bool mySndSaveClipToSD(const String& path);
bool mySndLoadClipFromSD(const char* path);
static float mySndHzToMel(float hz);
static float mySndMelToHz(float mel);
void mySndBuildMelFilterbank();
void mySndBuildTwiddleTable();
static void mySndPowerSpectrum(const float* frame, float* power, int n);
bool mySndComputeSpectrogram();
bool mySndLoadClipAndComputeSpectrogram(const char* path);
void mySndAllocateMemory();
void mySndExportHeader();
bool mySndLoadWeights();
void mySndSaveWeights();
void mySndForwardPass(float* input, float* logits);
void mySndBackwardDense(int label);
void mySndBackwardConv2();
void mySndBackwardPool1();
void mySndBackwardConv1();
void mySndAdamUpdate(float* w, float* g, float* m, float* v, int size, int step, float gradScale);
void mySndUpdateWeights(int step, float gradScale);
void mySndResetMenuState();
void mySndDrawMenu();
void mySndExecuteMenuItem(int idx);
void mySndHandleMenuNavigation();
void mySndActionCollect(int classIdx);
void mySndActionTrain();
void mySndActionInfer();
int myReadTouch();
void myResetTouchState();
void myUpdateTouchState();
int myCheckTouchInput();
String myMenuLabel(int idx);
void myExecuteMenuItem(int idx);
void myDrawMenu();
void myResetMenuState();
void myHandleMenuNavigation();
void mySndInferTask(void* pvParameters);
void myActionCombinedInfer();


// ======================================================
// CONFIGURATION & ML HYPERPARAMETERS (MOVED UP)
// ======================================================


#define VIS_NUM_CLASSES 3

String myVisClassLabels[VIS_NUM_CLASSES] = {"0unknown", "1round", "2square"};

const int myVisTotalItems = VIS_NUM_CLASSES + 2;      // VIS_NUM_CLASSES + 2 for menu training and inference

float VIS_LEARNING_RATE = 0.0003;
int VIS_BATCH_SIZE = 6;
int VIS_TARGET_EPOCHS = 20;
int VALIDATION_IMAGES = 3;  // v44: last N images per class held out for validation (0 = disabled)

const int myVisThresholdPress = 1100;
const int myVisThresholdRelease = 900;
//const unsigned long myVisScreenTimeout = 300000; // not used presently





// ======================================================
// UNIFIED TOUCH INPUT SYSTEM - IMPROVED FOR COMPUTATION
// ======================================================
struct VisTouchState {
  bool isTouching = false;
  int tapCount = 0;
  unsigned long firstTapTime = 0;
  unsigned long lastReleaseTime = 0;
  unsigned long lastCheckTime = 0;  // NEW: track when we last checked
  const unsigned long tapWindow = 800;        // INCREASED from 450ms for slow contexts
  const int longPressTaps = 3;                // 3+ taps = long press
  const unsigned long debounceDelay = 50;     // debounce time
};


VisTouchState myVisTouch;








// SYSTEM LOGIC VARIABLES
unsigned long myVisLastActivityTime = 0; 
unsigned long myVisLastTapTime = 0;
const int myVisTapCooldown = 250;
int myVisMenuIndex = 1;
bool myVisIsSelected = false;
bool myVisWeightsTrained = false; 

// XIAO ESP32-S3 Camera Pins
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     10
#define SIOD_GPIO_NUM     40
#define SIOC_GPIO_NUM     39
#define Y9_GPIO_NUM       48
#define Y8_GPIO_NUM       11
#define Y7_GPIO_NUM       12
#define Y6_GPIO_NUM       14
#define Y5_GPIO_NUM       16
#define Y4_GPIO_NUM       18
#define Y3_GPIO_NUM       17
#define Y2_GPIO_NUM       15
#define VSYNC_GPIO_NUM    38
#define HREF_GPIO_NUM     47
#define PCLK_GPIO_NUM     13

// ======================================================
// CONFIGURABLE INPUT RESOLUTION
// ======================================================
#define VIS_INPUT_SIZE 64

// ======================================================
// CNN ARCHITECTURE CONSTANTS
// ======================================================
#define VIS_CONV1_KERNEL_SIZE 3
#define VIS_CONV1_FILTERS 4
#define VIS_CONV1_WEIGHTS (VIS_CONV1_KERNEL_SIZE * VIS_CONV1_KERNEL_SIZE * 3 * VIS_CONV1_FILTERS)

#define VIS_CONV2_KERNEL_SIZE 3
#define VIS_CONV2_FILTERS 8
#define VIS_CONV2_WEIGHTS (VIS_CONV2_KERNEL_SIZE * VIS_CONV2_KERNEL_SIZE * 4 * VIS_CONV2_FILTERS)

#define VIS_CONV1_OUTPUT_SIZE (VIS_INPUT_SIZE - 2)
#define VIS_POOL1_OUTPUT_SIZE (VIS_CONV1_OUTPUT_SIZE / 2)
#define VIS_CONV2_OUTPUT_SIZE (VIS_POOL1_OUTPUT_SIZE - 2)
#define VIS_FLATTENED_SIZE (VIS_CONV2_OUTPUT_SIZE * VIS_CONV2_OUTPUT_SIZE * VIS_CONV2_FILTERS)

#define VIS_OUTPUT_WEIGHTS (VIS_FLATTENED_SIZE * VIS_NUM_CLASSES)

// ======================================================
// GLOBAL VARIABLE DEFINITIONS
// ======================================================

// Add near line 150 with other global buffers:
uint8_t* myVisRgbBuffer = nullptr;  // Reusable RGB buffer for inference

bool myVisSDavailable = false;  // set true in setup() if SD mounts ok

// ML Buffers (PSRAM)
float* myVisInputBuffer = nullptr;
float* myVisConv1_w = nullptr;
float* myVisConv1_b = nullptr;
float* myVisConv2_w = nullptr;
float* myVisConv2_b = nullptr;
float* myVisOutput_w = nullptr;
float* myVisOutput_b = nullptr;

// Gradient buffers
float* myVisConv1_w_grad = nullptr;
float* myVisConv1_b_grad = nullptr;
float* myVisConv2_w_grad = nullptr;
float* myVisConv2_b_grad = nullptr;
float* myVisOutput_w_grad = nullptr;
float* myVisOutput_b_grad = nullptr;

// Adam optimizer momentum buffers
float* myVisConv1_w_m = nullptr;
float* myVisConv1_w_v = nullptr;
float* myVisConv1_b_m = nullptr;
float* myVisConv1_b_v = nullptr;
float* myVisConv2_w_m = nullptr;
float* myVisConv2_w_v = nullptr;
float* myVisConv2_b_m = nullptr;
float* myVisConv2_b_v = nullptr;
float* myVisOutput_w_m = nullptr;
float* myVisOutput_w_v = nullptr;
float* myVisOutput_b_m = nullptr;
float* myVisOutput_b_v = nullptr;

// Forward pass buffers
float* myVisConv1_output = nullptr;
float* myVisPool1_output = nullptr;
float* myVisConv2_output = nullptr;
float* myVisDense_output = nullptr;

// Backward pass buffers
float* myVisDense_grad = nullptr;
float* myVisConv2_grad = nullptr;
float* myVisPool1_grad = nullptr;
float* myVisConv1_grad = nullptr;

struct VisTrainingItem {
  String path;
  int label;
};
std::vector<VisTrainingItem> myVisTrainingData;

// ======================================================
// UTILITY FUNCTIONS
// ======================================================
inline float Visclip_value(float v, float mn=-100, float mx=100) {
  if(isnan(v)||isinf(v)) return 0;
  return constrain(v,mn,mx);
}

inline float Visleaky_relu(float x) { return x>0 ? x : 0.1f*x; }
inline float Visleaky_relu_deriv(float x) { return x>0 ? 1.0f : 0.1f; }

// ======================================================
// UNIFIED TOUCH INPUT FUNCTIONS - NEW!
// ======================================================
int myVisReadTouch() {
  int sum = 0;
  for (int i = 0; i < 3; i++) {
    sum += analogRead(A0);
    delayMicroseconds(100);
  }
  return sum / 3;
}

void myVisResetTouchState() {
  myVisTouch.isTouching = false;
  myVisTouch.tapCount = 0;
  myVisTouch.firstTapTime = 0;
  myVisTouch.lastReleaseTime = 0;
  myVisTouch.lastCheckTime = 0;
}

// NEW: Background touch monitor that can be called less frequently
void myVisUpdateTouchState() {
  unsigned long now = millis();
  
  // Only check every 20ms to avoid overwhelming analogRead
  if (now - myVisTouch.lastCheckTime < 20) return;
  myVisTouch.lastCheckTime = now;
  
  int val = myVisReadTouch();
  bool touchActive = myVisTouch.isTouching 
                      ? (val > myVisThresholdRelease) 
                      : (val > myVisThresholdPress);

  // Touch just started
  if (touchActive && !myVisTouch.isTouching) {
    if (now - myVisTouch.lastReleaseTime < myVisTouch.debounceDelay) {
      return; // Debounce
    }
    
    myVisTouch.isTouching = true;
    
    // First tap or within tap window?
    if (myVisTouch.tapCount == 0 || (now - myVisTouch.firstTapTime < myVisTouch.tapWindow)) {
      if (myVisTouch.tapCount == 0) {
        myVisTouch.firstTapTime = now;
      }
      myVisTouch.tapCount++;
      Serial.printf("Tap #%d\n", myVisTouch.tapCount);
    } else {
      // Window expired, reset
      myVisTouch.tapCount = 1;
      myVisTouch.firstTapTime = now;
      Serial.println("Tap #1 (new window)");
    }
  }
  
  // Touch released
  if (!touchActive && myVisTouch.isTouching) {
    myVisTouch.isTouching = false;
    myVisTouch.lastReleaseTime = now;
  }
}

// Returns: 0=no action, 1=tap, 2=long press (3+ taps)
// NOTE: Always call myVisUpdateTouchState() before this in tight loops
int myVisCheckTouchInput() {
  myVisUpdateTouchState();  // Update state first
  
  unsigned long now = millis();
  
  // Check if tap window expired and we have taps
  if (myVisTouch.tapCount > 0 && !myVisTouch.isTouching) {
    if (now - myVisTouch.firstTapTime > myVisTouch.tapWindow) {
      int result = (myVisTouch.tapCount >= myVisTouch.longPressTaps) ? 2 : 1;
      int count = myVisTouch.tapCount;
      myVisResetTouchState();
      
      if (result == 2) {
        Serial.printf("LONG PRESS detected (%d taps)\n", count);
      } else {
        Serial.printf("TAP detected (%d tap%s)\n", count, count > 1 ? "s" : "");
      }
      return result;
    }
  }
  
  return 0;
}

// NEW: Non-blocking check - just updates state without consuming events
// Use this in heavy computation loops
void myVisCheckTouchBackground() {
  myVisUpdateTouchState();
}

// NEW: Check if we have a pending action without consuming it
int myVisPeekTouchAction() {
  myVisUpdateTouchState();
  unsigned long now = millis();
  
  if (myVisTouch.tapCount > 0 && !myVisTouch.isTouching) {
    if (now - myVisTouch.firstTapTime > myVisTouch.tapWindow) {
      return (myVisTouch.tapCount >= myVisTouch.longPressTaps) ? 2 : 1;
    }
  }
  return 0;
}




// ======================================================
// MEMORY ALLOCATION
// ======================================================
void myVisAllocateMemory() {
  if (myVisInputBuffer != nullptr) return;
  
  Serial.println("\n=== Allocating Memory ===");
  
  myVisInputBuffer = (float*)ps_malloc(VIS_INPUT_SIZE * VIS_INPUT_SIZE * 3 * sizeof(float));
  myVisConv1_w = (float*)ps_malloc(VIS_CONV1_WEIGHTS * sizeof(float));
  myVisConv1_b = (float*)ps_malloc(VIS_CONV1_FILTERS * sizeof(float));
  myVisConv2_w = (float*)ps_malloc(VIS_CONV2_WEIGHTS * sizeof(float));
  myVisConv2_b = (float*)ps_malloc(VIS_CONV2_FILTERS * sizeof(float));
  myVisOutput_w = (float*)ps_malloc(VIS_OUTPUT_WEIGHTS * sizeof(float));
  myVisOutput_b = (float*)ps_malloc(VIS_NUM_CLASSES * sizeof(float));

  myVisConv1_w_grad = (float*)ps_malloc(VIS_CONV1_WEIGHTS * sizeof(float));
  myVisConv1_b_grad = (float*)ps_malloc(VIS_CONV1_FILTERS * sizeof(float));
  myVisConv2_w_grad = (float*)ps_malloc(VIS_CONV2_WEIGHTS * sizeof(float));
  myVisConv2_b_grad = (float*)ps_malloc(VIS_CONV2_FILTERS * sizeof(float));
  myVisOutput_w_grad = (float*)ps_malloc(VIS_OUTPUT_WEIGHTS * sizeof(float));
  myVisOutput_b_grad = (float*)ps_malloc(VIS_NUM_CLASSES * sizeof(float));

  myVisConv1_w_m = (float*)ps_calloc(VIS_CONV1_WEIGHTS, sizeof(float));
  myVisConv1_w_v = (float*)ps_calloc(VIS_CONV1_WEIGHTS, sizeof(float));
  myVisConv1_b_m = (float*)ps_calloc(VIS_CONV1_FILTERS, sizeof(float));
  myVisConv1_b_v = (float*)ps_calloc(VIS_CONV1_FILTERS, sizeof(float));
  myVisConv2_w_m = (float*)ps_calloc(VIS_CONV2_WEIGHTS, sizeof(float));
  myVisConv2_w_v = (float*)ps_calloc(VIS_CONV2_WEIGHTS, sizeof(float));
  myVisConv2_b_m = (float*)ps_calloc(VIS_CONV2_FILTERS, sizeof(float));
  myVisConv2_b_v = (float*)ps_calloc(VIS_CONV2_FILTERS, sizeof(float));
  myVisOutput_w_m = (float*)ps_calloc(VIS_OUTPUT_WEIGHTS, sizeof(float));
  myVisOutput_w_v = (float*)ps_calloc(VIS_OUTPUT_WEIGHTS, sizeof(float));
  myVisOutput_b_m = (float*)ps_calloc(VIS_NUM_CLASSES, sizeof(float));
  myVisOutput_b_v = (float*)ps_calloc(VIS_NUM_CLASSES, sizeof(float));

  myVisConv1_output = (float*)ps_malloc(VIS_CONV1_OUTPUT_SIZE*VIS_CONV1_OUTPUT_SIZE*VIS_CONV1_FILTERS*sizeof(float));
  myVisPool1_output = (float*)ps_malloc(VIS_POOL1_OUTPUT_SIZE*VIS_POOL1_OUTPUT_SIZE*VIS_CONV1_FILTERS*sizeof(float));
  myVisConv2_output = (float*)ps_malloc(VIS_CONV2_OUTPUT_SIZE*VIS_CONV2_OUTPUT_SIZE*VIS_CONV2_FILTERS*sizeof(float));
  myVisDense_output = (float*)ps_malloc(VIS_NUM_CLASSES*sizeof(float));

  myVisDense_grad = (float*)ps_malloc(VIS_FLATTENED_SIZE*sizeof(float));
  myVisConv2_grad = (float*)ps_malloc(VIS_CONV2_OUTPUT_SIZE*VIS_CONV2_OUTPUT_SIZE*VIS_CONV2_FILTERS*sizeof(float));
  myVisPool1_grad = (float*)ps_malloc(VIS_POOL1_OUTPUT_SIZE*VIS_POOL1_OUTPUT_SIZE*VIS_CONV1_FILTERS*sizeof(float));
  myVisConv1_grad = (float*)ps_malloc(VIS_CONV1_OUTPUT_SIZE*VIS_CONV1_OUTPUT_SIZE*VIS_CONV1_FILTERS*sizeof(float));

  if (!myVisInputBuffer || !myVisConv1_w || !myVisConv2_w || !myVisOutput_w || 
      !myVisConv1_output || !myVisPool1_output || !myVisConv2_output) {
    Serial.println("FATAL: PSRAM allocation failed!");
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "PSRAM ERROR!"); } while (u8g2.nextPage());
    while(1) { delay(1000); }
  }

  Serial.printf("Free PSRAM after allocation: %d bytes\n", ESP.getFreePsram());

  // Initialize weights with He initialization
  float c1std = sqrt(2.0/(9.0*3));
  for(int i=0; i<VIS_CONV1_WEIGHTS; i++) myVisConv1_w[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f * c1std;
  for(int i=0; i<VIS_CONV1_FILTERS; i++) myVisConv1_b[i] = 0;
  
  float c2std = sqrt(2.0/36.0);
  for(int i=0; i<VIS_CONV2_WEIGHTS; i++) myVisConv2_w[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f * c2std;
  for(int i=0; i<VIS_CONV2_FILTERS; i++) myVisConv2_b[i] = 0;
  
  float dstd = sqrt(2.0/VIS_FLATTENED_SIZE);
  for(int i=0; i<VIS_OUTPUT_WEIGHTS; i++) myVisOutput_w[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f * dstd;
  for(int i=0; i<VIS_NUM_CLASSES; i++) myVisOutput_b[i] = 0;
  Serial.println("He-init random weights set");
}

// ======================================================
// WEIGHT SAVE/LOAD
// ======================================================
void myVisExportHeader() {
  if (!myVisSDavailable) {
    Serial.println("No SD card - cannot export header");
    return;
  }
  if (!SD.exists("/header")) SD.mkdir("/header");
  File file = SD.open("/header/myVisWeights.h", FILE_WRITE);
  if (!file) return;
  file.println("#ifndef MY_MODEL_H\n#define MY_MODEL_H");
  file.println("// ======================================================");
  file.println("// IMPORTANT: After copying this file to your sketch folder,");
  file.println("// update BOTH of the following lines in your main sketch");
  file.println("// to match the number of classes and labels used during training:");
  file.println("//");
  file.printf( "//   #define VIS_NUM_CLASSES %d\n", VIS_NUM_CLASSES);

  file.print("//   String myVisClassLabels[VIS_NUM_CLASSES] = {");
  for (int i = 0; i < VIS_NUM_CLASSES; i++) {
    file.printf("\"%s\"", myVisClassLabels[i].c_str());
    if (i < VIS_NUM_CLASSES - 1) file.print(", ");
  }
  file.println("};");

 // file.println("//   String myVisClassLabels[VIS_NUM_CLASSES] = {\"0Blank\", \"1Cup\", \"2Pen\", ...};");
  file.println("//");
  file.println("// Then uncomment:  #define USE_BAKED_WEIGHTS");
  file.println("// ======================================================");
  auto myVisDump = [&](const char* name, float* data, int size) {
    file.printf("const float %s[] = { ", name);
    for(int i=0; i<size; i++) {
      file.print(data[i], 6); file.print("f");
      if(i < size-1) file.print(", ");
      if((i+1)%8 == 0) file.println();
    }
    file.println(" };");
  };
  myVisDump("myVisModel_conv1_w",  myVisConv1_w,  VIS_CONV1_WEIGHTS);
  myVisDump("myVisModel_conv1_b",  myVisConv1_b,  VIS_CONV1_FILTERS);
  myVisDump("myVisModel_conv2_w",  myVisConv2_w,  VIS_CONV2_WEIGHTS);
  myVisDump("myVisModel_conv2_b",  myVisConv2_b,  VIS_CONV2_FILTERS);
  myVisDump("myVisModel_output_w", myVisOutput_w, VIS_OUTPUT_WEIGHTS);
  myVisDump("myVisModel_output_b", myVisOutput_b, VIS_NUM_CLASSES);
  file.println("#endif");
  Serial.println("You can copy /header/myVisWeights.h to the sketch folder, then uncomment #define USE_BAKED_WEIGHTS");
  file.close();
}

bool myVisLoadWeights() {
  if (!myVisSDavailable) {
    Serial.println("No SD card - skipping weight load");
    return false;
  }
  if (!SD.exists("/header/myVisWeights.bin")) {
    Serial.println("No SD weights file found");
    return false;
  }
  Serial.println("Loading weights from SD...");
  File f = SD.open("/header/myVisWeights.bin", FILE_READ);
  if (!f) return false;
  f.read((uint8_t*)myVisConv1_w, VIS_CONV1_WEIGHTS*4); 
  f.read((uint8_t*)myVisConv1_b, VIS_CONV1_FILTERS*4);
  f.read((uint8_t*)myVisConv2_w, VIS_CONV2_WEIGHTS*4); 
  f.read((uint8_t*)myVisConv2_b, VIS_CONV2_FILTERS*4);
  f.read((uint8_t*)myVisOutput_w, VIS_OUTPUT_WEIGHTS*4); 
  f.read((uint8_t*)myVisOutput_b, VIS_NUM_CLASSES*4);
  f.close();
  Serial.println("Weights loaded successfully");
  myVisWeightsTrained = true;
  return true;
}

void myVisSaveWeights() {
  if (!myVisSDavailable) {
    Serial.println("No SD card - cannot save weights");
    return;
  }
  if (!SD.exists("/header")) SD.mkdir("/header");
  File f = SD.open("/header/myVisWeights.bin", FILE_WRITE);
  if (f) {
    f.write((uint8_t*)myVisConv1_w, VIS_CONV1_WEIGHTS*4); 
    f.write((uint8_t*)myVisConv1_b, VIS_CONV1_FILTERS*4);
    f.write((uint8_t*)myVisConv2_w, VIS_CONV2_WEIGHTS*4); 
    f.write((uint8_t*)myVisConv2_b, VIS_CONV2_FILTERS*4);
    f.write((uint8_t*)myVisOutput_w, VIS_OUTPUT_WEIGHTS*4); 
    f.write((uint8_t*)myVisOutput_b, VIS_NUM_CLASSES*4);
    f.close();
    Serial.println("Weights saved to SD");
  }
  myVisExportHeader();
}

// ======================================================
// IMAGE LOADING FROM SD
// ======================================================
bool myVisLoadImageFromFile(const char* path, float* buf) {
  File f = SD.open(path);
  if(!f) return false;
  
  size_t sz = f.size();
  uint8_t* jpg = (uint8_t*)ps_malloc(sz);
  if(!jpg) { f.close(); return false; }
  f.read(jpg, sz);
  f.close();
  
  // v43 FIX 3: Use the pre-allocated global myVisRgbBuffer instead of allocating
  // 172KB of PSRAM on every single image load. The old code did ps_malloc(240*240*3)
  // here which was slow, fragmented PSRAM, and is why touch/serial felt unresponsive.
  if(!myVisRgbBuffer) { free(jpg); return false; }
  
  bool ok = fmt2rgb888(jpg, sz, PIXFORMAT_JPEG, myVisRgbBuffer);
  free(jpg);
  if(!ok) return false;
  
  for(int y=0; y<VIS_INPUT_SIZE; y++) {
    for(int x=0; x<VIS_INPUT_SIZE; x++) {
      int sy = (int)((y+0.5)*240.0/VIS_INPUT_SIZE);
      int sx = (int)((x+0.5)*240.0/VIS_INPUT_SIZE);
      if(sy>239) sy=239;
      if(sx>239) sx=239;
      int srcIdx = (sy*240 + sx)*3;
      int dstIdx = (y*VIS_INPUT_SIZE + x)*3;
      buf[dstIdx]   = myVisRgbBuffer[srcIdx]   / 255.0f;
      buf[dstIdx+1] = myVisRgbBuffer[srcIdx+1] / 255.0f;
      buf[dstIdx+2] = myVisRgbBuffer[srcIdx+2] / 255.0f;
    }
  }
  return true;
}

// ======================================================
// PART 0: SETUP AND LOOP
// ======================================================

// Forward declarations for functions defined in other parts
void myVisActionCollect(int classIdx);
void myVisActionTrain();
void myVisActionInfer();
void myVisResetMenuState();
void myVisHandleMenuNavigation();
void myVisDrawMenu();







// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 1: IMAGE COLLECTION FUNCTIONS                                      ██
// ██                                                                          ██
// ██  DEPENDENCIES (functions called from Part 0):                            ██
// ██  - myVisResetMenuState()                     [Part 4]                       ██
// ██  - myVisReadTouch()                          [Part 4]                       ██
// ██                                                                          ██
// ██  VARIABLES USED (defined in Part 0):                                     ██
// ██  - myVisClassLabels[VIS_NUM_CLASSES], myVisThresholdPress, myVisLongPressTime                   ██
// ██  - u8g2 (OLED display object)                                            ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████


// ======================================================
// SHARED OLED RENDER HELPER
// Renders myVisRgbBuffer (must already be filled) to OLED.
// imageCount >= 0  -> show count badge (post-capture mode)
// imageCount == -1 -> show LIVE badge (preview mode)
// ======================================================
void myVisRenderRgbToOLED(int imageCount) {
  int myVisOledWidth  = u8g2.getDisplayWidth();
  int myVisOledHeight = u8g2.getDisplayHeight();
  int myVisScaleX = 240 / myVisOledWidth;
  int myVisScaleY = 240 / myVisOledHeight;

  u8g2.firstPage();
  do {
    for (int myVisOledX = 0; myVisOledX < myVisOledWidth; myVisOledX++) {
      for (int myVisOledY = 0; myVisOledY < myVisOledHeight; myVisOledY++) {
        size_t myVisPixelIndex = ((myVisOledY * myVisScaleY) * 240 + (myVisOledX * myVisScaleX)) * 3;
        uint8_t myVisBrightness = (myVisRgbBuffer[myVisPixelIndex]     +
                                myVisRgbBuffer[myVisPixelIndex + 1] +
                                myVisRgbBuffer[myVisPixelIndex + 2]) / 3;
        if (myVisBrightness > 100) u8g2.drawPixel(myVisOledX, myVisOledY);
      }
    }
    if (imageCount >= 0) {
      // Post-capture: count badge top-left
      u8g2.setFont(u8g2_font_ncenB10_tr);
      u8g2.setColorIndex(0);
      u8g2.drawBox(0, 0, 20, 15);
      u8g2.setColorIndex(1);
      u8g2.setCursor(3, 10);
      u8g2.print(String(imageCount));
    } else {
      // Live preview: LIVE badge top-right
      u8g2.setFont(u8g2_font_5x7_tf);
      u8g2.setColorIndex(0);
      u8g2.drawBox(50, 0, 22, 8);
      u8g2.setColorIndex(1);
      u8g2.drawStr(52, 7, "LIVE");
    }
  } while (u8g2.nextPage());
}

// Post-capture snapshot: convert fb -> myVisRgbBuffer then render with count badge
void myVisDisplayImageOnOLED(camera_fb_t* fb, int imageCount) {
  if (!myVisRgbBuffer) {
    Serial.println("RGB buffer not allocated - skipping OLED preview");
    return;
  }
  if (!fmt2rgb888(fb->buf, fb->len, fb->format, myVisRgbBuffer)) {
    Serial.println("Failed to convert JPEG to RGB888 for OLED");
    return;
  }
  myVisRenderRgbToOLED(imageCount);
}


void myVisActionCollect(int classIdx) {
  if (!myVisSDavailable) {
    Serial.println("No SD card - cannot collect images");
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "No SD card"); } while (u8g2.nextPage());
    delay(2000);
    myVisResetMenuState();
    return;
  }

  Serial.printf("\n>>> Collection mode: %s\n", myVisClassLabels[classIdx].c_str());
  Serial.println("Instructions:");
  Serial.println("  TAP (1-2 taps) = Capture image");
  Serial.println("  LONG PRESS (3+ taps) = Exit to menu");
  Serial.println("  Serial: 'T'=capture, 'L'=exit");
  
  myVisResetTouchState();  // Clear touch state when entering
  
  String path = "/images/" + myVisClassLabels[classIdx];
  if (!SD.exists("/images")) SD.mkdir("/images");
  if (!SD.exists(path)) SD.mkdir(path);


  // Count only the active class — no need to scan all folders on menu entry
  int counts[VIS_NUM_CLASSES] = {};
  File root = SD.open("/images/" + myVisClassLabels[classIdx]);
  if(root) {
    while(File file = root.openNextFile()) {
      if(!file.isDirectory() && (String(file.name()).endsWith(".jpg") || 
        String(file.name()).endsWith(".JPG"))) {
        counts[classIdx]++;
      }
      file.close();
    }
    root.close();
  }

  unsigned long lastCameraDrain = 0;  // how often we service the camera buffer
  unsigned long lastOLED = 0;         // how often we actually update the OLED
  bool oledNeedsUpdate = false;
  bool shouldCapture = false;

  while (true) {
    unsigned long now = millis();

    // --- FAST LOOP: drain camera buffer every 50ms to prevent FB-OVF ---
    if (now - lastCameraDrain > 50) {
      lastCameraDrain = now;

      if (!shouldCapture) {  // don't grab preview frames if a capture is pending
        camera_fb_t* fb = esp_camera_fb_get();
        if (fb) {
          // Only pay for RGB conversion when the OLED is due for a refresh (250ms)
          if (now - lastOLED > 250 && myVisRgbBuffer) {
            if (fmt2rgb888(fb->buf, fb->len, fb->format, myVisRgbBuffer)) {
              oledNeedsUpdate = true;
              lastOLED = now;
            }
          }
          esp_camera_fb_return(fb);
        }
      }
    }

    // --- SLOW LOOP: render to OLED only when fresh RGB is ready ---
    if (oledNeedsUpdate) {
      oledNeedsUpdate = false;
      myVisRenderRgbToOLED(-1);  // -1 = show LIVE badge
    }

    // --- SERIAL INPUT ---
    if (Serial.available()) {
      char c = Serial.read();
      if (c == 'l' || c == 'L') {
        myVisResetMenuState();
        return;
      } else if (c == 't' || c == 'T') {
        shouldCapture = true;
      }
    }

    // --- TOUCH INPUT - unified system ---
    int touchAction = myVisCheckTouchInput();
    if (touchAction == 2) {
      // Long press (3+ taps) - exit
      Serial.println("Exiting collection mode");
      myVisResetMenuState();
      return;
    } else if (touchAction == 1) {
      // Tap (1-2 taps) - capture
      shouldCapture = true;
    }

    // --- CAPTURE ---
    if (shouldCapture) {
      shouldCapture = false;
      camera_fb_t* fb = esp_camera_fb_get();
      if (fb) {
        String fileName = path + "/img_" + String(millis()) + ".jpg";
        File file = SD.open(fileName, FILE_WRITE);
        if (file) {
          file.write(fb->buf, fb->len);
          file.close();
          counts[classIdx]++;
          Serial.printf("Saved: %s (Total: %d)\n", fileName.c_str(), counts[classIdx]);
          myVisDisplayImageOnOLED(fb, counts[classIdx]);  // shows count badge
          delay(300);
          lastOLED = millis();  // don't immediately overwrite the snapshot with LIVE
        }
        esp_camera_fb_return(fb);
      }
    }

    delay(5);
  }
}

// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 2: TRAINING FUNCTIONS (FORWARD/BACKWARD PASS, OPTIMIZER)           ██
// ██                                                                          ██
// ██  DEPENDENCIES (functions called from Part 0):                            ██
// ██  - myVisAllocateMemory()                     [Part 0]                       ██
// ██  - myVisLoadWeights()                        [Part 0]                       ██
// ██  - myVisSaveWeights()                        [Part 0]                       ██
// ██  - myVisLoadImageFromFile()                  [Part 0]                       ██
// ██                                                                          ██
// ██  VARIABLES USED (defined in Part 0):                                     ██
// ██  - All neural network weight/gradient buffers                            ██
// ██  - myVisClassLabels[VIS_NUM_CLASSES], VIS_LEARNING_RATE, VIS_BATCH_SIZE, VIS_TARGET_EPOCHS            ██
// ██  - myVisTrainingData vector, myVisInputBuffer                                  ██
// ██  - u8g2 (OLED display object)                                            ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████


// ======================================================
// FORWARD PASS
// ======================================================
void myVisForwardPass(float* input, float* logits) {
  // Conv1: VIS_INPUT_SIZE x VIS_INPUT_SIZE x 3 -> VIS_CONV1_OUTPUT_SIZE x VIS_CONV1_OUTPUT_SIZE x VIS_CONV1_FILTERS
  for(int f=0; f<VIS_CONV1_FILTERS; f++) {
    int ob = f*VIS_CONV1_OUTPUT_SIZE*VIS_CONV1_OUTPUT_SIZE;
    for(int y=0; y<VIS_CONV1_OUTPUT_SIZE; y++) {
      for(int x=0; x<VIS_CONV1_OUTPUT_SIZE; x++) {
        float sum = 0;
        for(int ky=0; ky<3; ky++) {
          for(int kx=0; kx<3; kx++) {
            int inPos = ((y+ky)*VIS_INPUT_SIZE+(x+kx))*3;
            int wPos = f*27 + ky*9 + kx*3;
            sum += input[inPos]*myVisConv1_w[wPos] + 
                   input[inPos+1]*myVisConv1_w[wPos+1] + 
                   input[inPos+2]*myVisConv1_w[wPos+2];
          }
        }
        myVisConv1_output[ob + y*VIS_CONV1_OUTPUT_SIZE + x] = Visleaky_relu(Visclip_value(sum + myVisConv1_b[f]));
      }
    }
  }
  
  // Pool1: VIS_CONV1_OUTPUT_SIZE x VIS_CONV1_OUTPUT_SIZE -> VIS_POOL1_OUTPUT_SIZE x VIS_POOL1_OUTPUT_SIZE
  for(int f=0; f<VIS_CONV1_FILTERS; f++) {
    int ib=f*VIS_CONV1_OUTPUT_SIZE*VIS_CONV1_OUTPUT_SIZE, ob=f*VIS_POOL1_OUTPUT_SIZE*VIS_POOL1_OUTPUT_SIZE;
    for(int y=0; y<VIS_POOL1_OUTPUT_SIZE; y++) {
      for(int x=0; x<VIS_POOL1_OUTPUT_SIZE; x++) {
        int iy=y*2, ix=x*2;
        float maxVal = myVisConv1_output[ib + iy*VIS_CONV1_OUTPUT_SIZE + ix];
        maxVal = max(maxVal, myVisConv1_output[ib + iy*VIS_CONV1_OUTPUT_SIZE + ix+1]);
        maxVal = max(maxVal, myVisConv1_output[ib + (iy+1)*VIS_CONV1_OUTPUT_SIZE + ix]);
        maxVal = max(maxVal, myVisConv1_output[ib + (iy+1)*VIS_CONV1_OUTPUT_SIZE + ix+1]);
        myVisPool1_output[ob + y*VIS_POOL1_OUTPUT_SIZE + x] = maxVal;
      }
    }
  }
  
  // Conv2: VIS_POOL1_OUTPUT_SIZE x VIS_POOL1_OUTPUT_SIZE x VIS_CONV1_FILTERS -> VIS_CONV2_OUTPUT_SIZE x VIS_CONV2_OUTPUT_SIZE x VIS_CONV2_FILTERS
  for(int f=0; f<VIS_CONV2_FILTERS; f++) {
    int ob=f*VIS_CONV2_OUTPUT_SIZE*VIS_CONV2_OUTPUT_SIZE;
    for(int y=0; y<VIS_CONV2_OUTPUT_SIZE; y++) {
      for(int x=0; x<VIS_CONV2_OUTPUT_SIZE; x++) {
        float sum = 0;
        for(int c=0; c<VIS_CONV1_FILTERS; c++) {
          int ib=c*VIS_POOL1_OUTPUT_SIZE*VIS_POOL1_OUTPUT_SIZE;
          for(int ky=0; ky<3; ky++) {
            for(int kx=0; kx<3; kx++) {
              sum += myVisPool1_output[ib + (y+ky)*VIS_POOL1_OUTPUT_SIZE + (x+kx)] * 
                     myVisConv2_w[f*36 + c*9 + ky*3 + kx];
            }
          }
        }
        myVisConv2_output[ob + y*VIS_CONV2_OUTPUT_SIZE + x] = Visleaky_relu(Visclip_value(sum + myVisConv2_b[f]));
      }
    }
  }
  
  // Dense layer
  for(int c=0; c<VIS_NUM_CLASSES; c++) {
    double sum = 0, comp = 0;
    for(int i=0; i<VIS_FLATTENED_SIZE; i++) {
      double term = myVisConv2_output[i] * myVisOutput_w[c*VIS_FLATTENED_SIZE + i];
      double y = term - comp;
      double t = sum + y;
      comp = (t - sum) - y;
      sum = t;
    }
    myVisDense_output[c] = Visclip_value((float)sum + myVisOutput_b[c], -50, 50);
  }
  
  // Softmax
  float mx = myVisDense_output[0];
  for(int i=1; i<VIS_NUM_CLASSES; i++) mx = max(mx, myVisDense_output[i]);
  float expSum = 0;
  for(int i=0; i<VIS_NUM_CLASSES; i++) expSum += exp(myVisDense_output[i]-mx);
  for(int i=0; i<VIS_NUM_CLASSES; i++) {
    logits[i] = myVisDense_output[i];
    myVisDense_output[i] = exp(myVisDense_output[i]-mx) / expSum;
  }
}

// ======================================================
// BACKWARD PASS
// ======================================================
void myVisBackwardDense(int label) {
  // v43 FIX 1: myVisDense_grad is a per-image propagation signal — zero it fresh each image.
  // myVisOutput_w_grad and myVisOutput_b_grad use += so all images in the batch accumulate.
  // (Batch-level zeroing of those buffers is done once at the start of each batch loop.)
  memset(myVisDense_grad, 0, VIS_FLATTENED_SIZE * sizeof(float));
  for(int c=0; c<VIS_NUM_CLASSES; c++) {
    float error = myVisDense_output[c] - (c==label ? 1.0f : 0.0f);
    for(int i=0; i<VIS_FLATTENED_SIZE; i++) {
      myVisOutput_w_grad[c*VIS_FLATTENED_SIZE+i] += error * myVisConv2_output[i];  // v43: += accumulates
      myVisDense_grad[i] += error * myVisOutput_w[c*VIS_FLATTENED_SIZE+i];
    }
    myVisOutput_b_grad[c] += error;  // v43: += accumulates
  }
}

void myVisBackwardConv2() {
  for(int i=0; i<VIS_FLATTENED_SIZE; i++) {
    myVisConv2_grad[i] = myVisDense_grad[i] * Visleaky_relu_deriv(myVisConv2_output[i]);
  }
  
  // v43 FIX 1: myVisConv2_w_grad and myVisConv2_b_grad are weight accumulators — do NOT zero
  // them here; the batch-level memset at the start of the batch loop handles that.
  // myVisPool1_grad IS zeroed here because it is a per-image propagation signal.
  memset(myVisPool1_grad, 0, VIS_POOL1_OUTPUT_SIZE*VIS_POOL1_OUTPUT_SIZE*VIS_CONV1_FILTERS*sizeof(float));
  
  for(int f=0; f<VIS_CONV2_FILTERS; f++) {
    int ob=f*VIS_CONV2_OUTPUT_SIZE*VIS_CONV2_OUTPUT_SIZE;
    for(int y=0; y<VIS_CONV2_OUTPUT_SIZE; y++) {
      for(int x=0; x<VIS_CONV2_OUTPUT_SIZE; x++) {
        float grad = myVisConv2_grad[ob+y*VIS_CONV2_OUTPUT_SIZE+x];
        myVisConv2_b_grad[f] += grad;
        for(int c=0; c<VIS_CONV1_FILTERS; c++) {
          int ib=c*VIS_POOL1_OUTPUT_SIZE*VIS_POOL1_OUTPUT_SIZE;
          for(int ky=0; ky<3; ky++) {
            for(int kx=0; kx<3; kx++) {
              int pi = ib+(y+ky)*VIS_POOL1_OUTPUT_SIZE+(x+kx);
              int wi = f*36+c*9+ky*3+kx;
              myVisConv2_w_grad[wi] += grad * myVisPool1_output[pi];
              myVisPool1_grad[pi] += grad * myVisConv2_w[wi];
            }
          }
        }
      }
    }
  }
}

void myVisBackwardPool1() {
  memset(myVisConv1_grad, 0, VIS_CONV1_OUTPUT_SIZE*VIS_CONV1_OUTPUT_SIZE*VIS_CONV1_FILTERS*sizeof(float));
  for(int f=0; f<VIS_CONV1_FILTERS; f++) {
    int ib=f*VIS_CONV1_OUTPUT_SIZE*VIS_CONV1_OUTPUT_SIZE, ob=f*VIS_POOL1_OUTPUT_SIZE*VIS_POOL1_OUTPUT_SIZE;
    for(int y=0; y<VIS_POOL1_OUTPUT_SIZE; y++) {
      for(int x=0; x<VIS_POOL1_OUTPUT_SIZE; x++) {
        int iy=y*2, ix=x*2;
        float poolVal = myVisPool1_output[ob+y*VIS_POOL1_OUTPUT_SIZE+x];
        float grad = myVisPool1_grad[ob+y*VIS_POOL1_OUTPUT_SIZE+x];
        if(myVisConv1_output[ib+iy*VIS_CONV1_OUTPUT_SIZE+ix] == poolVal) myVisConv1_grad[ib+iy*VIS_CONV1_OUTPUT_SIZE+ix] += grad;
        if(myVisConv1_output[ib+iy*VIS_CONV1_OUTPUT_SIZE+ix+1] == poolVal) myVisConv1_grad[ib+iy*VIS_CONV1_OUTPUT_SIZE+ix+1] += grad;
        if(myVisConv1_output[ib+(iy+1)*VIS_CONV1_OUTPUT_SIZE+ix] == poolVal) myVisConv1_grad[ib+(iy+1)*VIS_CONV1_OUTPUT_SIZE+ix] += grad;
        if(myVisConv1_output[ib+(iy+1)*VIS_CONV1_OUTPUT_SIZE+ix+1] == poolVal) myVisConv1_grad[ib+(iy+1)*VIS_CONV1_OUTPUT_SIZE+ix+1] += grad;
      }
    }
  }
}

void myVisBackwardConv1() {
  for(int i=0; i<VIS_CONV1_OUTPUT_SIZE*VIS_CONV1_OUTPUT_SIZE*VIS_CONV1_FILTERS; i++) {
    myVisConv1_grad[i] *= Visleaky_relu_deriv(myVisConv1_output[i]);
  }
  
  // v43 FIX 1: myVisConv1_w_grad and myVisConv1_b_grad are weight accumulators — do NOT zero
  // them here; the batch-level memset at the start of the batch loop handles that.
  
  for(int f=0; f<VIS_CONV1_FILTERS; f++) {
    int ob=f*VIS_CONV1_OUTPUT_SIZE*VIS_CONV1_OUTPUT_SIZE;
    for(int y=0; y<VIS_CONV1_OUTPUT_SIZE; y++) {
      for(int x=0; x<VIS_CONV1_OUTPUT_SIZE; x++) {
        float grad = myVisConv1_grad[ob+y*VIS_CONV1_OUTPUT_SIZE+x];
        myVisConv1_b_grad[f] += grad;
        
        for(int ky=0; ky<3; ky++) {
          for(int kx=0; kx<3; kx++) {
            int inPos = ((y+ky)*VIS_INPUT_SIZE+(x+kx))*3;
            int wPos = f*27 + ky*9 + kx*3;
            myVisConv1_w_grad[wPos] += grad * myVisInputBuffer[inPos];
            myVisConv1_w_grad[wPos+1] += grad * myVisInputBuffer[inPos+1];
            myVisConv1_w_grad[wPos+2] += grad * myVisInputBuffer[inPos+2];
          }
        }
      }
    }
  }
}

// ======================================================
// OPTIMIZER
// ======================================================
void myVisAdamUpdate(float* w, float* g, float* m, float* v, int size, int step) {
  float b1=0.9f, b2=0.999f, eps=1e-6f;  // v43 FIX 2: eps 1e-8->1e-6f (float32 NaN prevention)
  float lr_t = VIS_LEARNING_RATE * sqrt(1-pow(b2,step)) / (1-pow(b1,step));
  for(int i=0; i<size; i++) {
    m[i] = b1*m[i] + (1-b1)*g[i];
    v[i] = b2*v[i] + (1-b2)*g[i]*g[i];
    w[i] -= lr_t*m[i]/(sqrt(v[i])+eps);
    w[i] = Visclip_value(w[i], -10, 10);
  }
}

void myVisUpdateWeights(int step) {
  myVisAdamUpdate(myVisConv1_w, myVisConv1_w_grad, myVisConv1_w_m, myVisConv1_w_v, VIS_CONV1_WEIGHTS, step);
  myVisAdamUpdate(myVisConv1_b, myVisConv1_b_grad, myVisConv1_b_m, myVisConv1_b_v, VIS_CONV1_FILTERS, step);
  myVisAdamUpdate(myVisConv2_w, myVisConv2_w_grad, myVisConv2_w_m, myVisConv2_w_v, VIS_CONV2_WEIGHTS, step);
  myVisAdamUpdate(myVisConv2_b, myVisConv2_b_grad, myVisConv2_b_m, myVisConv2_b_v, VIS_CONV2_FILTERS, step);
  myVisAdamUpdate(myVisOutput_w, myVisOutput_w_grad, myVisOutput_w_m, myVisOutput_w_v, VIS_OUTPUT_WEIGHTS, step);
  myVisAdamUpdate(myVisOutput_b, myVisOutput_b_grad, myVisOutput_b_m, myVisOutput_b_v, VIS_NUM_CLASSES, step);
}

// ======================================================
// TRAINING FUNCTION
// ======================================================


void myVisActionTrain() {
  if (!myVisSDavailable) {
    Serial.println("No SD card - cannot train");
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "No SD card"); } while (u8g2.nextPage());
    delay(2000);
    myVisResetMenuState();
    return;
  }

  Serial.println("\n>>> Training mode");
  Serial.println("Instructions:");
  Serial.println("  During training: 3+ taps = Save and exit");
  Serial.println("  After completion: TAP = Train again, 3+ taps = Exit");
  Serial.println("  Serial: 'T'=train again, 'L'=exit");
  
  myVisResetTouchState();  // Clear touch state when entering

  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_6x10_tf);
    u8g2.drawStr(0, 12, "TRAINING MODE");
    u8g2.drawStr(0, 24, "Loading...");
  } while (u8g2.nextPage());
  
  if (myVisLoadWeights()) {
    Serial.println("Continuing from saved weights");
  } else {
    //myVisAllocateMemory();
    Serial.println("Starting fresh training");
  }

  while (true) {
    // Load training data
    myVisTrainingData.clear();
    for(int i=0; i<VIS_NUM_CLASSES; i++) {
      File root = SD.open("/images/" + myVisClassLabels[i]);
      if (root) {
        while(File file = root.openNextFile()) {
          if(!file.isDirectory()) {
            String fn = String(file.name());
            if(fn.endsWith(".jpg") || fn.endsWith(".JPG")) {
              myVisTrainingData.push_back({file.path(), i});
            }
          }
          file.close();
        }
        root.close();
      }
    }
    
    if(myVisTrainingData.empty()) { 
      u8g2.firstPage();
      do { u8g2.drawStr(0, 20, "No Images!"); } while (u8g2.nextPage());
      delay(2000);
      myVisResetMenuState();
      return; 
    }

    // v44: Sort by path for a deterministic split, then hold out the last
    // VALIDATION_IMAGES images per class as a validation set.
    std::sort(myVisTrainingData.begin(), myVisTrainingData.end(),
              [](const VisTrainingItem& a, const VisTrainingItem& b){ return a.path < b.path; });

    std::vector<VisTrainingItem> myVisValidationData;
    if (VALIDATION_IMAGES > 0) {
      int counts[VIS_NUM_CLASSES] = {};
      for (auto& item : myVisTrainingData) counts[item.label]++;
      int skip[VIS_NUM_CLASSES];
      for (int c = 0; c < VIS_NUM_CLASSES; c++) skip[c] = min(VALIDATION_IMAGES, counts[c]);

      std::vector<VisTrainingItem> trainOnly;
      int seen[VIS_NUM_CLASSES] = {};
      for (int i = (int)myVisTrainingData.size() - 1; i >= 0; i--) {
        int c = myVisTrainingData[i].label;
        if (seen[c] < skip[c]) {
          myVisValidationData.push_back(myVisTrainingData[i]);
          seen[c]++;
        } else {
          trainOnly.push_back(myVisTrainingData[i]);
        }
      }
      myVisTrainingData = trainOnly;
      Serial.printf("Val: %d images  Train: %d images\n",
                    (int)myVisValidationData.size(), (int)myVisTrainingData.size());
    }

    int total = myVisTrainingData.size();
    int batchesPerEpoch = (total + VIS_BATCH_SIZE - 1) / VIS_BATCH_SIZE;
    int totalBatches = VIS_TARGET_EPOCHS * batchesPerEpoch;
    
    Serial.printf("Training: %d images, %d batches\n", total, totalBatches);
    
    // Training loop
    std::vector<int> indices;
    for(int i=0; i<total; i++) indices.push_back(i);
    
    float runningLoss = 0;
    int lossCount = 0;
    
    for(int batch=0; batch<totalBatches; batch++) {
      // Check for exit during training
      if (Serial.available()) {
        char c = Serial.read();
        // v43 FIX 4: accept 'l'/'L' as well as 'x'/'X' — consistent with all other modes
        if (c == 'x' || c == 'X' || c == 'l' || c == 'L') {
          Serial.println("Stopping training...");
          myVisSaveWeights();
          myVisWeightsTrained = true; 
          myVisResetMenuState();
          return;
        }
      }
      
      // Touch input during training - check in background
      myVisCheckTouchBackground();  // Update touch state without blocking
      if (myVisPeekTouchAction() == 2) {
        myVisCheckTouchInput();  // Consume the action
        Serial.println("Long press - stopping training");
        myVisSaveWeights();
        myVisWeightsTrained = true; 
        myVisResetMenuState();
        return;
      }
      
      // Shuffle at epoch start
      if(batch % batchesPerEpoch == 0) {
        int epoch = batch/batchesPerEpoch + 1;
        Serial.printf("\n--- Epoch %d/%d ---\n", epoch, VIS_TARGET_EPOCHS);
        for(int i=total-1; i>0; i--) {
          int j = random(i+1);
          int tmp = indices[i];
          indices[i] = indices[j];
          indices[j] = tmp;
        }
      }
      
      int batchStart = (batch % batchesPerEpoch) * VIS_BATCH_SIZE;
      int batchEnd = min(batchStart + VIS_BATCH_SIZE, total);
      
      float batchLoss = 0;
      int correctCount = 0;
      
      // v43 FIX 1: Zero ALL weight gradient buffers once per batch before accumulating.
      // This replaces the per-function memsets that were incorrectly resetting grads
      // between images mid-batch in v42.
      memset(myVisConv1_w_grad,  0, VIS_CONV1_WEIGHTS  * sizeof(float));
      memset(myVisConv1_b_grad,  0, VIS_CONV1_FILTERS  * sizeof(float));
      memset(myVisConv2_w_grad,  0, VIS_CONV2_WEIGHTS  * sizeof(float));
      memset(myVisConv2_b_grad,  0, VIS_CONV2_FILTERS  * sizeof(float));
      memset(myVisOutput_w_grad, 0, VIS_OUTPUT_WEIGHTS * sizeof(float));
      memset(myVisOutput_b_grad, 0, VIS_NUM_CLASSES    * sizeof(float));
      
      // Train on batch
      for(int i=batchStart; i<batchEnd; i++) {
        int idx = indices[i];
        VisTrainingItem& img = myVisTrainingData[idx];
        
        if(!myVisLoadImageFromFile(img.path.c_str(), myVisInputBuffer)) continue;
        
        float logits[VIS_NUM_CLASSES];
        myVisForwardPass(myVisInputBuffer, logits);
        
        float loss = -log(max(myVisDense_output[img.label], 1e-7f));
        batchLoss += loss;
        
        int pred = 0;
        for(int j=1; j<VIS_NUM_CLASSES; j++) if(myVisDense_output[j] > myVisDense_output[pred]) pred = j;
        if(pred == img.label) correctCount++;
        
        myVisBackwardDense(img.label);
        myVisBackwardConv2();
        myVisBackwardPool1();
        myVisBackwardConv1();
        
        // Update touch state during heavy computation
        // v43 FIX 5: also peek for action here so a tap exits within one image,
        // not at the end of the entire batch (which can be a 5-15 second wait)
        if (i % 3 == 0) {
          myVisCheckTouchBackground();
          if (myVisPeekTouchAction() == 2) {
            myVisCheckTouchInput();  // consume
            Serial.println("Long press - stopping training");
            myVisSaveWeights();
            myVisWeightsTrained = true; 
            myVisResetMenuState();
            return;
          }
          if (Serial.available()) {
            char c = Serial.read();
            if (c == 'x' || c == 'X' || c == 'l' || c == 'L') {
              Serial.println("Stopping training...");
              myVisSaveWeights();
              myVisWeightsTrained = true; 
              myVisResetMenuState();
              return;
            }
          }
        }
      }
      
      myVisUpdateWeights(batch+1);
      
      float avgLoss = batchLoss / (batchEnd - batchStart);
      float batchAcc = (float)correctCount / (batchEnd - batchStart);
      runningLoss += avgLoss;
      lossCount++;
      
      // Update display
      if((batch+1) % 5 == 0) {
        float displayLoss = runningLoss / lossCount;
        u8g2.firstPage();
        do {
          u8g2.setFont(u8g2_font_5x7_tf);   // _6x10_tf
          u8g2.setCursor(0, 12); u8g2.print("Training...");
          u8g2.setCursor(0, 24); 
          u8g2.print("B:"); u8g2.print(batch+1); 
          u8g2.print("/"); u8g2.print(totalBatches);
          u8g2.setCursor(0, 36); 
          u8g2.print("L:"); u8g2.print(displayLoss, 3);
          u8g2.print(" A:"); u8g2.print((int)(batchAcc*100)); u8g2.print("%");
        } while (u8g2.nextPage());
        runningLoss = 0;
        lossCount = 0;
      }
      
      if((batch+1) % 10 == 0) {
        Serial.printf("Batch %d/%d - Loss: %.4f - Acc: %.1f%%\n", 
                     batch+1, totalBatches, avgLoss, batchAcc*100);
      }
    }
    
    Serial.println("\n--- Training Complete ---");

    // v44: Run forward pass on held-out validation images and report accuracy.
    if (!myVisValidationData.empty()) {
      int valCorrect = 0;
      int valCount   = 0;
      for (auto& vitem : myVisValidationData) {
        if (!myVisLoadImageFromFile(vitem.path.c_str(), myVisInputBuffer)) continue;
        float logits[VIS_NUM_CLASSES];
        myVisForwardPass(myVisInputBuffer, logits);
        int pred = 0;
        for (int j = 1; j < VIS_NUM_CLASSES; j++)
          if (myVisDense_output[j] > myVisDense_output[pred]) pred = j;
        if (pred == vitem.label) valCorrect++;
        valCount++;
      }
      if (valCount > 0) {
        Serial.printf("Validation Accuracy: %.1f%%  (%d/%d correct)\n",
                      100.0f * valCorrect / valCount, valCorrect, valCount);
      }
    }

    myVisSaveWeights();
    myVisWeightsTrained = true;

    u8g2.firstPage();
    do { 
      u8g2.drawStr(0, 12, "DONE!");
      u8g2.drawStr(0, 24, "Tap:Again");
      u8g2.drawStr(0, 36, "3+Taps:Exit");
    } while (u8g2.nextPage());

    myVisResetTouchState();
    
    Serial.println("Waiting for input...");
    Serial.println("  Serial: T=train again  L=exit");
    Serial.println("  Touch:  1-2 taps=train again  3+taps=exit");
    while (true) {
      if (Serial.available()) {
        char c = Serial.read();
        // v43 FIX: added l/L as exit — was missing here (only x/X worked before)
        if (c == 'x' || c == 'X' || c == 'l' || c == 'L') {
          myVisResetMenuState();
          return;
        } else if (c == 't' || c == 'T') {
          break;
        }
      }
      int touchAction = myVisCheckTouchInput();
      if (touchAction == 2) {
        myVisResetMenuState();
        return;
      } else if (touchAction == 1) {
        Serial.println("Starting new training cycle");
        break;
      }
      delay(10);
    }
  }
}


// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 3: INFERENCE FUNCTION - OPTIMIZED                                  ██
// ██                                                                          ██
// ██  DEPENDENCIES (functions called from Part 0):                            ██
// ██  - myVisLoadWeights()    // Weights loaded in setup()                       ██
// ██  - myVisForwardPass()                        [Part 2]                       ██
// ██  - myVisRgbBuffer (global, allocated in setup)                              ██
// ██                                                                          ██
// ██  VARIABLES USED (defined in Part 0):                                     ██
// ██  - myVisInputBuffer, myVisDense_output (probabilities)                         ██
// ██  - myVisClassLabels[VIS_NUM_CLASSES], myVisThresholdPress                                    ██
// ██  - u8g2 (OLED display object)                                            ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████


void myVisActionInfer() {
  // Guard: refuse to run if no trained weights are loaded
  if (!myVisWeightsTrained) {
    Serial.println("ERROR: No trained weights! Please run menu item 4 (Train) first.");
    u8g2.firstPage();
    do {
      u8g2.setFont(u8g2_font_6x10_tf);
      u8g2.drawStr(0, 12, "No weights!");
      u8g2.drawStr(0, 24, "Train first");
      u8g2.drawStr(0, 36, "(menu item 4)");
    } while (u8g2.nextPage());
    delay(3000);
    myVisResetMenuState();
    return;
  }
  Serial.println("\n>>> Inference mode - OPTIMIZED");
  Serial.println("Instructions:");
  Serial.println("  T or L exit to menu");

  
  myVisResetTouchState();  // Clear touch state when entering
  
  // Weights already loaded in setup() - just verify PSRAM is ready
  if (!myVisInputBuffer || !myVisDense_output) {
    Serial.println("ERROR: Memory not allocated - cannot infer");
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "NOT READY!"); } while (u8g2.nextPage());
    delay(2000);
    myVisResetMenuState();
    return;
  }
  
  // Pre-compute resize lookup tables (done once)
  static int sy_lookup[VIS_INPUT_SIZE];
  static int sx_lookup[VIS_INPUT_SIZE];
  static bool lookup_initialized = false;
  
  if (!lookup_initialized) {
    for(int i=0; i<VIS_INPUT_SIZE; i++) {
      sy_lookup[i] = min((int)((i+0.5)*240.0/VIS_INPUT_SIZE), 239);
      sx_lookup[i] = min((int)((i+0.5)*240.0/VIS_INPUT_SIZE), 239);
    }
    lookup_initialized = true;
    Serial.println("Resize lookup tables initialized");
  }
  
  // Timing arrays for 10-frame batches
  unsigned long frameTimes[10];
  int frameIndex = 0;
  int pred = 0;  // Store prediction outside loop for printing
  
  while (true) {
    unsigned long frameStart = millis();
    
    // Serial input check (fast, every frame)
    if (Serial.available()) {
      char c = Serial.read();
      if (c == 't' || c == 'T' || c == 'l' || c == 'L') {
        myVisResetMenuState();
        return;
      }
    }
    
    // Get camera frame
    camera_fb_t * fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera frame failed - retrying");
      delay(10);
      continue;
    }
    
    // Check if RGB buffer is allocated
    if (!myVisRgbBuffer) {
      Serial.println("ERROR: myVisRgbBuffer not allocated!");
      esp_camera_fb_return(fb);
      delay(10);
      continue;
    }
    
    // Convert JPEG to RGB (reusing pre-allocated buffer)
    if (fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, myVisRgbBuffer)) {
      
      // Optimized resize using lookup tables
      for(int y=0; y<VIS_INPUT_SIZE; y++) {
        int sy = sy_lookup[y];
        int sy_offset = sy * 240;
        int dst_y_offset = y * VIS_INPUT_SIZE;
        
        for(int x=0; x<VIS_INPUT_SIZE; x++) {
          int srcIdx = (sy_offset + sx_lookup[x]) * 3;
          int dstIdx = (dst_y_offset + x) * 3;
          myVisInputBuffer[dstIdx] = myVisRgbBuffer[srcIdx] * 0.003921569f;      // /255.0
          myVisInputBuffer[dstIdx+1] = myVisRgbBuffer[srcIdx+1] * 0.003921569f;
          myVisInputBuffer[dstIdx+2] = myVisRgbBuffer[srcIdx+2] * 0.003921569f;
        }
      }
      
      // Run inference
      float myVisLogits[VIS_NUM_CLASSES];
      myVisForwardPass(myVisInputBuffer, myVisLogits);
      
      // Find prediction
      pred = 0;
      for(int i=1; i<VIS_NUM_CLASSES; i++) {
        if(myVisDense_output[i] > myVisDense_output[pred]) pred = i;
      }

      // Every 10th frame: draw live image + label overlay on OLED.
      // Done HERE while myVisRgbBuffer is still valid (before fb is returned).
      if (frameIndex == 9) {
        int oW = u8g2.getDisplayWidth();
        int oH = u8g2.getDisplayHeight();
        int scX = 240 / oW;
        int scY = 240 / oH;
        u8g2.firstPage();
        do {
          // Draw downsampled camera image
          for (int ox = 0; ox < oW; ox++) {
            for (int oy = 0; oy < oH; oy++) {
              int pi = ((oy * scY) * 240 + (ox * scX)) * 3;
              uint8_t bright = (myVisRgbBuffer[pi] + myVisRgbBuffer[pi+1] + myVisRgbBuffer[pi+2]) / 3;
              if (bright > 100) u8g2.drawPixel(ox, oy);
            }
          }
          // Label overlay bar at bottom
          u8g2.setFont(u8g2_font_5x7_tf);
          u8g2.setColorIndex(0);
          u8g2.drawBox(0, oH - 9, oW, 9);
          u8g2.setColorIndex(1);
          char buf[20];
          snprintf(buf, sizeof(buf), "%s %d%%",
                   myVisClassLabels[pred].c_str(),
                   (int)(myVisDense_output[pred] * 100));
          u8g2.drawStr(1, oH - 1, buf);
        } while (u8g2.nextPage());
      }
    }
    
    esp_camera_fb_return(fb);
    
    // Record frame timing
    frameTimes[frameIndex] = millis() - frameStart;
    float fps2 = 1000.0 / frameTimes[frameIndex];
    Serial.printf("Frame %d: %lu ms (%.1f FPS) ", frameIndex+1, frameTimes[frameIndex], fps2);
    frameIndex++;
    Serial.printf("Current Pred: %s (%.1f%%) | All:", 
                   myVisClassLabels[pred].c_str(), myVisDense_output[pred]*100);
    for(int i=0; i<VIS_NUM_CLASSES; i++) Serial.printf(" %.0f%%", myVisDense_output[i]*100);
    Serial.println();
   
    // Every 10th frame: touch exit check (OLED image already drawn above before fb return)
    if (frameIndex >= 10) {
      int touchVal = myVisReadTouch();
      if (touchVal > myVisThresholdPress) {
        Serial.println("Touch detected - exiting inference");
        delay(200);
        myVisResetMenuState();
        return;
      }
      frameIndex = 0;
    }
  }
}


// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 4: MENU SYSTEM FUNCTIONS                                           ██
// ██                                                                          ██
// ██  DEPENDENCIES (functions called from Part 0):                            ██
// ██  - myVisActionCollect(int classIdx)          [Part 1]                       ██
// ██  - myVisActionTrain()                        [Part 2]                       ██
// ██  - myVisActionInfer()                        [Part 3]                       ██
// ██                                                                          ██
// ██  VARIABLES USED (defined in Part 0):                                     ██
// ██  - myVisClassLabels[VIS_NUM_CLASSES]                                                      ██
// ██  - myVisTotalItems, myVisThresholdPress, myVisThresholdRelease                    ██
// ██  - myVisLastActivityTime, myVisLastTapTime, myVisTapCooldown                      ██
// ██  - myVisIsTouching, myVisLongPressTriggered, myVisMenuIndex, myVisIsSelected         ██
// ██  - u8g2 (OLED display object)                                            ██
// ██                                                                          ██
// ██  NOTE: This part is called from loop() in Part 0                         ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████


void myVisResetMenuState() {
  myVisIsSelected = false;
  myVisResetTouchState();  // Use unified touch reset
  myVisLastActivityTime = millis();
  // NOTE: myVisDrawMenu() call removed here — in the combined sketch the
  // unified myDrawMenu() (see COMBINED MENU section) redraws the OLED
  // after any action returns, so this model-local menu stays silent.
}

void myVisDrawMenu() {
  // ===== SERIAL MENU =====
  Serial.println("\n=== MENU ===");
  for (int i = 1; i <= myVisTotalItems; i++) {
    String label =
      (i <= VIS_NUM_CLASSES) ? myVisClassLabels[i - 1] :
      (i == VIS_NUM_CLASSES + 1) ? "Train" : "Infer";

    if (i == myVisMenuIndex) Serial.print(" > ");
    else                 Serial.print("   ");

    Serial.printf("%d. %s\n", i, label.c_str());
  }
  Serial.println("Commands: t=next (tap)  l=select (longpress)");

  // ===== OLED MENU =====
  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_6x10_tf);
    u8g2.drawStr(0, 8, "TAP:Next HOLD:Ok");

    int myVisStartItem = (myVisMenuIndex <= VIS_NUM_CLASSES) ? 1 : myVisMenuIndex - 2;

    for (int i = 0; i < 3; i++) {
      int cur = myVisStartItem + i;
      if (cur > myVisTotalItems) break;

      String label =
        (cur <= VIS_NUM_CLASSES) ? myVisClassLabels[cur - 1] :
        (cur == VIS_NUM_CLASSES + 1) ? "Train" : "Infer";

      int y = 18 + i * 9;
      if (cur == myVisMenuIndex)
        u8g2.drawStr(0, y, ("> " + label).c_str());
      else
        u8g2.drawStr(0, y, ("  " + label).c_str());
    }
  } while (u8g2.nextPage());
}

// Helper: execute the currently selected menu item
void myVisExecuteMenuItem(int idx) {
  if (idx <= VIS_NUM_CLASSES)        myVisActionCollect(idx - 1);
  else if (idx == VIS_NUM_CLASSES+1) myVisActionTrain();
  else                           myVisActionInfer();
}

void myVisHandleMenuNavigation() {
  unsigned long myVisCurrentMillis = millis();

  // --------------------------------------------------------------------------
  // SERIAL INPUT
  // --------------------------------------------------------------------------
  if (!myVisIsSelected && Serial.available()) {
    char c = Serial.read();

    // Single-digit direct selection (works for VIS_NUM_CLASSES up to 9+2=11 items via digit keys)
    if (c >= '1' && c <= '9') {
      int newIndex = c - '0';
      if (newIndex <= myVisTotalItems) {
        myVisMenuIndex = newIndex;
        myVisIsSelected = true;
        myVisLastActivityTime = myVisCurrentMillis;
        myVisExecuteMenuItem(myVisMenuIndex);
      }
    }
    else if (c == 't' || c == 'T') {
      if (myVisCurrentMillis - myVisLastTapTime > myVisTapCooldown) {
        myVisMenuIndex++;
        if (myVisMenuIndex > myVisTotalItems) myVisMenuIndex = 1;
        myVisDrawMenu();
        myVisLastTapTime = myVisCurrentMillis;
        myVisLastActivityTime = myVisCurrentMillis;
      }
    }
    else if (c == 'l' || c == 'L') {
      myVisIsSelected = true;
      myVisLastActivityTime = myVisCurrentMillis;
      myVisExecuteMenuItem(myVisMenuIndex);
    }
  }

  // --------------------------------------------------------------------------
  // TOUCH INPUT - NOW USING UNIFIED SYSTEM
  // --------------------------------------------------------------------------
  if (!myVisIsSelected) {
    int touchAction = myVisCheckTouchInput();
    
    if (touchAction == 1) {
      // Tap detected - advance menu
      if (myVisCurrentMillis - myVisLastTapTime > myVisTapCooldown) {
        myVisMenuIndex++;
        if (myVisMenuIndex > myVisTotalItems) myVisMenuIndex = 1;
        myVisDrawMenu();
        myVisLastTapTime = myVisCurrentMillis;
        myVisLastActivityTime = myVisCurrentMillis;
      }
    }
    else if (touchAction == 2) {
      // Long press detected - select menu item
      myVisIsSelected = true;
      myVisLastActivityTime = myVisCurrentMillis;
      myVisExecuteMenuItem(myVisMenuIndex);
    }
  }
}// ======================================================
// XIAO ESP32-S3 SENSE
// FULL AUDIO ML — WAKE WORD CLASSIFICATION v13
//
//
//
// SD card layout:
//   /audio/<ClassName>/clip_NNNNN.wav   — training clips
//   /header/mySndWeights.bin               — saved weights (binary)
//   /header/mySndWeights.h                 — saved weights (C header)
//
// INPUT:  1-second PDM @ 16 kHz -> 40-band Mel spectrogram (40x32 = 1280)
// MODEL:  Conv1(3x3,4) -> MaxPool -> Conv2(3x3,8) -> Dense -> Softmax
//
// Arduino IDE Tools:
//   Board:           XIAO_ESP32S3
//   USB CDC On Boot: Enabled
//   PSRAM:           OPI PSRAM
//
// By Jeremy Ellis  https://github.com/hpssjellis
// MIT License
// ======================================================

//#define USE_BAKED_WEIGHTS
#ifdef USE_BAKED_WEIGHTS
  #include "mySndWeights.h"
#endif

#include "ESP_I2S.h"
#include "FS.h"
#include "SD.h"
#include "SPI.h"
#include <vector>
#include <algorithm>
#include <math.h>
#include <U8g2lib.h>
#include <Wire.h>
#include "freertos/FreeRTOS.h"   // for vTaskDelay(0) — yields to scheduler/idle task

// u8g2 object is declared once, near the top of the Vision section above.

// ======================================================
// CLASS LABELS — edit to match your wake words
// ======================================================
#define SND_NUM_CLASSES 3
String mySndClassLabels[SND_NUM_CLASSES] = {"0unknown", "1round", "2square"};
const int mySndTotalItems = SND_NUM_CLASSES + 2;   // classes + Train + Infer

// ======================================================
// HYPERPARAMETERS
// ======================================================
float SND_LEARNING_RATE    = 0.003f;   //= 0.0003f;
int   SND_BATCH_SIZE       = 12;          // 6;
int   SND_TARGET_EPOCHS    = 20;         //  20;
int   VALIDATION_CLIPS = 4l;      // Note this is 4 and "l" for long data type (a 64-bit integer) 
                                  // instead of a standard int (a 32-bit integer).
                                  // clips per class held out for validation
// ======================================================
// AUDIO / SPECTROGRAM CONSTANTS
// ======================================================
#define SND_SAMPLE_RATE   16000
#define SND_CLIP_SECONDS  1
#define SND_CLIP_SAMPLES  (SND_SAMPLE_RATE * SND_CLIP_SECONDS)   // 16000 samples
#define SND_MEL_BINS      40
#define SND_MEL_FRAMES    32
#define SND_FRAME_SIZE    (SND_CLIP_SAMPLES / SND_MEL_FRAMES)    // 500 samples/frame
#define SND_FFT_SIZE      512

#define SND_INPUT_W       SND_MEL_BINS    // 40  (spectrogram "width")
#define SND_INPUT_H       SND_MEL_FRAMES  // 32  (spectrogram "height")
#define SND_INPUT_CH      1           // single-channel (greyscale spectrogram)

// ======================================================
// CNN ARCHITECTURE  (mirrors vision v44, 1-channel input)
// ======================================================
#define SND_CONV1_FILTERS  4
#define SND_CONV1_WEIGHTS  (3 * 3 * SND_INPUT_CH * SND_CONV1_FILTERS)        //  36

#define SND_CONV2_FILTERS  8
#define SND_CONV2_WEIGHTS  (3 * 3 * SND_CONV1_FILTERS * SND_CONV2_FILTERS)   // 288

#define SND_CONV1_OUT_H   (SND_INPUT_H - 2)         // 30
#define SND_CONV1_OUT_W   (SND_INPUT_W - 2)         // 38
#define SND_POOL1_OUT_H   (SND_CONV1_OUT_H / 2)     // 15
#define SND_POOL1_OUT_W   (SND_CONV1_OUT_W / 2)     // 19
#define SND_CONV2_OUT_H   (SND_POOL1_OUT_H - 2)     // 13
#define SND_CONV2_OUT_W   (SND_POOL1_OUT_W - 2)     // 17

#define SND_FLATTENED_SIZE  (SND_CONV2_OUT_H * SND_CONV2_OUT_W * SND_CONV2_FILTERS)  // 1768
#define SND_OUTPUT_WEIGHTS  (SND_FLATTENED_SIZE * SND_NUM_CLASSES)

// ======================================================
// TOUCH INPUT
// ======================================================
const int mySndThresholdPress   = 1100;
const int mySndThresholdRelease = 900;

struct SndTouchState {
  bool          isTouching      = false;
  int           tapCount        = 0;
  unsigned long firstTapTime    = 0;
  unsigned long lastReleaseTime = 0;
  unsigned long lastCheckTime   = 0;
  const unsigned long tapWindow     = 800;
  const int           longPressTaps = 3;
  const unsigned long debounceDelay = 50;
};
SndTouchState mySndTouch;

// ======================================================
// SYSTEM STATE
// ======================================================
unsigned long mySndLastActivityTime = 0;
unsigned long mySndLastTapTime      = 0;
const int     mySndTapCooldown      = 250;
int  mySndMenuIndex      = 1;
bool mySndIsSelected     = false;
bool mySndWeightsTrained = false;
bool mySndSDavailable    = false;

// ======================================================
// AUDIO
// ======================================================
I2SClass mySndI2S;
int16_t* mySndAudioBuffer = nullptr;

// ======================================================
// SPECTROGRAM SCRATCH BUFFERS — allocated once in PSRAM
// ======================================================
float* mySndSpectroFrame = nullptr;   // SND_FFT_SIZE floats
float* mySndSpectroPower = nullptr;   // SND_FFT_SIZE/2+1 floats

// ======================================================
// ML WEIGHT / ACTIVATION / GRADIENT BUFFERS (PSRAM)
// ======================================================
float* mySndInputBuffer  = nullptr;

float* mySndConv1_w  = nullptr;  float* mySndConv1_b  = nullptr;
float* mySndConv2_w  = nullptr;  float* mySndConv2_b  = nullptr;
float* mySndOutput_w = nullptr;  float* mySndOutput_b = nullptr;

float* mySndConv1_w_grad  = nullptr; float* mySndConv1_b_grad  = nullptr;
float* mySndConv2_w_grad  = nullptr; float* mySndConv2_b_grad  = nullptr;
float* mySndOutput_w_grad = nullptr; float* mySndOutput_b_grad = nullptr;

float* mySndConv1_w_m = nullptr; float* mySndConv1_w_v = nullptr;
float* mySndConv1_b_m = nullptr; float* mySndConv1_b_v = nullptr;
float* mySndConv2_w_m = nullptr; float* mySndConv2_w_v = nullptr;
float* mySndConv2_b_m = nullptr; float* mySndConv2_b_v = nullptr;
float* mySndOutput_w_m = nullptr; float* mySndOutput_w_v = nullptr;
float* mySndOutput_b_m = nullptr; float* mySndOutput_b_v = nullptr;

float* mySndConv1_output = nullptr;
float* mySndPool1_output = nullptr;
float* mySndConv2_output = nullptr;
float* mySndDense_output = nullptr;

// Pool1 argmax: stores which of the 4 cells (0-3) in each 2x2 window
// was the maximum, so the backward pass can route gradients correctly
// without fragile float equality comparisons.
// Size = SND_POOL1_OUT_H * SND_POOL1_OUT_W * SND_CONV1_FILTERS
uint8_t* mySndPool1_argmax = nullptr;

float* mySndDense_grad = nullptr;
float* mySndConv2_grad = nullptr;
float* mySndPool1_grad = nullptr;
float* mySndConv1_grad = nullptr;

struct SndTrainingItem { String path; int label; };
std::vector<SndTrainingItem> mySndTrainingData;

// ======================================================
// UTILITY
// ======================================================
inline float Sndclip_value(float v, float mn = -100, float mx = 100) {
  if (isnan(v) || isinf(v)) return 0;
  return constrain(v, mn, mx);
}
inline float Sndleaky_relu(float x)       { return x > 0 ? x : 0.1f * x; }
inline float Sndleaky_relu_deriv(float x) { return x > 0 ? 1.0f : 0.1f; }

// ======================================================
// TOUCH INPUT
// ======================================================
int mySndReadTouch() {
  int sum = 0;
  for (int i = 0; i < 3; i++) { sum += analogRead(A0); delayMicroseconds(100); }
  return sum / 3;
}

void mySndResetTouchState() {
  mySndTouch.isTouching      = false;
  mySndTouch.tapCount        = 0;
  mySndTouch.firstTapTime    = 0;
  mySndTouch.lastReleaseTime = 0;
  mySndTouch.lastCheckTime   = 0;
}

void mySndUpdateTouchState() {
  unsigned long now = millis();
  if (now - mySndTouch.lastCheckTime < 20) return;
  mySndTouch.lastCheckTime = now;
  int val = mySndReadTouch();
  bool active = mySndTouch.isTouching ? (val > mySndThresholdRelease) : (val > mySndThresholdPress);
  if (active && !mySndTouch.isTouching) {
    if (now - mySndTouch.lastReleaseTime < mySndTouch.debounceDelay) return;
    mySndTouch.isTouching = true;
    if (mySndTouch.tapCount == 0 || (now - mySndTouch.firstTapTime < mySndTouch.tapWindow)) {
      if (mySndTouch.tapCount == 0) mySndTouch.firstTapTime = now;
      mySndTouch.tapCount++;
    } else {
      mySndTouch.tapCount     = 1;
      mySndTouch.firstTapTime = now;
    }
  }
  if (!active && mySndTouch.isTouching) {
    mySndTouch.isTouching      = false;
    mySndTouch.lastReleaseTime = now;
  }
}

int mySndCheckTouchInput() {
  mySndUpdateTouchState();
  unsigned long now = millis();
  if (mySndTouch.tapCount > 0 && !mySndTouch.isTouching &&
      now - mySndTouch.firstTapTime > mySndTouch.tapWindow) {
    int result = (mySndTouch.tapCount >= mySndTouch.longPressTaps) ? 2 : 1;
    mySndResetTouchState();
    return result;
  }
  return 0;
}

void mySndCheckTouchBackground() { mySndUpdateTouchState(); }

int mySndPeekTouchAction() {
  mySndUpdateTouchState();
  if (mySndTouch.tapCount > 0 && !mySndTouch.isTouching &&
      millis() - mySndTouch.firstTapTime > mySndTouch.tapWindow)
    return (mySndTouch.tapCount >= mySndTouch.longPressTaps) ? 2 : 1;
  return 0;
}

// ======================================================
// RECORD AND SAVE A CLIP TO SD
// ======================================================
bool mySndSaveClipToSD(const String& path) {
  size_t   wav_size   = 0;
  uint8_t* wav_buffer = mySndI2S.recordWAV(SND_CLIP_SECONDS, &wav_size);
  if (!wav_buffer || wav_size == 0) {
    Serial.println("recordWAV() returned null/empty");
    return false;
  }
  File f = SD.open(path, FILE_WRITE);
  if (!f) {
    free(wav_buffer);
    Serial.println("SD open failed for write");
    return false;
  }
  bool ok = (f.write(wav_buffer, wav_size) == wav_size);
  f.close();
  free(wav_buffer);
  if (!ok) Serial.println("SD write incomplete");
  return ok;
}

// ======================================================
// LOAD A CLIP FROM SD INTO mySndAudioBuffer
// ======================================================
bool mySndLoadClipFromSD(const char* path) {
  File f = SD.open(path);
  if (!f) {
    Serial.printf("  SD.open FAILED: %s\n", path);
    return false;
  }
  f.seek(44);   // skip WAV header
  size_t want = SND_CLIP_SAMPLES * sizeof(int16_t);
  size_t got  = f.read((uint8_t*)mySndAudioBuffer, want);
  f.close();
  if (got < want) memset((uint8_t*)mySndAudioBuffer + got, 0, want - got);
  return (got > 0);
}

// ======================================================
// MEL FILTERBANK (built once, cached in PSRAM)
// ======================================================
static float mySndHzToMel(float hz)  { return 2595.0f * log10f(1.0f + hz / 700.0f); }
static float mySndMelToHz(float mel) { return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f); }

static float* mySndMelFbLo  = nullptr;
static float* mySndMelFbCtr = nullptr;
static float* mySndMelFbHi  = nullptr;

void mySndBuildMelFilterbank() {
  if (mySndMelFbCtr) return;
  mySndMelFbLo  = (float*)ps_malloc(SND_MEL_BINS * sizeof(float));
  mySndMelFbCtr = (float*)ps_malloc(SND_MEL_BINS * sizeof(float));
  mySndMelFbHi  = (float*)ps_malloc(SND_MEL_BINS * sizeof(float));
  if (!mySndMelFbLo || !mySndMelFbCtr || !mySndMelFbHi) {
    Serial.println("FATAL: mel filterbank alloc failed");
    while (1) delay(1000);
  }
  float melLow   = mySndHzToMel(20.0f);
  float melHigh  = mySndHzToMel(SND_SAMPLE_RATE / 2.0f);
  float melStep  = (melHigh - melLow) / (SND_MEL_BINS + 1);
  float binScale = (float)SND_FFT_SIZE / (float)SND_SAMPLE_RATE;
  for (int m = 0; m < SND_MEL_BINS; m++) {
    mySndMelFbLo[m]  = mySndMelToHz(melLow + (m    ) * melStep) * binScale;
    mySndMelFbCtr[m] = mySndMelToHz(melLow + (m + 1) * melStep) * binScale;
    mySndMelFbHi[m]  = mySndMelToHz(melLow + (m + 2) * melStep) * binScale;
  }
  Serial.println("Mel filterbank built");
}

// ======================================================
// POWER SPECTRUM — inline radix-2 Cooley-Tukey FFT  O(N log N)
//
// Replaces the O(N²) DFT from v10.  For SND_FFT_SIZE=512:
//   Old DFT : 512 × 257 × 2 trig calls × 32 frames ≈ 8.4 M trig calls/clip
//   New FFT : 512 × log2(512) × 32 frames            ≈ 147 K trig calls/clip
// Measured speedup on ESP32-S3: ~35× per clip, from ~1.5 s to ~45 ms.
//
// Requirements: n must be a power of 2 (SND_FFT_SIZE=512 satisfies this).
// In-place bit-reversal + butterfly on separate real/imag scratch arrays.
// mySndFftReal / mySndFftImag are reused across calls (allocated in PSRAM).
// vTaskDelay(0) is still called once per frame to keep the WDT happy.
// ======================================================

// Scratch arrays — allocated once alongside the other PSRAM buffers.
// Declared here as extern; definition lives in mySndAllocateMemory().
static float* mySndFftReal = nullptr;
static float* mySndFftImag = nullptr;

// Precomputed twiddle table (cos/sin) for SND_FFT_SIZE/2 factors, built once.
static float* mySndTwiddleCos = nullptr;
static float* mySndTwiddleSin = nullptr;

void mySndBuildTwiddleTable() {
  if (mySndTwiddleCos) return;  // already built
  mySndTwiddleCos = (float*)ps_malloc((SND_FFT_SIZE / 2) * sizeof(float));
  mySndTwiddleSin = (float*)ps_malloc((SND_FFT_SIZE / 2) * sizeof(float));
  if (!mySndTwiddleCos || !mySndTwiddleSin) {
    Serial.println("FATAL: twiddle table alloc failed");
    while (1) delay(1000);
  }
  for (int k = 0; k < SND_FFT_SIZE / 2; k++) {
    float angle = -2.0f * (float)M_PI * k / SND_FFT_SIZE;
    mySndTwiddleCos[k] = cosf(angle);
    mySndTwiddleSin[k] = sinf(angle);
  }
}

// Compute power spectrum of `frame` (length n, must be power-of-2).
// Result written to `power[0..n/2]`  (n/2+1 bins).
// Uses mySndFftReal / mySndFftImag as scratch; caller must ensure they are allocated.
static void mySndPowerSpectrum(const float* frame, float* power, int n) {
  // Copy input into real array, zero imaginary
  for (int i = 0; i < n; i++) { mySndFftReal[i] = frame[i]; mySndFftImag[i] = 0.0f; }

  // --- Bit-reversal permutation ---
  int half = n >> 1;
  for (int i = 1, j = 0; i < n; i++) {
    int bit = half;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      float tr = mySndFftReal[i]; mySndFftReal[i] = mySndFftReal[j]; mySndFftReal[j] = tr;
      float ti = mySndFftImag[i]; mySndFftImag[i] = mySndFftImag[j]; mySndFftImag[j] = ti;
    }
  }

  // --- Cooley-Tukey butterfly stages ---
  for (int len = 2; len <= n; len <<= 1) {
    int half_len = len >> 1;
    int step = n / len;   // step into twiddle table
    for (int i = 0; i < n; i += len) {
      for (int k = 0; k < half_len; k++) {
        int tw = k * step;
        float wr = mySndTwiddleCos[tw];
        float wi = mySndTwiddleSin[tw];
        float ur = mySndFftReal[i + k];
        float ui = mySndFftImag[i + k];
        float vr = mySndFftReal[i + k + half_len] * wr - mySndFftImag[i + k + half_len] * wi;
        float vi = mySndFftReal[i + k + half_len] * wi + mySndFftImag[i + k + half_len] * wr;
        mySndFftReal[i + k]           = ur + vr;
        mySndFftImag[i + k]           = ui + vi;
        mySndFftReal[i + k + half_len] = ur - vr;
        mySndFftImag[i + k + half_len] = ui - vi;
      }
    }
  }

  // --- Power spectrum (magnitude squared) for bins 0..n/2 ---
  int bins = n / 2 + 1;
  for (int k = 0; k < bins; k++)
    power[k] = mySndFftReal[k] * mySndFftReal[k] + mySndFftImag[k] * mySndFftImag[k];
}

// ======================================================
// SPECTROGRAM  (mySndAudioBuffer -> mySndInputBuffer, normalised 0-1)
//
// v8 normalization fix:
//   Old code tracked globalMax initialised to 1e-8. Since real audio
//   log10 energies are negative (-8 to -1), globalMax stayed at ~0 and
//   the normalization range was always ~8, regardless of actual clip
//   content. Silence clips have a true max well below 0, so almost all
//   values were mapped to ≈1.0 — constant grey images.
//   Fix: two-pass — first pass stores raw log energies and finds the
//   true per-clip [min, max]; second pass normalises to [0, 1].
//   Range is floored at 1.0 so a pure-silence clip still produces a
//   meaningful (all-zero) image rather than dividing by ~0.
// ======================================================
bool mySndComputeSpectrogram() {
  if (!mySndAudioBuffer || !mySndInputBuffer || !mySndMelFbCtr) return false;
  if (!mySndSpectroFrame || !mySndSpectroPower)              return false;

  // Pass 1: compute raw log-mel energies into mySndInputBuffer
  float clipMin =  1e9f;
  float clipMax = -1e9f;

  for (int fr = 0; fr < SND_MEL_FRAMES; fr++) {
    int startSample = fr * SND_FRAME_SIZE;
    memset(mySndSpectroFrame, 0, SND_FFT_SIZE * sizeof(float));
    for (int i = 0; i < SND_FRAME_SIZE && i < SND_FFT_SIZE; i++) {
      float hann = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (SND_FRAME_SIZE - 1)));
      mySndSpectroFrame[i] = (mySndAudioBuffer[startSample + i] / 32768.0f) * hann;
    }
    mySndPowerSpectrum(mySndSpectroFrame, mySndSpectroPower, SND_FFT_SIZE);

    for (int m = 0; m < SND_MEL_BINS; m++) {
      int loB  = (int)mySndMelFbLo[m];
      int ctrB = (int)mySndMelFbCtr[m];
      int hiB  = (int)min((float)(SND_FFT_SIZE / 2), mySndMelFbHi[m]);
      float energy = 0;
      for (int b = loB; b < ctrB && b <= SND_FFT_SIZE / 2; b++) {
        float w = (b - mySndMelFbLo[m]) / (mySndMelFbCtr[m] - mySndMelFbLo[m] + 1e-8f);
        energy += w * mySndSpectroPower[b];
      }
      for (int b = ctrB; b < hiB && b <= SND_FFT_SIZE / 2; b++) {
        float w = (mySndMelFbHi[m] - b) / (mySndMelFbHi[m] - mySndMelFbCtr[m] + 1e-8f);
        energy += w * mySndSpectroPower[b];
      }
      float logE = log10f(energy + 1e-6f);   // +1e-6 avoids -inf on silence
      mySndInputBuffer[fr * SND_MEL_BINS + m] = logE;
      if (logE < clipMin) clipMin = logE;
      if (logE > clipMax) clipMax = logE;
    }

    // With the FFT the spectrogram is ~35x faster; vTaskDelay(0) after
    // each frame still keeps the WDT happy with negligible overhead.
    vTaskDelay(0);
  }

  // Pass 2: normalise using true per-clip range
  float range = clipMax - clipMin;
  if (range < 1.0f) range = 1.0f;   // floor: silence clips map to all-zero
  for (int i = 0; i < SND_MEL_FRAMES * SND_MEL_BINS; i++)
    mySndInputBuffer[i] = constrain((mySndInputBuffer[i] - clipMin) / range, 0.0f, 1.0f);

  return true;
}

bool mySndLoadClipAndComputeSpectrogram(const char* path) {
  return mySndLoadClipFromSD(path) && mySndComputeSpectrogram();
}

// ======================================================
// MEMORY ALLOCATION (PSRAM) — every allocation checked
// ======================================================
// Helper macro so failures name themselves
#define SND_MY_PSRAM_MALLOC(ptr, type, count, label)                        \
  ptr = (type*)ps_malloc((count) * sizeof(type));                       \
  if (!ptr) { Serial.println("PSRAM fail: " label); mySndPsramFailed=true; }

#define SND_MY_PSRAM_CALLOC(ptr, type, count, label)                        \
  ptr = (type*)ps_calloc((count), sizeof(type));                        \
  if (!ptr) { Serial.println("PSRAM fail: " label); mySndPsramFailed=true; }

void mySndAllocateMemory() {
  if (mySndInputBuffer) return;
  Serial.println("\n=== Allocating PSRAM ===");
  bool mySndPsramFailed = false;

  SND_MY_PSRAM_MALLOC(mySndAudioBuffer,  int16_t, SND_CLIP_SAMPLES,           "audioBuffer")
  SND_MY_PSRAM_MALLOC(mySndInputBuffer,  float,   SND_MEL_FRAMES * SND_MEL_BINS,  "inputBuffer")
  SND_MY_PSRAM_MALLOC(mySndSpectroFrame, float,   SND_FFT_SIZE,               "spectroFrame")
  SND_MY_PSRAM_MALLOC(mySndSpectroPower, float,   SND_FFT_SIZE / 2 + 1,       "spectroPower")
  SND_MY_PSRAM_MALLOC(mySndFftReal,      float,   SND_FFT_SIZE,               "fftReal")      // FFT scratch
  SND_MY_PSRAM_MALLOC(mySndFftImag,      float,   SND_FFT_SIZE,               "fftImag")      // FFT scratch

  SND_MY_PSRAM_MALLOC(mySndConv1_w,  float, SND_CONV1_WEIGHTS,  "conv1_w")
  SND_MY_PSRAM_MALLOC(mySndConv1_b,  float, SND_CONV1_FILTERS,  "conv1_b")
  SND_MY_PSRAM_MALLOC(mySndConv2_w,  float, SND_CONV2_WEIGHTS,  "conv2_w")
  SND_MY_PSRAM_MALLOC(mySndConv2_b,  float, SND_CONV2_FILTERS,  "conv2_b")
  SND_MY_PSRAM_MALLOC(mySndOutput_w, float, SND_OUTPUT_WEIGHTS,  "output_w")
  SND_MY_PSRAM_MALLOC(mySndOutput_b, float, SND_NUM_CLASSES,     "output_b")

  SND_MY_PSRAM_MALLOC(mySndConv1_w_grad,  float, SND_CONV1_WEIGHTS,  "conv1_w_grad")
  SND_MY_PSRAM_MALLOC(mySndConv1_b_grad,  float, SND_CONV1_FILTERS,  "conv1_b_grad")
  SND_MY_PSRAM_MALLOC(mySndConv2_w_grad,  float, SND_CONV2_WEIGHTS,  "conv2_w_grad")
  SND_MY_PSRAM_MALLOC(mySndConv2_b_grad,  float, SND_CONV2_FILTERS,  "conv2_b_grad")
  SND_MY_PSRAM_MALLOC(mySndOutput_w_grad, float, SND_OUTPUT_WEIGHTS,  "output_w_grad")
  SND_MY_PSRAM_MALLOC(mySndOutput_b_grad, float, SND_NUM_CLASSES,     "output_b_grad")

  SND_MY_PSRAM_CALLOC(mySndConv1_w_m,  float, SND_CONV1_WEIGHTS,  "conv1_w_m")
  SND_MY_PSRAM_CALLOC(mySndConv1_w_v,  float, SND_CONV1_WEIGHTS,  "conv1_w_v")
  SND_MY_PSRAM_CALLOC(mySndConv1_b_m,  float, SND_CONV1_FILTERS,  "conv1_b_m")
  SND_MY_PSRAM_CALLOC(mySndConv1_b_v,  float, SND_CONV1_FILTERS,  "conv1_b_v")
  SND_MY_PSRAM_CALLOC(mySndConv2_w_m,  float, SND_CONV2_WEIGHTS,  "conv2_w_m")
  SND_MY_PSRAM_CALLOC(mySndConv2_w_v,  float, SND_CONV2_WEIGHTS,  "conv2_w_v")
  SND_MY_PSRAM_CALLOC(mySndConv2_b_m,  float, SND_CONV2_FILTERS,  "conv2_b_m")
  SND_MY_PSRAM_CALLOC(mySndConv2_b_v,  float, SND_CONV2_FILTERS,  "conv2_b_v")
  SND_MY_PSRAM_CALLOC(mySndOutput_w_m, float, SND_OUTPUT_WEIGHTS,  "output_w_m")
  SND_MY_PSRAM_CALLOC(mySndOutput_w_v, float, SND_OUTPUT_WEIGHTS,  "output_w_v")
  SND_MY_PSRAM_CALLOC(mySndOutput_b_m, float, SND_NUM_CLASSES,     "output_b_m")
  SND_MY_PSRAM_CALLOC(mySndOutput_b_v, float, SND_NUM_CLASSES,     "output_b_v")

  SND_MY_PSRAM_MALLOC(mySndConv1_output, float, SND_CONV1_OUT_H * SND_CONV1_OUT_W * SND_CONV1_FILTERS, "conv1_out")
  SND_MY_PSRAM_MALLOC(mySndPool1_output, float, SND_POOL1_OUT_H * SND_POOL1_OUT_W * SND_CONV1_FILTERS, "pool1_out")
  mySndPool1_argmax = (uint8_t*)ps_malloc(SND_POOL1_OUT_H * SND_POOL1_OUT_W * SND_CONV1_FILTERS * sizeof(uint8_t));
  if (!mySndPool1_argmax) { Serial.println("PSRAM fail: pool1_argmax"); mySndPsramFailed = true; }
  SND_MY_PSRAM_MALLOC(mySndConv2_output, float, SND_CONV2_OUT_H * SND_CONV2_OUT_W * SND_CONV2_FILTERS, "conv2_out")
  SND_MY_PSRAM_MALLOC(mySndDense_output, float, SND_NUM_CLASSES,  "dense_out")

  SND_MY_PSRAM_MALLOC(mySndDense_grad, float, SND_FLATTENED_SIZE,                              "dense_grad")
  SND_MY_PSRAM_MALLOC(mySndConv2_grad, float, SND_CONV2_OUT_H * SND_CONV2_OUT_W * SND_CONV2_FILTERS,  "conv2_grad")
  SND_MY_PSRAM_MALLOC(mySndPool1_grad, float, SND_POOL1_OUT_H * SND_POOL1_OUT_W * SND_CONV1_FILTERS,  "pool1_grad")
  SND_MY_PSRAM_MALLOC(mySndConv1_grad, float, SND_CONV1_OUT_H * SND_CONV1_OUT_W * SND_CONV1_FILTERS,  "conv1_grad")

  if (mySndPsramFailed) {
    Serial.println("FATAL: PSRAM allocation failed!");
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "PSRAM ERROR!"); } while (u8g2.nextPage());
    while (1) delay(1000);
  }
  Serial.printf("Free PSRAM after alloc: %d bytes\n", ESP.getFreePsram());

  // He initialisation
  float c1std = sqrtf(2.0f / (9.0f * SND_INPUT_CH));
  for (int i = 0; i < SND_CONV1_WEIGHTS; i++)
    mySndConv1_w[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * c1std;
  for (int i = 0; i < SND_CONV1_FILTERS; i++) mySndConv1_b[i] = 0;

  float c2std = sqrtf(2.0f / (9.0f * SND_CONV1_FILTERS));
  for (int i = 0; i < SND_CONV2_WEIGHTS; i++)
    mySndConv2_w[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * c2std;
  for (int i = 0; i < SND_CONV2_FILTERS; i++) mySndConv2_b[i] = 0;

  float dstd = sqrtf(2.0f / SND_FLATTENED_SIZE);
  for (int i = 0; i < SND_OUTPUT_WEIGHTS; i++)
    mySndOutput_w[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * dstd;
  for (int i = 0; i < SND_NUM_CLASSES; i++) mySndOutput_b[i] = 0;

  Serial.println("He-init complete");
}

// ======================================================
// WEIGHT SAVE / LOAD / EXPORT
// ======================================================
void mySndExportHeader() {
  if (!mySndSDavailable) return;
  if (!SD.exists("/header")) SD.mkdir("/header");
  File file = SD.open("/header/mySndWeights.h", FILE_WRITE);
  if (!file) return;
  file.println("#ifndef MY_MODEL_H\n#define MY_MODEL_H");
  file.printf("// SND_NUM_CLASSES %d\n// Labels:", SND_NUM_CLASSES);
  for (int i = 0; i < SND_NUM_CLASSES; i++) file.printf(" \"%s\"", mySndClassLabels[i].c_str());
  file.println("\n// Uncomment #define USE_BAKED_WEIGHTS in main sketch to use these.");
  auto mySndDump = [&](const char* name, float* data, int size) {
    file.printf("const float %s[] = {", name);
    for (int i = 0; i < size; i++) {
      file.print(data[i], 6); file.print("f");
      if (i < size - 1) file.print(",");
      if ((i + 1) % 8 == 0) file.println();
    }
    file.println("};");
  };
  mySndDump("mySndModel_conv1_w",  mySndConv1_w,  SND_CONV1_WEIGHTS);
  mySndDump("mySndModel_conv1_b",  mySndConv1_b,  SND_CONV1_FILTERS);
  mySndDump("mySndModel_conv2_w",  mySndConv2_w,  SND_CONV2_WEIGHTS);
  mySndDump("mySndModel_conv2_b",  mySndConv2_b,  SND_CONV2_FILTERS);
  mySndDump("mySndModel_output_w", mySndOutput_w, SND_OUTPUT_WEIGHTS);
  mySndDump("mySndModel_output_b", mySndOutput_b, SND_NUM_CLASSES);
  file.println("#endif");
  file.close();
  Serial.println("Header exported: /header/mySndWeights.h");
}

bool mySndLoadWeights() {
  if (!mySndSDavailable || !SD.exists("/header/mySndWeights.bin")) return false;
  File f = SD.open("/header/mySndWeights.bin", FILE_READ);
  if (!f) return false;
  f.read((uint8_t*)mySndConv1_w,  SND_CONV1_WEIGHTS  * 4);
  f.read((uint8_t*)mySndConv1_b,  SND_CONV1_FILTERS  * 4);
  f.read((uint8_t*)mySndConv2_w,  SND_CONV2_WEIGHTS  * 4);
  f.read((uint8_t*)mySndConv2_b,  SND_CONV2_FILTERS  * 4);
  f.read((uint8_t*)mySndOutput_w, SND_OUTPUT_WEIGHTS * 4);
  f.read((uint8_t*)mySndOutput_b, SND_NUM_CLASSES    * 4);
  f.close();
  mySndWeightsTrained = true;
  Serial.println("Weights loaded from SD");
  return true;
}

void mySndSaveWeights() {
  if (!mySndSDavailable) return;
  if (!SD.exists("/header")) SD.mkdir("/header");
  File f = SD.open("/header/mySndWeights.bin", FILE_WRITE);
  if (f) {
    f.write((uint8_t*)mySndConv1_w,  SND_CONV1_WEIGHTS  * 4);
    f.write((uint8_t*)mySndConv1_b,  SND_CONV1_FILTERS  * 4);
    f.write((uint8_t*)mySndConv2_w,  SND_CONV2_WEIGHTS  * 4);
    f.write((uint8_t*)mySndConv2_b,  SND_CONV2_FILTERS  * 4);
    f.write((uint8_t*)mySndOutput_w, SND_OUTPUT_WEIGHTS * 4);
    f.write((uint8_t*)mySndOutput_b, SND_NUM_CLASSES    * 4);
    f.close();
    Serial.println("Weights saved");
  }
  mySndExportHeader();
}

// ██████████████████████████████████████████████████████████████████████████████
// ██  FORWARD / BACKWARD PASS + OPTIMIZER                                      ██
// ██████████████████████████████████████████████████████████████████████████████

void mySndForwardPass(float* input, float* logits) {
  // Conv1 (1-channel input)
  for (int f = 0; f < SND_CONV1_FILTERS; f++) {
    int ob = f * SND_CONV1_OUT_H * SND_CONV1_OUT_W;
    for (int y = 0; y < SND_CONV1_OUT_H; y++) {
      for (int x = 0; x < SND_CONV1_OUT_W; x++) {
        float sum = 0;
        for (int ky = 0; ky < 3; ky++)
          for (int kx = 0; kx < 3; kx++)
            sum += input[(y + ky) * SND_INPUT_W + (x + kx)] * mySndConv1_w[f * 9 + ky * 3 + kx];
        mySndConv1_output[ob + y * SND_CONV1_OUT_W + x] = Sndleaky_relu(Sndclip_value(sum + mySndConv1_b[f]));
      }
    }
  }
  // Pool1 (2x2 max) — records which cell was max in mySndPool1_argmax
  // Offsets: 0=(iy,ix)  1=(iy,ix+1)  2=(iy+1,ix)  3=(iy+1,ix+1)
  for (int f = 0; f < SND_CONV1_FILTERS; f++) {
    int ib = f * SND_CONV1_OUT_H * SND_CONV1_OUT_W;
    int ob = f * SND_POOL1_OUT_H * SND_POOL1_OUT_W;
    for (int y = 0; y < SND_POOL1_OUT_H; y++) {
      for (int x = 0; x < SND_POOL1_OUT_W; x++) {
        int iy = y * 2, ix = x * 2;
        float v0 = mySndConv1_output[ib +  iy      * SND_CONV1_OUT_W + ix    ];
        float v1 = mySndConv1_output[ib +  iy      * SND_CONV1_OUT_W + ix + 1];
        float v2 = mySndConv1_output[ib + (iy + 1) * SND_CONV1_OUT_W + ix    ];
        float v3 = mySndConv1_output[ib + (iy + 1) * SND_CONV1_OUT_W + ix + 1];
        float mv = v0; uint8_t argmax = 0;
        if (v1 > mv) { mv = v1; argmax = 1; }
        if (v2 > mv) { mv = v2; argmax = 2; }
        if (v3 > mv) { mv = v3; argmax = 3; }
        mySndPool1_output[ob + y * SND_POOL1_OUT_W + x] = mv;
        mySndPool1_argmax[ob + y * SND_POOL1_OUT_W + x] = argmax;
      }
    }
  }
  // Conv2
  for (int f = 0; f < SND_CONV2_FILTERS; f++) {
    int ob = f * SND_CONV2_OUT_H * SND_CONV2_OUT_W;
    for (int y = 0; y < SND_CONV2_OUT_H; y++) {
      for (int x = 0; x < SND_CONV2_OUT_W; x++) {
        float sum = 0;
        for (int c = 0; c < SND_CONV1_FILTERS; c++) {
          int ib = c * SND_POOL1_OUT_H * SND_POOL1_OUT_W;
          for (int ky = 0; ky < 3; ky++)
            for (int kx = 0; kx < 3; kx++)
              sum += mySndPool1_output[ib + (y + ky) * SND_POOL1_OUT_W + (x + kx)] *
                     mySndConv2_w[f * (SND_CONV1_FILTERS * 9) + c * 9 + ky * 3 + kx];
        }
        mySndConv2_output[ob + y * SND_CONV2_OUT_W + x] = Sndleaky_relu(Sndclip_value(sum + mySndConv2_b[f]));
      }
    }
  }
  // Dense + Softmax (Kahan summation for numerical stability)
  for (int c = 0; c < SND_NUM_CLASSES; c++) {
    double sum = 0, comp = 0;
    for (int i = 0; i < SND_FLATTENED_SIZE; i++) {
      double term = mySndConv2_output[i] * mySndOutput_w[c * SND_FLATTENED_SIZE + i];
      double yy = term - comp; double t = sum + yy;
      comp = (t - sum) - yy; sum = t;
    }
    mySndDense_output[c] = Sndclip_value((float)sum + mySndOutput_b[c], -50, 50);
  }
  float mx = mySndDense_output[0];
  for (int i = 1; i < SND_NUM_CLASSES; i++) mx = max(mx, mySndDense_output[i]);
  float expSum = 0;
  for (int i = 0; i < SND_NUM_CLASSES; i++) expSum += expf(mySndDense_output[i] - mx);
  for (int i = 0; i < SND_NUM_CLASSES; i++) {
    logits[i]         = mySndDense_output[i];
    mySndDense_output[i] = expf(mySndDense_output[i] - mx) / expSum;
  }
}

void mySndBackwardDense(int label) {
  memset(mySndDense_grad, 0, SND_FLATTENED_SIZE * sizeof(float));
  for (int c = 0; c < SND_NUM_CLASSES; c++) {
    float err = mySndDense_output[c] - (c == label ? 1.0f : 0.0f);
    for (int i = 0; i < SND_FLATTENED_SIZE; i++) {
      mySndOutput_w_grad[c * SND_FLATTENED_SIZE + i] += err * mySndConv2_output[i];
      mySndDense_grad[i] += err * mySndOutput_w[c * SND_FLATTENED_SIZE + i];
    }
    mySndOutput_b_grad[c] += err;
  }
}

void mySndBackwardConv2() {

/*
  for (int i = 0; i < SND_FLATTENED_SIZE; i++)
    mySndConv2_grad[i] = mySndDense_grad[i] * Sndleaky_relu_deriv(mySndConv2_output[i]);
*/
for (int f = 0; f < SND_CONV2_FILTERS; f++) {
    int base = f * SND_CONV2_OUT_H * SND_CONV2_OUT_W;
    for (int y = 0; y < SND_CONV2_OUT_H; y++) {
        for (int x = 0; x < SND_CONV2_OUT_W; x++) {
            int idx = base + y * SND_CONV2_OUT_W + x;
            mySndConv2_grad[idx] =
                mySndDense_grad[idx] * Sndleaky_relu_deriv(mySndConv2_output[idx]);
        }
    }
}


  memset(mySndPool1_grad, 0, SND_POOL1_OUT_H * SND_POOL1_OUT_W * SND_CONV1_FILTERS * sizeof(float));
  for (int f = 0; f < SND_CONV2_FILTERS; f++) {
    int ob = f * SND_CONV2_OUT_H * SND_CONV2_OUT_W;
    for (int y = 0; y < SND_CONV2_OUT_H; y++) {
      for (int x = 0; x < SND_CONV2_OUT_W; x++) {
        float grad = mySndConv2_grad[ob + y * SND_CONV2_OUT_W + x];
        mySndConv2_b_grad[f] += grad;
        for (int c = 0; c < SND_CONV1_FILTERS; c++) {
          int ib = c * SND_POOL1_OUT_H * SND_POOL1_OUT_W;
          for (int ky = 0; ky < 3; ky++) {
            for (int kx = 0; kx < 3; kx++) {
              int pi = ib + (y + ky) * SND_POOL1_OUT_W + (x + kx);
              int wi = f * (SND_CONV1_FILTERS * 9) + c * 9 + ky * 3 + kx;
              mySndConv2_w_grad[wi] += grad * mySndPool1_output[pi];
              mySndPool1_grad[pi]   += grad * mySndConv2_w[wi];
            }
          }
        }
      }
    }
  }
}

void mySndBackwardPool1() {
  // v8 fix: use mySndPool1_argmax recorded during the forward pass.
  // The old exact float equality (== poolVal) was fragile after Sndleaky_relu
  // and silently zeroed most Conv1 gradients, killing early-layer learning.
  memset(mySndConv1_grad, 0, SND_CONV1_OUT_H * SND_CONV1_OUT_W * SND_CONV1_FILTERS * sizeof(float));
  for (int f = 0; f < SND_CONV1_FILTERS; f++) {
    int ib = f * SND_CONV1_OUT_H * SND_CONV1_OUT_W;
    int ob = f * SND_POOL1_OUT_H * SND_POOL1_OUT_W;
    for (int y = 0; y < SND_POOL1_OUT_H; y++) {
      for (int x = 0; x < SND_POOL1_OUT_W; x++) {
        float grad   = mySndPool1_grad  [ob + y * SND_POOL1_OUT_W + x];
        uint8_t amax = mySndPool1_argmax[ob + y * SND_POOL1_OUT_W + x];
        int iy = y * 2, ix = x * 2;
        // Route gradient to exactly the cell that was the max
        int srcOffset;
        switch (amax) {
          case 0: srcOffset = ib +  iy      * SND_CONV1_OUT_W + ix;     break;
          case 1: srcOffset = ib +  iy      * SND_CONV1_OUT_W + ix + 1; break;
          case 2: srcOffset = ib + (iy + 1) * SND_CONV1_OUT_W + ix;     break;
          default:srcOffset = ib + (iy + 1) * SND_CONV1_OUT_W + ix + 1; break;
        }
        mySndConv1_grad[srcOffset] += grad;
      }
    }
  }
}

void mySndBackwardConv1() {
  for (int i = 0; i < SND_CONV1_OUT_H * SND_CONV1_OUT_W * SND_CONV1_FILTERS; i++)
    mySndConv1_grad[i] *= Sndleaky_relu_deriv(mySndConv1_output[i]);
  for (int f = 0; f < SND_CONV1_FILTERS; f++) {
    int ob = f * SND_CONV1_OUT_H * SND_CONV1_OUT_W;
    for (int y = 0; y < SND_CONV1_OUT_H; y++) {
      for (int x = 0; x < SND_CONV1_OUT_W; x++) {
        float grad = mySndConv1_grad[ob + y * SND_CONV1_OUT_W + x];
        mySndConv1_b_grad[f] += grad;
        for (int ky = 0; ky < 3; ky++)
          for (int kx = 0; kx < 3; kx++)
            mySndConv1_w_grad[f * 9 + ky * 3 + kx] +=
              grad * mySndInputBuffer[(y + ky) * SND_INPUT_W + (x + kx)];
      }
    }
  }
}

// ======================================================
// ADAM OPTIMIZER — v7 fixes:
//   1. Accepts a gradient scale factor (1/processed) for batch normalization.
//   2. Precomputes bias-correction factor outside the per-weight loop.
//   3. Gradient clipping applied before m/v update.
// ======================================================
void mySndAdamUpdate(float* w, float* g, float* m, float* v,
                  int size, int step, float gradScale) {
  const float b1  = 0.9f, b2 = 0.999f, eps = 1e-6f;
  // Precompute bias-corrected learning rate for this step
  float b1pow  = powf(b1, step);
  float b2pow  = powf(b2, step);
  float lr_t   = SND_LEARNING_RATE * sqrtf(1.0f - b2pow) / (1.0f - b1pow + 1e-12f);

  for (int i = 0; i < size; i++) {
    float gi = g[i] * gradScale;                             // normalize by batch
    gi = constrain(gi, -1.0f, 1.0f);                         // gradient clip
    m[i] = b1 * m[i] + (1.0f - b1) * gi;
    v[i] = b2 * v[i] + (1.0f - b2) * gi * gi;
    w[i] -= lr_t * m[i] / (sqrtf(v[i]) + eps);
    w[i]  = Sndclip_value(w[i], -10.0f, 10.0f);
  }
}

void mySndUpdateWeights(int step, float gradScale) {
  mySndAdamUpdate(mySndConv1_w,  mySndConv1_w_grad,  mySndConv1_w_m,  mySndConv1_w_v,  SND_CONV1_WEIGHTS,  step, gradScale);
  mySndAdamUpdate(mySndConv1_b,  mySndConv1_b_grad,  mySndConv1_b_m,  mySndConv1_b_v,  SND_CONV1_FILTERS,  step, gradScale);
  mySndAdamUpdate(mySndConv2_w,  mySndConv2_w_grad,  mySndConv2_w_m,  mySndConv2_w_v,  SND_CONV2_WEIGHTS,  step, gradScale);
  mySndAdamUpdate(mySndConv2_b,  mySndConv2_b_grad,  mySndConv2_b_m,  mySndConv2_b_v,  SND_CONV2_FILTERS,  step, gradScale);
  mySndAdamUpdate(mySndOutput_w, mySndOutput_w_grad, mySndOutput_w_m, mySndOutput_w_v, SND_OUTPUT_WEIGHTS, step, gradScale);
  mySndAdamUpdate(mySndOutput_b, mySndOutput_b_grad, mySndOutput_b_m, mySndOutput_b_v, SND_NUM_CLASSES,    step, gradScale);
}

// ██████████████████████████████████████████████████████████████████████████████
// ██  MENU SYSTEM                                                               ██
// ██████████████████████████████████████████████████████████████████████████████

void mySndResetMenuState();
void mySndDrawMenu();
void mySndActionCollect(int classIdx);
void mySndActionTrain();
void mySndActionInfer();

void mySndResetMenuState() {
  mySndIsSelected = false;
  mySndResetTouchState();
  mySndLastActivityTime = millis();
  // NOTE: mySndDrawMenu() call removed here — in the combined sketch the
  // unified myDrawMenu() (see COMBINED MENU section) redraws the OLED
  // after any action returns, so this model-local menu stays silent.
}

void mySndDrawMenu() {
  Serial.println("\n=== MENU ===");
  for (int i = 1; i <= mySndTotalItems; i++) {
    String label = (i <= SND_NUM_CLASSES) ? mySndClassLabels[i - 1] :
                   (i == SND_NUM_CLASSES + 1) ? "Train" : "Infer";
    Serial.printf("%s %d. %s\n", i == mySndMenuIndex ? " >" : "  ", i, label.c_str());
  }
  Serial.println("t=next  l=select  1-9=direct");

  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_6x10_tf);
    u8g2.drawStr(0, 8, "TAP:Next HOLD:Ok");
    int start = (mySndMenuIndex <= SND_NUM_CLASSES) ? 1 : mySndMenuIndex - 2;
    for (int i = 0; i < 3; i++) {
      int cur = start + i;
      if (cur > mySndTotalItems) break;
      String label = (cur <= SND_NUM_CLASSES) ? mySndClassLabels[cur - 1] :
                     (cur == SND_NUM_CLASSES + 1) ? "Train" : "Infer";
      u8g2.drawStr(0, 18 + i * 9,
                   (cur == mySndMenuIndex ? "> " + label : "  " + label).c_str());
    }
  } while (u8g2.nextPage());
}

void mySndExecuteMenuItem(int idx) {
  if (idx <= SND_NUM_CLASSES)           mySndActionCollect(idx - 1);
  else if (idx == SND_NUM_CLASSES + 1)  mySndActionTrain();
  else                              mySndActionInfer();
}

void mySndHandleMenuNavigation() {
  unsigned long now = millis();
  if (!mySndIsSelected && Serial.available()) {
    char c = Serial.read();
    if (c >= '1' && c <= '9') {
      int n = c - '0';
      if (n <= mySndTotalItems) {
        mySndMenuIndex = n; mySndIsSelected = true;
        mySndLastActivityTime = now; mySndExecuteMenuItem(mySndMenuIndex);
      }
    } else if (c == 't' || c == 'T') {
      if (now - mySndLastTapTime > mySndTapCooldown) {
        mySndMenuIndex = (mySndMenuIndex % mySndTotalItems) + 1;
        mySndDrawMenu(); mySndLastTapTime = now; mySndLastActivityTime = now;
      }
    } else if (c == 'l' || c == 'L') {
      mySndIsSelected = true; mySndLastActivityTime = now; mySndExecuteMenuItem(mySndMenuIndex);
    }
  }
  if (!mySndIsSelected) {
    int ta = mySndCheckTouchInput();
    if (ta == 1) {
      if (now - mySndLastTapTime > mySndTapCooldown) {
        mySndMenuIndex = (mySndMenuIndex % mySndTotalItems) + 1;
        mySndDrawMenu(); mySndLastTapTime = now; mySndLastActivityTime = now;
      }
    } else if (ta == 2) {
      mySndIsSelected = true; mySndLastActivityTime = now; mySndExecuteMenuItem(mySndMenuIndex);
    }
  }
}

// ██████████████████████████████████████████████████████████████████████████████
// ██  SETUP & LOOP                                                              ██
// ██████████████████████████████████████████████████████████████████████████████





// ██████████████████████████████████████████████████████████████████████████████
// ██  PART 1: AUDIO COLLECTION                                                  ██
// ██████████████████████████████████████████████████████████████████████████████

void mySndActionCollect(int classIdx) {
  if (!mySndSDavailable) {
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "No SD card"); } while (u8g2.nextPage());
    delay(2000); mySndResetMenuState(); return;
  }

  Serial.printf("\n>>> Collect: %s\n", mySndClassLabels[classIdx].c_str());
  Serial.println("  TAP=record  LONG PRESS=exit  L=exit");
  mySndResetTouchState();

  String folderPath = "/audio/" + mySndClassLabels[classIdx];
  if (!SD.exists("/audio"))   SD.mkdir("/audio");
  if (!SD.exists(folderPath)) SD.mkdir(folderPath);

  // Count existing clips
  int clipCount = 0;
  {
    File root = SD.open(folderPath);
    if (root) {
      while (File f = root.openNextFile()) {
        if (!f.isDirectory() && String(f.name()).endsWith(".wav")) clipCount++;
        f.close();
      }
      root.close();
    }
  }

  auto mySndShowIdle = [&]() {
    u8g2.firstPage();
    do {
      u8g2.setFont(u8g2_font_6x10_tf);
      u8g2.drawStr(0, 10, mySndClassLabels[classIdx].c_str());
      char buf[20]; snprintf(buf, sizeof(buf), "Clips: %d", clipCount);
      u8g2.drawStr(0, 22, buf);
      u8g2.drawStr(0, 34, "TAP to record");
    } while (u8g2.nextPage());
  };
  mySndShowIdle();

  bool shouldRecord = false;

  while (true) {
    if (Serial.available()) {
      char c = Serial.read();
      if      (c == 'l' || c == 'L') { mySndResetMenuState(); return; }
      else if (c == 't' || c == 'T') shouldRecord = true;
    }
    int ta = mySndCheckTouchInput();
    if      (ta == 2) { mySndResetMenuState(); return; }
    else if (ta == 1) shouldRecord = true;

    if (shouldRecord) {
      shouldRecord = false;

      u8g2.firstPage();
      do {
        u8g2.setFont(u8g2_font_6x10_tf);
        u8g2.drawStr(0, 12, "GET READY...");
        u8g2.drawStr(0, 26, "Recording in 1s");
      } while (u8g2.nextPage());
      delay(800);

      u8g2.firstPage();
      do {
        u8g2.setFont(u8g2_font_6x10_tf);
        u8g2.drawStr(0, 15, "** RECORDING **");
        u8g2.drawStr(0, 28, "1 second...");
      } while (u8g2.nextPage());

      String fileName = folderPath + "/clip_" + String(millis()) + ".wav";
      if (mySndSaveClipToSD(fileName)) {
        clipCount++;
        Serial.printf("Saved: %s  (total: %d)\n", fileName.c_str(), clipCount);
      } else {
        Serial.println("ERROR: save failed");
        u8g2.firstPage();
        do { u8g2.drawStr(0, 15, "SAVE FAILED"); } while (u8g2.nextPage());
        delay(1000);
      }

      mySndShowIdle();
    }
    delay(10);
  }
}

// ██████████████████████████████████████████████████████████████████████████████
// ██  PART 2: TRAINING                                                          ██
// ██████████████████████████████████████████████████████████████████████████████
//
// v7 key changes vs v6:
//   - Gradient scale = 1 / processed clips per batch (fixes the biggest bug).
//   - Adam step counter increments only on actual updates (not batch index).
//   - Serial prints every epoch: Ep N/20  Loss:x.xx  Acc:xx%  Val:xx%
//   - OLED updates every epoch showing: epoch/total, loss, acc, free PSRAM.
//   - OLED also shows a mini progress bar for the full training run.
//   - Validation split skips hold-out if class has too few clips.
//   - NaN/Inf check after loss; corrupt samples skipped cleanly.
//   - Watchdog yields inside clip loading loop.

void mySndActionTrain() {
  if (!mySndSDavailable) {
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "No SD card"); } while (u8g2.nextPage());
    delay(2000); mySndResetMenuState(); return;
  }

  Serial.println("\n>>> Training  (L or 3+ taps = save & exit)");
  mySndResetTouchState();

  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_6x10_tf);
    u8g2.drawStr(0, 15, "TRAINING MODE");
    u8g2.drawStr(0, 28, "Scanning SD...");
  } while (u8g2.nextPage());

  if (mySndLoadWeights()) Serial.println("Continuing from saved weights");
  else                 Serial.println("Starting fresh (He-init)");

  while (true) {
    // ---- Scan SD — build FULL paths explicitly ----
    mySndTrainingData.clear();
    int classCounts[SND_NUM_CLASSES] = {};

    for (int i = 0; i < SND_NUM_CLASSES; i++) {
      String folderPath = "/audio/" + mySndClassLabels[i];
      File root = SD.open(folderPath);
      if (root) {
        while (File f = root.openNextFile()) {
          if (!f.isDirectory() && String(f.name()).endsWith(".wav")) {
            String fname = String(f.name());
            if (fname.startsWith("/")) fname = fname.substring(fname.lastIndexOf("/") + 1);
            String fullPath = folderPath + "/" + fname;
            mySndTrainingData.push_back({fullPath, i});
            classCounts[i]++;
          }
          f.close();
        }
        root.close();
      }
    }

    // Report scan results
    Serial.println("--- Clip scan ---");
    for (int i = 0; i < SND_NUM_CLASSES; i++)
      Serial.printf("  %s: %d clips\n", mySndClassLabels[i].c_str(), classCounts[i]);

    u8g2.firstPage();
    do {
      u8g2.setFont(u8g2_font_5x7_tf);
      u8g2.drawStr(0, 8, "Clips found:");
      for (int i = 0; i < min(SND_NUM_CLASSES, 3); i++) {
        char buf[24];
        snprintf(buf, sizeof(buf), " %s:%d", mySndClassLabels[i].c_str(), classCounts[i]);
        u8g2.drawStr(0, 16 + i * 9, buf);
      }
    } while (u8g2.nextPage());
    delay(1500);

    if (mySndTrainingData.empty()) {
      u8g2.firstPage();
      do { u8g2.drawStr(0, 20, "No clips found!"); } while (u8g2.nextPage());
      delay(2000); mySndResetMenuState(); return;
    }

    // Sort for deterministic validation split
    std::sort(mySndTrainingData.begin(), mySndTrainingData.end(),
              [](const SndTrainingItem& a, const SndTrainingItem& b) { return a.path < b.path; });

    // ---- Validation split — safe for small datasets ----
    // Each class keeps at least 1 clip for training.
    // If a class has <= VALIDATION_CLIPS clips, 0 are held out (warn).
    std::vector<SndTrainingItem> mySndValidationData;
    {
      int counts[SND_NUM_CLASSES] = {};
      for (auto& item : mySndTrainingData) counts[item.label]++;

      int holdOut[SND_NUM_CLASSES];
      for (int c = 0; c < SND_NUM_CLASSES; c++) {
        if (counts[c] <= VALIDATION_CLIPS + 1) {
          holdOut[c] = 0;   // not enough clips — keep all for training
          Serial.printf("  WARN: %s has only %d clips, skipping val hold-out\n",
                        mySndClassLabels[c].c_str(), counts[c]);
        } else {
          holdOut[c] = VALIDATION_CLIPS;
        }
      }

      std::vector<SndTrainingItem> trainOnly;
      int seen[SND_NUM_CLASSES] = {};
      for (int i = (int)mySndTrainingData.size() - 1; i >= 0; i--) {
        int c = mySndTrainingData[i].label;
        if (seen[c] < holdOut[c]) { mySndValidationData.push_back(mySndTrainingData[i]); seen[c]++; }
        else                       trainOnly.push_back(mySndTrainingData[i]);
      }
      mySndTrainingData = trainOnly;
    }

    int total           = (int)mySndTrainingData.size();
    int batchesPerEpoch = max(1, (total + SND_BATCH_SIZE - 1) / SND_BATCH_SIZE);

    // Safety check — can't train with nothing
    if (total == 0) {
      Serial.println("ERROR: No training clips after validation split!");
      u8g2.firstPage();
      do { u8g2.drawStr(0, 20, "No train clips!"); } while (u8g2.nextPage());
      delay(2000); mySndResetMenuState(); return;
    }

    Serial.printf("Train: %d  Val: %d  BatchSz: %d  Epochs: %d\n",
                  total, (int)mySndValidationData.size(), SND_BATCH_SIZE, SND_TARGET_EPOCHS);
    // Rough timing: each clip takes ~4 s for the O(N^2) spectrogram
    float secsPerEpoch = total * 0.05f;   // FFT: ~50 ms/clip vs ~1.5 s for old DFT
    Serial.printf("Estimated: ~%.0f s/epoch  (~%.0f min total)\n",
                  secsPerEpoch, secsPerEpoch * SND_TARGET_EPOCHS / 60.0f);

    std::vector<int> indices;
    indices.reserve(total);
    for (int i = 0; i < total; i++) indices.push_back(i);

    int adamStep = 0;   // tracks actual gradient updates for Adam bias correction

    for (int epoch = 0; epoch < SND_TARGET_EPOCHS; epoch++) {

      // Shuffle indices at epoch start
      for (int i = total - 1; i > 0; i--) {
        int j = random(i + 1);
        int tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
      }

      // Print epoch header to serial
      Serial.printf("\n--- Epoch %d/%d ---\n", epoch + 1, SND_TARGET_EPOCHS);
      Serial.printf("  Free heap: %d  Free PSRAM: %d\n",
                    ESP.getFreeHeap(), ESP.getFreePsram());

      float epochLoss   = 0;
      int   epochCorrect = 0;
      int   epochSamples = 0;
      int   loadFails    = 0;

      for (int b = 0; b < batchesPerEpoch; b++) {
        // Exit check
        if (Serial.available()) {
          char c = Serial.read();
          if (c == 'l' || c == 'L' || c == 'x' || c == 'X') {
            mySndSaveWeights(); mySndWeightsTrained = true; mySndResetMenuState(); return;
          }
        }
        mySndCheckTouchBackground();
        if (mySndPeekTouchAction() == 2) {
          mySndCheckTouchInput();
          mySndSaveWeights(); mySndWeightsTrained = true; mySndResetMenuState(); return;
        }

        int batchStart = b * SND_BATCH_SIZE;
        int batchEnd   = min(batchStart + SND_BATCH_SIZE, total);

        // Zero gradients
        memset(mySndConv1_w_grad,  0, SND_CONV1_WEIGHTS  * sizeof(float));
        memset(mySndConv1_b_grad,  0, SND_CONV1_FILTERS  * sizeof(float));
        memset(mySndConv2_w_grad,  0, SND_CONV2_WEIGHTS  * sizeof(float));
        memset(mySndConv2_b_grad,  0, SND_CONV2_FILTERS  * sizeof(float));
        memset(mySndOutput_w_grad, 0, SND_OUTPUT_WEIGHTS * sizeof(float));
        memset(mySndOutput_b_grad, 0, SND_NUM_CLASSES    * sizeof(float));

        int processed = 0;
        float batchLoss = 0;
        int   batchCorrect = 0;
        int   batchLoadFails = 0;

        for (int i = batchStart; i < batchEnd; i++) {
          SndTrainingItem& clip = mySndTrainingData[indices[i]];

          if (!mySndLoadClipAndComputeSpectrogram(clip.path.c_str())) {
            loadFails++; batchLoadFails++;
            Serial.printf("  LOAD FAIL [%d]: %s\n", i, clip.path.c_str());
            continue;
          }

          float logits[SND_NUM_CLASSES];
          mySndForwardPass(mySndInputBuffer, logits);

          float prob = max(mySndDense_output[clip.label], 1e-7f);
          float sampleLoss = -logf(prob);

          // NaN/Inf guard — skip this sample
          if (isnan(sampleLoss) || isinf(sampleLoss)) {
            Serial.println("  WARN: NaN/Inf loss — skipping sample");
            continue;
          }

          batchLoss += sampleLoss;

          int pred = 0;
          for (int j = 1; j < SND_NUM_CLASSES; j++)
            if (mySndDense_output[j] > mySndDense_output[pred]) pred = j;
          if (pred == clip.label) batchCorrect++;

          mySndBackwardDense(clip.label);
          mySndBackwardConv2();
          mySndBackwardPool1();
          mySndBackwardConv1();

          processed++;

          // Periodic exit check inside batch
          if (i % 3 == 0) {
            mySndCheckTouchBackground();
            if (mySndPeekTouchAction() == 2) {
              mySndCheckTouchInput();
              mySndSaveWeights(); mySndWeightsTrained = true; mySndResetMenuState(); return;
            }
            if (Serial.available()) {
              char c = Serial.read();
              if (c == 'l' || c == 'L' || c == 'x' || c == 'X') {
                mySndSaveWeights(); mySndWeightsTrained = true; mySndResetMenuState(); return;
              }
            }
          }

          yield();   // watchdog feed inside clip loop
        }

        if (processed > 0) {
          adamStep++;
          float gradScale = 1.0f / processed;   // normalize gradients by batch size
          mySndUpdateWeights(adamStep, gradScale);

          epochLoss    += batchLoss / processed;
          epochCorrect += batchCorrect;
          epochSamples += processed;
        }

        if (batchLoadFails > 0)
          Serial.printf("  Batch %d: %d load fail(s)\n", b + 1, batchLoadFails);
      }

      // ---- Per-epoch summary ----
      float avgLoss = epochSamples > 0 ? epochLoss / batchesPerEpoch : 0;
      float avgAcc  = epochSamples > 0 ? 100.0f * epochCorrect / epochSamples : 0;

      // Validation pass
      float valAcc = -1;
      if (!mySndValidationData.empty()) {
        int valCorrect = 0, valCount = 0;
        for (auto& vitem : mySndValidationData) {
          if (!mySndLoadClipAndComputeSpectrogram(vitem.path.c_str())) continue;
          float logits[SND_NUM_CLASSES];
          mySndForwardPass(mySndInputBuffer, logits);
          int pred = 0;
          for (int j = 1; j < SND_NUM_CLASSES; j++)
            if (mySndDense_output[j] > mySndDense_output[pred]) pred = j;
          if (pred == vitem.label) valCorrect++;
          valCount++;
          yield();
        }
        if (valCount > 0) valAcc = 100.0f * valCorrect / valCount;
      }

      // Serial epoch summary — always printed
      if (valAcc >= 0)
        Serial.printf("Ep %d/%d  Loss:%.4f  Acc:%.1f%%  Val:%.1f%%\n",
                      epoch + 1, SND_TARGET_EPOCHS, avgLoss, avgAcc, valAcc);
      else
        Serial.printf("Ep %d/%d  Loss:%.4f  Acc:%.1f%%\n",
                      epoch + 1, SND_TARGET_EPOCHS, avgLoss, avgAcc);

      // OLED epoch summary — always updated
      {
        int oledW  = u8g2.getDisplayWidth();   // 72
        int barFull = (int)((float)(epoch + 1) / SND_TARGET_EPOCHS * oledW);

        u8g2.firstPage();
        do {
          u8g2.setFont(u8g2_font_5x7_tf);

          // Line 1: Ep N/20
          char line1[24];
          snprintf(line1, sizeof(line1), "Ep%d/%d", epoch + 1, SND_TARGET_EPOCHS);
          u8g2.drawStr(0, 8, line1);

          // Line 2: Loss
          char line2[24];
          snprintf(line2, sizeof(line2), "Loss:%.3f", avgLoss);
          u8g2.drawStr(0, 17, line2);

          // Line 3: Train acc  (+ val if available)
          char line3[24];
          if (valAcc >= 0)
            snprintf(line3, sizeof(line3), "A:%.0f%% V:%.0f%%", avgAcc, valAcc);
          else
            snprintf(line3, sizeof(line3), "Acc: %.1f%%", avgAcc);
          u8g2.drawStr(0, 26, line3);

          // Line 4: progress bar (full width, 3 px tall)
          u8g2.drawFrame(0, 30, oledW, 6);
          if (barFull > 0) u8g2.drawBox(0, 30, barFull, 6);

        } while (u8g2.nextPage());
      }
    } // end epoch loop

    Serial.println("\n--- Training complete ---");
    mySndSaveWeights();
    mySndWeightsTrained = true;

    u8g2.firstPage();
    do {
      u8g2.drawStr(0, 12, "DONE!");
      u8g2.drawStr(0, 24, "Tap: train again");
      u8g2.drawStr(0, 36, "3+:  exit");
    } while (u8g2.nextPage());

    mySndResetTouchState();
    Serial.println("T=train again  L=exit");
    while (true) {
      if (Serial.available()) {
        char c = Serial.read();
        if (c == 'l' || c == 'L' || c == 'x' || c == 'X') { mySndResetMenuState(); return; }
        else if (c == 't' || c == 'T') break;
      }
      int ta = mySndCheckTouchInput();
      if      (ta == 2) { mySndResetMenuState(); return; }
      else if (ta == 1) break;
      delay(10);
    }
  }
}

// ██████████████████████████████████████████████████████████████████████████████
// ██  PART 3: INFERENCE (VAD-triggered)                                         ██
// ██████████████████████████████████████████████████████████████████████████████

#define mySndVadWindowMs  100
#define mySndVadThreshold 500
#define mySndVadHoldMs    150

void mySndActionInfer() {
  if (!mySndWeightsTrained) {
    u8g2.firstPage();
    do {
      u8g2.setFont(u8g2_font_6x10_tf);
      u8g2.drawStr(0, 12, "No weights!");
      u8g2.drawStr(0, 24, "Train first");
    } while (u8g2.nextPage());
    delay(3000); mySndResetMenuState(); return;
  }

  Serial.println("\n>>> Inference (VAD)  T or L to exit");
  mySndResetTouchState();

  int   lastPred  = 0;
  float lastConf  = 0;
  bool  vadActive = false;
  unsigned long vadRiseTime = 0;

  const int VAD_SAMPLES = (SND_SAMPLE_RATE * mySndVadWindowMs) / 1000;
  static int16_t vadBuf[1600];   // static — avoid stack allocation of 3200 bytes

  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_6x10_tf);
    u8g2.drawStr(0, 12, "LISTENING...");
    u8g2.drawStr(0, 26, "Waiting 4 sound");
  } while (u8g2.nextPage());

  while (true) {
    if (Serial.available()) {
      char c = Serial.read();
      if (c == 't' || c == 'T' || c == 'l' || c == 'L') { mySndResetMenuState(); return; }
    }
    if (mySndCheckTouchInput() != 0) { mySndResetMenuState(); return; }

    size_t got        = mySndI2S.readBytes((char*)vadBuf, VAD_SAMPLES * sizeof(int16_t));
    int    samplesGot = (int)(got / sizeof(int16_t));
    int64_t sumSq = 0;
    for (int i = 0; i < samplesGot; i++) sumSq += (int64_t)vadBuf[i] * vadBuf[i];
    float rms = samplesGot > 0 ? sqrtf((float)(sumSq / samplesGot)) : 0;

    if (rms > mySndVadThreshold) {
      if (!vadActive) { vadActive = true; vadRiseTime = millis(); }
      else if (millis() - vadRiseTime >= mySndVadHoldMs) {
        vadActive = false;
        //Serial.printf("VAD triggered (RMS=%.0f)\n", rms);  // I don't need this

        size_t   wav_size   = 0;
        uint8_t* wav_buffer = mySndI2S.recordWAV(SND_CLIP_SECONDS, &wav_size);
        bool classified = false;

        if (wav_buffer && wav_size > 44) {
          size_t pcmBytes = min(wav_size - 44,
                                (size_t)(SND_CLIP_SAMPLES * sizeof(int16_t)));
          memcpy(mySndAudioBuffer, wav_buffer + 44, pcmBytes);
          if (pcmBytes < SND_CLIP_SAMPLES * sizeof(int16_t))
            memset((uint8_t*)mySndAudioBuffer + pcmBytes, 0,
                   SND_CLIP_SAMPLES * sizeof(int16_t) - pcmBytes);
          free(wav_buffer);

          if (mySndComputeSpectrogram()) {
            float logits[SND_NUM_CLASSES];
            mySndForwardPass(mySndInputBuffer, logits);
            lastPred = 0;
            for (int i = 1; i < SND_NUM_CLASSES; i++)
              if (mySndDense_output[i] > mySndDense_output[lastPred]) lastPred = i;
            lastConf   = mySndDense_output[lastPred];
            classified = true;

            Serial.printf("=> %s (%.1f%%)", mySndClassLabels[lastPred].c_str(), lastConf * 100);
            for (int i = 0; i < SND_NUM_CLASSES; i++)
              Serial.printf("  %s:%.0f%%", mySndClassLabels[i].c_str(), mySndDense_output[i] * 100);
            Serial.println();

            // ---- OLED: class name + percent (top), bar chart (bottom) ----
            {
              const char* lbl = mySndClassLabels[lastPred].c_str();
              if (lbl[0] >= '0' && lbl[0] <= '9') lbl++;   // skip numeric prefix

              int oW  = u8g2.getDisplayWidth();    // 72
              int oH  = u8g2.getDisplayHeight();   // 40
              int pct = (int)(lastConf * 100);

              u8g2.firstPage();
              do {
                // --- Top: class name + confidence percent ---
                u8g2.setFont(u8g2_font_6x10_tf);
                u8g2.drawStr(0, 10, lbl);

                char pctBuf[8];
                snprintf(pctBuf, sizeof(pctBuf), "%d%%", pct);
                u8g2.drawStr(0, 22, pctBuf);

                // --- Separator ---
                u8g2.drawHLine(0, 24, oW);

                // --- Bottom: mini bar per class ---
                u8g2.setFont(u8g2_font_4x6_tf);
                int barAreaTop = 26;
                int barH = (oH - barAreaTop) / SND_NUM_CLASSES;
                if (barH < 4) barH = 4;

                for (int c = 0; c < SND_NUM_CLASSES; c++) {
                  int y    = barAreaTop + c * barH;
                  int barW = (int)(mySndDense_output[c] * (oW - 1));
                  if (barW < 1) barW = 1;
                  if (c == lastPred) u8g2.drawBox(0, y, barW, barH - 1);
                  else               u8g2.drawFrame(0, y, barW, barH - 1);
                }
              } while (u8g2.nextPage());
              // No delay — return to listening loop immediately
            }
          }
        } else {
          if (wav_buffer) free(wav_buffer);
          Serial.println("recordWAV() failed in inference");
        }

        // Back to listening — show last result continuously
        u8g2.firstPage();
        do {
          u8g2.setFont(u8g2_font_6x10_tf);
          if (classified) {
            const char* lbl = mySndClassLabels[lastPred].c_str();
            if (lbl[0] >= '0' && lbl[0] <= '9') lbl++;
            u8g2.drawStr(0, 12, lbl);
            char buf[8];
            snprintf(buf, sizeof(buf), "%d%%", (int)(lastConf * 100));
            u8g2.drawStr(0, 26, buf);
          } else {
            u8g2.drawStr(0, 12, "Listening...");
          }
        } while (u8g2.nextPage());
      }
    } else {
      vadActive = false;
    }

    delay(5);
  }
}

// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                            ██
// ██  COMBINED / SHARED LAYER                                                  ██
// ██  Dual-core glue + unified top-level menu + merged OLED rendering          ██
// ██                                                                            ██
// ██  Everything above this point (Vision section, then Sound section) is the  ██
// ██  original v44 / v13 code, mechanically namespaced:                       ██
// ██    Vision globals/functions -> myVis... / VIS_...                        ██
// ██    Sound  globals/functions -> mySnd... / SND_...                        ██
// ██  Both models' own myVisDrawMenu()/mySndDrawMenu() etc. are still defined  ██
// ██  but are no longer called — the unified menu below replaces them.         ██
// ██                                                                            ██
// ██████████████████████████████████████████████████████████████████████████████

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

// ======================================================
// UNIFIED TOP-LEVEL TOUCH INPUT (single physical A0 pad,
// used only for navigating the combined menu on core 1)
// ======================================================
struct TouchState {
  bool          isTouching      = false;
  int           tapCount        = 0;
  unsigned long firstTapTime    = 0;
  unsigned long lastReleaseTime = 0;
  unsigned long lastCheckTime   = 0;
  const unsigned long tapWindow     = 800;
  const int           longPressTaps = 3;
  const unsigned long debounceDelay = 50;
};
TouchState myTouch;

const int myThresholdPress   = 1100;
const int myThresholdRelease = 900;

int myReadTouch() {
  int sum = 0;
  for (int i = 0; i < 3; i++) { sum += analogRead(A0); delayMicroseconds(100); }
  return sum / 3;
}

void myResetTouchState() {
  myTouch.isTouching      = false;
  myTouch.tapCount        = 0;
  myTouch.firstTapTime    = 0;
  myTouch.lastReleaseTime = 0;
  myTouch.lastCheckTime   = 0;
}

void myUpdateTouchState() {
  unsigned long now = millis();
  if (now - myTouch.lastCheckTime < 20) return;
  myTouch.lastCheckTime = now;
  int val = myReadTouch();
  bool active = myTouch.isTouching ? (val > myThresholdRelease) : (val > myThresholdPress);
  if (active && !myTouch.isTouching) {
    if (now - myTouch.lastReleaseTime < myTouch.debounceDelay) return;
    myTouch.isTouching = true;
    if (myTouch.tapCount == 0 || (now - myTouch.firstTapTime < myTouch.tapWindow)) {
      if (myTouch.tapCount == 0) myTouch.firstTapTime = now;
      myTouch.tapCount++;
    } else {
      myTouch.tapCount     = 1;
      myTouch.firstTapTime = now;
    }
  }
  if (!active && myTouch.isTouching) {
    myTouch.isTouching      = false;
    myTouch.lastReleaseTime = now;
  }
}

// returns 0 = nothing, 1 = short tap (advance), 2 = long-press (select)
int myCheckTouchInput() {
  myUpdateTouchState();
  unsigned long now = millis();
  if (myTouch.tapCount > 0 && !myTouch.isTouching &&
      now - myTouch.firstTapTime > myTouch.tapWindow) {
    int result = (myTouch.tapCount >= myTouch.longPressTaps) ? 2 : 1;
    myResetTouchState();
    return result;
  }
  return 0;
}

// ======================================================
// UNIFIED MENU — flat list spanning both models
//   1..VIS_NUM_CLASSES              Vision: Collect <class>
//   +1                              Vision: Train
//   +1                              Vision: Infer (camera only)
//   +1..+SND_NUM_CLASSES            Sound:  Collect <class>
//   +1                              Sound:  Train
//   +1                              Sound:  Infer (mic only)
//   +1                              BOTH:   Combined Infer
// ======================================================
const int myVisCollectBase = 0;                                   // items 1..VIS_NUM_CLASSES
const int myVisTrainItem   = VIS_NUM_CLASSES + 1;
const int myVisInferItem   = VIS_NUM_CLASSES + 2;
const int mySndCollectBase = VIS_NUM_CLASSES + 2;                 // items +1..+SND_NUM_CLASSES
const int mySndTrainItem   = mySndCollectBase + SND_NUM_CLASSES + 1;
const int mySndInferItem   = mySndCollectBase + SND_NUM_CLASSES + 2;
const int myCombinedItem   = mySndCollectBase + SND_NUM_CLASSES + 3;
const int myTotalItems     = myCombinedItem;

int  myMenuIndex  = 1;
bool myIsSelected = false;
unsigned long myLastActivityTime = 0;
unsigned long myLastTapTime      = 0;
const int     myTapCooldown      = 250;

void myActionCombinedInfer();  // forward decl, defined below

String myMenuLabel(int idx) {
  if (idx <= VIS_NUM_CLASSES)                 return "V:" + myVisClassLabels[idx - 1];
  if (idx == myVisTrainItem)                  return "Vision Train";
  if (idx == myVisInferItem)                  return "Vision Infer";
  if (idx <= mySndCollectBase + SND_NUM_CLASSES)
                                               return "S:" + mySndClassLabels[idx - mySndCollectBase - 1];
  if (idx == mySndTrainItem)                  return "Sound Train";
  if (idx == mySndInferItem)                  return "Sound Infer";
  return "BOTH Infer";
}

void myExecuteMenuItem(int idx) {
  if (idx <= VIS_NUM_CLASSES)                 myVisActionCollect(idx - 1);
  else if (idx == myVisTrainItem)             myVisActionTrain();
  else if (idx == myVisInferItem)             myVisActionInfer();
  else if (idx <= mySndCollectBase + SND_NUM_CLASSES)
                                               mySndActionCollect(idx - mySndCollectBase - 1);
  else if (idx == mySndTrainItem)             mySndActionTrain();
  else if (idx == mySndInferItem)             mySndActionInfer();
  else                                        myActionCombinedInfer();
}

void myDrawMenu() {
  Serial.println("\n=== COMBINED MENU ===");
  for (int i = 1; i <= myTotalItems; i++) {
    Serial.printf("%s %d. %s\n", i == myMenuIndex ? " >" : "  ", i, myMenuLabel(i).c_str());
  }
  Serial.println("t=next  l=select  1-9,a=10,b=11=direct");

  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_6x10_tf);
    u8g2.drawStr(0, 8, "TAP:Next HOLD:Ok");
    int start = max(1, myMenuIndex - 1);
    for (int i = 0; i < 3; i++) {
      int cur = start + i;
      if (cur > myTotalItems) break;
      String label = myMenuLabel(cur);
      u8g2.drawStr(0, 18 + i * 9,
                   (cur == myMenuIndex ? "> " + label : "  " + label).c_str());
    }
  } while (u8g2.nextPage());
}

void myResetMenuState() {
  myIsSelected = false;
  myResetTouchState();
  myLastActivityTime = millis();
  myDrawMenu();
}

void myHandleMenuNavigation() {
  unsigned long now = millis();
  if (!myIsSelected && Serial.available()) {
    char c = Serial.read();
    int  directIndex = 0;
    if (c >= '1' && c <= '9')      directIndex = c - '0';        // 1-9
    else if (c == 'a' || c == 'A') directIndex = 10;              // 10th item
    else if (c == 'b' || c == 'B') directIndex = 11;              // 11th item

    if (directIndex > 0) {
      if (directIndex <= myTotalItems) {
        myMenuIndex = directIndex;
        myIsSelected = true; myLastActivityTime = now; myExecuteMenuItem(myMenuIndex);
      }
    }
    else if (c == 't' || c == 'T') {
      if (now - myLastTapTime > myTapCooldown) {
        myMenuIndex = (myMenuIndex % myTotalItems) + 1;
        myDrawMenu(); myLastTapTime = now; myLastActivityTime = now;
      }
    } else if (c == 'l' || c == 'L') {
      myIsSelected = true; myLastActivityTime = now; myExecuteMenuItem(myMenuIndex);
    }
  }
  if (!myIsSelected) {
    int ta = myCheckTouchInput();
    if (ta == 1) {
      if (now - myLastTapTime > myTapCooldown) {
        myMenuIndex = (myMenuIndex % myTotalItems) + 1;
        myDrawMenu(); myLastTapTime = now; myLastActivityTime = now;
      }
    } else if (ta == 2) {
      myIsSelected = true; myLastActivityTime = now; myExecuteMenuItem(myMenuIndex);
    }
  }
  // every action function calls its own model-local *ResetMenuState() when it
  // finishes (which no longer redraws its own menu — see the NOTE edits above),
  // so we always redraw the unified menu here once control returns to us.
  if (myIsSelected) {
    myIsSelected = false;
    myDrawMenu();
  }
}

// ======================================================
// DUAL-CORE SCAFFOLDING
// Sound inference runs as a FreeRTOS task pinned to core 0.
// It is only alive while "Combined Infer" (menu item 11) is active.
// It never touches the OLED — it only writes into gSoundResult,
// which core 1 reads (under mutex) once per rendered vision frame.
// ======================================================
typedef struct {
  int   classIdx;
  float confidence;
  float confAll[SND_NUM_CLASSES];   // full per-class distribution, for raw-value logging
  bool  resultReady;
  unsigned long lastUpdateMs;
} SoundResult;

volatile SoundResult gSoundResult = {0, 0.0f, {0}, false, 0};
SemaphoreHandle_t     gSoundResultMutex = nullptr;
TaskHandle_t          gSoundTaskHandle  = nullptr;
volatile bool         gSoundTaskRun     = false;

// Body of the original mySndActionInfer() VAD-listen loop, stripped of
// OLED/serial/exit handling, running forever on core 0 until deleted.
void mySndInferTask(void* pvParameters) {
  const int myVadWindowMsLocal  = 100;
  const int myVadThresholdLocal = 500;
  const int myVadHoldMsLocal    = 150;

  bool  vadActive    = false;
  unsigned long vadRiseTime = 0;
  const int VAD_SAMPLES = (SND_SAMPLE_RATE * myVadWindowMsLocal) / 1000;
  static int16_t vadBuf[1600];

  while (gSoundTaskRun) {
    size_t got        = mySndI2S.readBytes((char*)vadBuf, VAD_SAMPLES * sizeof(int16_t));
    int    samplesGot = (int)(got / sizeof(int16_t));
    int64_t sumSq = 0;
    for (int i = 0; i < samplesGot; i++) sumSq += (int64_t)vadBuf[i] * vadBuf[i];
    float rms = samplesGot > 0 ? sqrtf((float)(sumSq / samplesGot)) : 0;

    if (rms > myVadThresholdLocal) {
      if (!vadActive) { vadActive = true; vadRiseTime = millis(); }
      else if (millis() - vadRiseTime >= (unsigned long)myVadHoldMsLocal) {
        vadActive = false;

        size_t   wav_size   = 0;
        uint8_t* wav_buffer = mySndI2S.recordWAV(SND_CLIP_SECONDS, &wav_size);

        if (wav_buffer && wav_size > 44) {
          size_t pcmBytes = min(wav_size - 44,
                                (size_t)(SND_CLIP_SAMPLES * sizeof(int16_t)));
          memcpy(mySndAudioBuffer, wav_buffer + 44, pcmBytes);
          if (pcmBytes < SND_CLIP_SAMPLES * sizeof(int16_t))
            memset((uint8_t*)mySndAudioBuffer + pcmBytes, 0,
                   SND_CLIP_SAMPLES * sizeof(int16_t) - pcmBytes);
          free(wav_buffer);

          if (mySndComputeSpectrogram()) {
            float logits[SND_NUM_CLASSES];
            mySndForwardPass(mySndInputBuffer, logits);
            int   pred = 0;
            for (int i = 1; i < SND_NUM_CLASSES; i++)
              if (mySndDense_output[i] > mySndDense_output[pred]) pred = i;
            float conf = mySndDense_output[pred];

            if (xSemaphoreTake(gSoundResultMutex, pdMS_TO_TICKS(50)) == pdTRUE) {
              gSoundResult.classIdx     = pred;
              gSoundResult.confidence   = conf;
              for (int i = 0; i < SND_NUM_CLASSES; i++)
                gSoundResult.confAll[i] = mySndDense_output[i];
              gSoundResult.resultReady  = true;
              gSoundResult.lastUpdateMs = millis();
              xSemaphoreGive(gSoundResultMutex);
            }
          } else {
            Serial.println("[Sound/core0] spectrogram compute failed, skipping");
          }
        } else if (wav_buffer) {
          free(wav_buffer);
        }
      }
    } else {
      vadActive = false;
    }
    vTaskDelay(pdMS_TO_TICKS(5));   // yield to idle task / watchdog
  }
  gSoundTaskHandle = nullptr;
  vTaskDelete(nullptr);            // task deletes itself on exit
}

// ======================================================
// COMBINED INFERENCE — menu item 11
// Core 1 (this function): camera capture + vision inference,
//   same cadence/logic as myVisActionInfer(), but the OLED
//   render also draws the latest sound result along the bottom.
// Core 0 (mySndInferTask): VAD-listen -> record -> classify,
//   posted into gSoundResult every time a clip completes.
// ======================================================
void myActionCombinedInfer() {
  if (!myVisWeightsTrained || !mySndWeightsTrained) {
    u8g2.firstPage();
    do {
      u8g2.setFont(u8g2_font_6x10_tf);
      u8g2.drawStr(0, 12, "Need BOTH");
      u8g2.drawStr(0, 24, "models trained");
    } while (u8g2.nextPage());
    delay(2500);
    myResetMenuState();
    return;
  }
  if (!myVisRgbBuffer || !myVisInputBuffer || !myVisDense_output) {
    Serial.println("ERROR: Vision memory not allocated");
    myResetMenuState();
    return;
  }

  Serial.println("\n>>> Combined Infer (vision core1 + sound core0)  T/L to exit");
  myResetTouchState();

  if (gSoundResultMutex == nullptr) gSoundResultMutex = xSemaphoreCreateMutex();
  gSoundResult.resultReady = false;
  gSoundTaskRun = true;
  xTaskCreatePinnedToCore(
    mySndInferTask, "SndInferTask",
    16384,            // stack: generous headroom; all big buffers are PSRAM, not stack
    nullptr,
    1,                // priority
    &gSoundTaskHandle,
    0                 // pin to core 0 (PRO_CPU) — vision stays on core 1 (APP_CPU)
  );

  static int sy_lookup[VIS_INPUT_SIZE];
  static int sx_lookup[VIS_INPUT_SIZE];
  static bool lookup_initialized = false;
  if (!lookup_initialized) {
    for (int i = 0; i < VIS_INPUT_SIZE; i++) {
      sy_lookup[i] = min((int)((i + 0.5) * 240.0 / VIS_INPUT_SIZE), 239);
      sx_lookup[i] = min((int)((i + 0.5) * 240.0 / VIS_INPUT_SIZE), 239);
    }
    lookup_initialized = true;
  }

  int frameIndex = 0;
  int visPred = 0;

  while (true) {
    if (Serial.available()) {
      char c = Serial.read();
      if (c == 't' || c == 'T' || c == 'l' || c == 'L') break;
    }

    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) { delay(10); continue; }

    if (fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, myVisRgbBuffer)) {
      for (int y = 0; y < VIS_INPUT_SIZE; y++) {
        int sy = sy_lookup[y];
        int sy_offset = sy * 240;
        int dst_y_offset = y * VIS_INPUT_SIZE;
        for (int x = 0; x < VIS_INPUT_SIZE; x++) {
          int srcIdx = (sy_offset + sx_lookup[x]) * 3;
          int dstIdx = (dst_y_offset + x) * 3;
          myVisInputBuffer[dstIdx]     = myVisRgbBuffer[srcIdx]     * 0.003921569f;
          myVisInputBuffer[dstIdx + 1] = myVisRgbBuffer[srcIdx + 1] * 0.003921569f;
          myVisInputBuffer[dstIdx + 2] = myVisRgbBuffer[srcIdx + 2] * 0.003921569f;
        }
      }
      float visLogits[VIS_NUM_CLASSES];
      myVisForwardPass(myVisInputBuffer, visLogits);
      visPred = 0;
      for (int i = 1; i < VIS_NUM_CLASSES; i++)
        if (myVisDense_output[i] > myVisDense_output[visPred]) visPred = i;

      // ---- snapshot the latest sound result under mutex (every frame — cheap) ----
      int   sndPred = -1;
      float sndConf = 0;
      float sndAll[SND_NUM_CLASSES] = {0};
      if (gSoundResultMutex && xSemaphoreTake(gSoundResultMutex, pdMS_TO_TICKS(20)) == pdTRUE) {
        if (gSoundResult.resultReady) {
          sndPred = gSoundResult.classIdx;
          sndConf = gSoundResult.confidence;
          for (int i = 0; i < SND_NUM_CLASSES; i++) sndAll[i] = gSoundResult.confAll[i];
        }
        xSemaphoreGive(gSoundResultMutex);
      }

      // ---- one uniform status line, every vision frame (~7 FPS) ----
      {
        char visRaw[48] = "";
        int off = 0;
        for (int i = 0; i < VIS_NUM_CLASSES && off < (int)sizeof(visRaw) - 1; i++)
          off += snprintf(visRaw + off, sizeof(visRaw) - off, "%s%d",
                          i == 0 ? "" : ", ", (int)(myVisDense_output[i] * 100));

        char sndRaw[48] = "";
        off = 0;
        for (int i = 0; i < SND_NUM_CLASSES && off < (int)sizeof(sndRaw) - 1; i++)
          off += snprintf(sndRaw + off, sizeof(sndRaw) - off, "%s%d",
                          i == 0 ? "" : ", ", (int)(sndAll[i] * 100));

        char sndWinner[24];
        if (sndPred >= 0)
          snprintf(sndWinner, sizeof(sndWinner), "%s %d%%",
                   mySndClassLabels[sndPred].c_str(), (int)(sndConf * 100));
        else
          snprintf(sndWinner, sizeof(sndWinner), "listening");

        Serial.printf("[Combined] V:%s %d%%  |  S:%s,   |  V:%s, S: %s\n",
                      myVisClassLabels[visPred].c_str(), (int)(myVisDense_output[visPred] * 100),
                      sndWinner, visRaw, sndRaw);
      }

      if (frameIndex == 9) {
        int oW = u8g2.getDisplayWidth();
        int oH = u8g2.getDisplayHeight();
        int scX = 240 / oW;
        int scY = 240 / oH;
        int soundBarH = 10;                 // bottom strip reserved for sound result
        int visBottom = oH - soundBarH;

        u8g2.firstPage();
        do {
          // --- top: downsampled camera image, cropped to leave room at bottom ---
          for (int ox = 0; ox < oW; ox++) {
            for (int oy = 0; oy < visBottom; oy++) {
              int pi = ((oy * scY) * 240 + (ox * scX)) * 3;
              uint8_t bright = (myVisRgbBuffer[pi] + myVisRgbBuffer[pi + 1] + myVisRgbBuffer[pi + 2]) / 3;
              if (bright > 100) u8g2.drawPixel(ox, oy);
            }
          }
          // --- vision label bar, just above the sound strip ---
          u8g2.setFont(u8g2_font_5x7_tf);
          u8g2.setColorIndex(0);
          u8g2.drawBox(0, visBottom - 9, oW, 9);
          u8g2.setColorIndex(1);
          char visBuf[24];
          snprintf(visBuf, sizeof(visBuf), "V : %s %d%%",
                   myVisClassLabels[visPred].c_str(), (int)(myVisDense_output[visPred] * 100));
          u8g2.drawStr(1, visBottom - 1, visBuf);

          // --- sound strip along the very bottom ---
          u8g2.setColorIndex(1);
          u8g2.drawBox(0, visBottom, oW, soundBarH);
          u8g2.setColorIndex(0);
          char sndBuf[24];
          if (sndPred >= 0) {
            snprintf(sndBuf, sizeof(sndBuf), "S : %s %d%%",
                     mySndClassLabels[sndPred].c_str(), (int)(sndConf * 100));
          } else {
            snprintf(sndBuf, sizeof(sndBuf), "S : listening");
          }
          u8g2.drawStr(1, oH - 1, sndBuf);
          u8g2.setColorIndex(1);
        } while (u8g2.nextPage());
      }
    }
    esp_camera_fb_return(fb);

    frameIndex++;
    if (frameIndex >= 10) {
      int touchVal = myReadTouch();
      if (touchVal > myThresholdPress) { delay(200); break; }
      frameIndex = 0;
    }
  }

  // stop the core-0 sound task cleanly before leaving
  gSoundTaskRun = false;
  unsigned long waitStart = millis();
  while (gSoundTaskHandle != nullptr && millis() - waitStart < 500) delay(5);

  myResetMenuState();
}

// ██████████████████████████████████████████████████████████████████████████████
// ██  SETUP & LOOP  (merged)                                                    ██
// ██████████████████████████████████████████████████████████████████████████████

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000);
  delay(500);
  Serial.println("\n=== XIAO ESP32-S3 Combined Vision+Sound ML ===");
  Serial.printf("Heap: %d  PSRAM: %d\n", ESP.getFreeHeap(), ESP.getFreePsram());

  pinMode(A0, INPUT);
  u8g2.begin();

  // ---- SD card: single shared init (slow-speed probe, like the vision sketch) ----
  pinMode(21, OUTPUT); digitalWrite(21, HIGH); delay(100);
  SPI.begin();
  SPI.setFrequency(400000);
  bool sdOk = SD.begin(21, SPI, 400000, "/sd", 5, false);
  myVisSDavailable = sdOk;
  mySndSDavailable = sdOk;
  if (!sdOk) {
    SD.end();
    Serial.println("No SD card - continuing without it");
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "No SD card"); } while (u8g2.nextPage());
    delay(2000);
  } else {
    Serial.println("SD card mounted successfully");
  }

  // ---- Camera (vision, core 1 / DVP peripheral) ----
  myVisRgbBuffer = (uint8_t*)ps_malloc(240 * 240 * 3);
  if (!myVisRgbBuffer) Serial.println("Failed to allocate RGB buffer!");

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM; config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM; config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM; config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM; config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM; config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM; config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM; config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM; config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000; config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_240X240; config.jpeg_quality = 12;
  config.fb_count = 1;
  esp_camera_init(&config);
  Serial.println("Camera initialized");
  sensor_t* s = esp_camera_sensor_get();
  if (s != NULL) s->set_hmirror(s, 1);

  esp_log_level_set("*", ESP_LOG_WARN);
  esp_log_level_set("esp_camera", ESP_LOG_ERROR);

  // ---- Mic (sound, core 0 / PDM peripheral) ----
  mySndI2S.setPinsPdmRx(42, 41);
  if (!mySndI2S.begin(I2S_MODE_PDM_RX, SND_SAMPLE_RATE, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO)) {
    Serial.println("FATAL: I2S init failed!");
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "MIC FAIL!"); } while (u8g2.nextPage());
    while (1) delay(1000);
  }
  Serial.println("I2S mic OK");

  // ---- PSRAM allocation + weight loading for BOTH models ----
  myVisAllocateMemory();
  mySndAllocateMemory();
  mySndBuildMelFilterbank();
  mySndBuildTwiddleTable();

  if (myVisLoadWeights()) Serial.println("Vision: SD weights loaded");
  if (mySndLoadWeights()) Serial.println("Sound:  SD weights loaded");

  gSoundResultMutex = xSemaphoreCreateMutex();

  myLastActivityTime = millis();
  myResetMenuState();
  delay(1000);
  Serial.println("Ready. Tap A0 to navigate, 3+ taps to select.");
  myDrawMenu();
}

void loop() {
  myHandleMenuNavigation();
}
