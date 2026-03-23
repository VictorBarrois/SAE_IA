#include <M5GFX.h>
#include <../CNN/mnist_float32.h>
#include <math.h>
#include <WiFi.h>

M5GFX display;

#define GRID_SIZE 28
#define TEXT_SIZE 2

float grid[GRID_SIZE][GRID_SIZE];

void printGrid();
void runCNN();
void resetGrid();
void showResultOnScreen(int predicted, float confidence, float inference_s);
void showReadyScreen();

const char* ssid = "ESP32_IA_PROJECT";
const char* password = "GeiiTailscale2024$";

// =====================================================
// SOFTMAX (stable)
// =====================================================
void softmax(float *input, float *output, int size) {
  float maxVal = input[0];

  for (int i = 1; i < size; i++) {
    if (input[i] > maxVal) maxVal = input[i];
  }

  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    output[i] = expf(input[i] - maxVal);
    sum += output[i];
  }

  for (int i = 0; i < size; i++) {
    output[i] /= sum;
  }
}


// =====================================================
// RESET GRID
// =====================================================
void resetGrid() {
  for (int y = 0; y < GRID_SIZE; y++) {
    for (int x = 0; x < GRID_SIZE; x++) {
      grid[y][x] = 0.0f;
    }
  }
}


// =====================================================
// ECRAN D'ACCUEIL
// =====================================================
void showReadyScreen() {
  display.fillScreen(TFT_BLACK);
  display.setTextColor(TFT_WHITE, TFT_BLACK);
  display.setTextSize(TEXT_SIZE);
  display.setCursor(10, 20);
  display.println("Dessine un chiffre");
}


// =====================================================
// SETUP
// =====================================================
void setup() {
  Serial.begin(115200);

  display.init();
  display.startWrite();
  display.setRotation(1);
  display.fillScreen(TFT_BLACK);
  display.setTextColor(TFT_WHITE, TFT_BLACK);

  resetGrid();
  showReadyScreen();
  WiFi.begin(ssid, password);
}


// =====================================================
// LOOP : dessin tactile
// =====================================================
void loop() {

  lgfx::touch_point_t tp[1];
  int nums = display.getTouchRaw(tp, 1);

  static bool drawing = false;

  if (nums) {
    if (!drawing) {
      display.fillScreen(TFT_BLACK);
      drawing = true;
    }

    int x = tp[0].x;
    int y = tp[0].y;

    // Dessin écran
    display.fillCircle(x, y, 4, TFT_WHITE);

    // écran -> grille
    int gx = map(x, 0, 320, 0, 27);
    int gy = map(y, 0, 240, 0, 27);

    // Pixel central = 1.0
    if (gx >= 0 && gx < GRID_SIZE && gy >= 0 && gy < GRID_SIZE) {
      grid[gy][gx] = 1.0f;
    }

    // Épaissir autour = 0.5
    for (int dy = -1; dy <= 1; dy++) {
      for (int dx = -1; dx <= 1; dx++) {
        int nx = gx + dx;
        int ny = gy + dy;

        if (nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE) {
          if (dx != 0 || dy != 0) {
            if (grid[ny][nx] < 0.5f) {
              grid[ny][nx] = 0.5f;
            }
          }
        }
      }
    }
  }
  else if (drawing) {
    // doigt relâché
    printGrid();
    runCNN();

    drawing = false;

    // laisser le résultat visible
    delay(2000);

    // reset pour le prochain dessin
    resetGrid();
    showReadyScreen();
  }
}


// =====================================================
// AFFICHAGE RESULTAT SUR ECRAN
// =====================================================
void showResultOnScreen(int predicted, float confidence, float inference_s) {
  display.fillScreen(TFT_BLACK);
  display.setTextColor(TFT_WHITE, TFT_BLACK);
  display.setTextSize(TEXT_SIZE);

  display.setCursor(10, 20);
  display.print("Chiffre: ");
  display.println(predicted);

  display.setCursor(10, 90);
  display.print("Confidence: ");
  display.print(confidence, 2);
  display.println("%");

  display.setCursor(10, 160);
  display.print("Temps: ");
  display.print(inference_s, 4);
  display.println(" s");
}


// =====================================================
// EXECUTION CNN + SOFTMAX
// =====================================================
void runCNN() {
  input_t input;
  dense_9_output_type output;

  for (int y = 0; y < 28; y++) {
    for (int x = 0; x < 28; x++) {
      input[y][x][0] = grid[y][x];
    }
  }

  unsigned long t0 = micros();
  cnn(input, output);
  unsigned long t1 = micros();

  float inference_s = (t1 - t0) / 1000000.0f;

  float probs[10];
  softmax(output, probs, 10);

  int predicted = 0;
  float maxVal = probs[0];

  for (int i = 1; i < 10; i++) {
    if (probs[i] > maxVal) {
      maxVal = probs[i];
      predicted = i;
    }
  }

  float confidence = probs[predicted] * 100.0f;

  Serial.println("=========== RESULTAT CNN ===========");

  for (int i = 0; i < 10; i++) {
    Serial.print(i);
    Serial.print(" : ");
    Serial.print(probs[i] * 100.0f, 4);
    Serial.println("%");
  }

  Serial.print("Chiffre reconnu : ");
  Serial.print(predicted);
  Serial.print(" --> ");
  Serial.print(confidence, 2);
  Serial.println("%");

  Serial.print("Temps d'inference : ");
  Serial.print(inference_s, 6);
  Serial.println(" s");

  Serial.println("====================================");

  showResultOnScreen(predicted, confidence, inference_s);
}


// =====================================================
// DEBUG : affichage grille
// =====================================================
void printGrid() {
  Serial.println("============= GRID =============");

  for (int y = 0; y < GRID_SIZE; y++) {
    for (int x = 0; x < GRID_SIZE; x++) {
      Serial.print(grid[y][x], 1);
      Serial.print(" ");
    }
    Serial.println();
  }

  Serial.println("================================");
}