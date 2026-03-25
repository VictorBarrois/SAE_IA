#include <M5GFX.h>
#include <../CNN/mnist_fixed_int16.h>
#include <math.h>
#include <WiFi.h>
#include <stdint.h>

M5GFX display;

#define GRID_SIZE 28
#define TEXT_SIZE 2

// =====================================================
// PARAMETRES PTQ A AJUSTER SELON TON MODELE
// =====================================================
//
// Exemple classique si l'entrée réelle est normalisée entre 0.0 et 1.0
// et quantifiée en int8 : q = round(real / scale) + zero_point
//
// Ici je mets des valeurs d'exemple.
// Tu dois idéalement remplacer avec les vraies valeurs du modèle exporté.
//
static const float INPUT_SCALE = 1.0f / 255.0f;   // exemple PTQ
static const int   INPUT_ZERO_POINT = 0;          // exemple PTQ

// Si ton modèle attend du int8 signé : plage [-128,127]
// Si ton modèle attend du uint8 : plage [0,255]
// Si ton modèle attend du int16 : il faudra adapter le clamp.
// Ici on suppose une entrée signée compacte.
static const int INPUT_QMIN = -128;
static const int INPUT_QMAX = 127;

// =====================================================
// GRILLE QUANTIFIEE
// =====================================================
//
// 0   = noir
// 255 = blanc
//
uint8_t grid[GRID_SIZE][GRID_SIZE];

// WiFi AP + serveur TCP
const char* ssid = "ESP32_IA_PROJECT";
const char* password = "GeiiTailscale2024$";
WiFiServer server(12345);
WiFiClient client;

// Prototypes
void printGrid();
void runCNN(unsigned long preprocessing_us);
void resetGrid();
void showResultOnScreen(int predicted, int confidence_percent, unsigned long preprocessing_ms, unsigned long inference_ms, unsigned long total_ms);
void showReadyScreen();
void handleWifiClient();
void sendLine(const String& msg);
void sendText(const String& msg);

// Quantification PTQ
int quantize_input_u8_to_model(uint8_t pixel);
void softmaxFromInt16(const int16_t *input, float *output, int size);

// =====================================================
// QUANTIFICATION PTQ
// =====================================================
//
// pixel est en [0..255]
// on le ramène en réel [0..1] puis quantification PTQ:
//
// real_value = pixel / 255.0
// q = round(real_value / INPUT_SCALE) + INPUT_ZERO_POINT
//
int quantize_input_u8_to_model(uint8_t pixel) {
  float real_value = (float)pixel / 255.0f;
  int q = (int)roundf(real_value / INPUT_SCALE) + INPUT_ZERO_POINT;

  if (q < INPUT_QMIN) q = INPUT_QMIN;
  if (q > INPUT_QMAX) q = INPUT_QMAX;

  return q;
}

// =====================================================
// SOFTMAX SUR SORTIE INT16
// =====================================================
void softmaxFromInt16(const int16_t *input, float *output, int size) {
  int16_t maxVal = input[0];
  for (int i = 1; i < size; i++) {
    if (input[i] > maxVal) {
      maxVal = input[i];
    }
  }

  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    output[i] = expf((float)(input[i] - maxVal));
    sum += output[i];
  }

  if (sum > 0.0f) {
    for (int i = 0; i < size; i++) {
      output[i] /= sum;
    }
  }
}

// =====================================================
// ENVOI TEXTE
// =====================================================
void sendLine(const String& msg) {
  Serial.println(msg);
  if (client && client.connected()) {
    client.println(msg);
  }
}

void sendText(const String& msg) {
  Serial.print(msg);
  if (client && client.connected()) {
    client.print(msg);
  }
}

// =====================================================
// RESET GRID
// =====================================================
void resetGrid() {
  for (int y = 0; y < GRID_SIZE; y++) {
    for (int x = 0; x < GRID_SIZE; x++) {
      grid[y][x] = 0;
    }
  }
}

// =====================================================
// ECRAN READY
// =====================================================
void showReadyScreen() {
  display.fillScreen(TFT_BLACK);
  display.setTextColor(TFT_WHITE, TFT_BLACK);
  display.setTextSize(TEXT_SIZE);

  display.setCursor(10, 20);
  display.println("Dessine un chiffre");

  display.setCursor(10, 60);
  display.println("Puis relache");
}

// =====================================================
// WIFI CLIENT
// =====================================================
void handleWifiClient() {
  if (!client || !client.connected()) {
    WiFiClient newClient = server.available();
    if (newClient) {
      client = newClient;
      Serial.println("Client TCP connecte.");
      client.println("Connexion ESP32 OK");
    }
  }
}

// =====================================================
// AFFICHAGE RESULTAT
// =====================================================
void showResultOnScreen(int predicted, int confidence_percent, unsigned long preprocessing_ms, unsigned long inference_ms, unsigned long total_ms) {
  display.fillScreen(TFT_BLACK);
  display.setTextColor(TFT_WHITE, TFT_BLACK);
  display.setTextSize(TEXT_SIZE);

  display.setCursor(10, 20);
  display.print("Chiffre: ");
  display.println(predicted);

  display.setCursor(10, 60);
  display.print("Confidence: ");
  display.print(confidence_percent);
  display.println("%");

  display.setCursor(10, 100);
  display.print("Preproc: ");
  display.print(preprocessing_ms);
  display.println(" ms");

  display.setCursor(10, 140);
  display.print("Inference: ");
  display.print(inference_ms);
  display.println(" ms");

  display.setCursor(10, 180);
  display.print("Total: ");
  display.print(total_ms);
  display.println(" ms");
}

// =====================================================
// CNN
// =====================================================
void runCNN(unsigned long preprocessing_us) {
  input_t input;
  dense_5_output_type output;

  // -------------------------------------------------
  // Remplissage de l'entrée quantifiée
  // -------------------------------------------------
  //
  // ATTENTION :
  // cette affectation suppose que input_t accepte un entier
  // compatible avec le type généré dans mnist_fixed_int16.h
  //
  // Si ton input_t est int8_t, uint8_t, int16_t : ça ira.
  // Sinon il faudra adapter précisément le cast.
  //
  for (int y = 0; y < GRID_SIZE; y++) {
    for (int x = 0; x < GRID_SIZE; x++) {
      int q = quantize_input_u8_to_model(grid[y][x]);
      input[y][x][0] = q;
    }
  }

  unsigned long t0 = micros();
  cnn(input, output);
  unsigned long t1 = micros();

  unsigned long inference_us = t1 - t0;

  // temps en millisecondes entières
  unsigned long preprocessing_ms = preprocessing_us / 1000UL;
  unsigned long inference_ms = inference_us / 1000UL;
  unsigned long total_ms = preprocessing_ms + inference_ms;

  float probs[10];
  softmaxFromInt16(output, probs, 10);

  int predicted = 0;
  float maxProb = probs[0];

  for (int i = 1; i < 10; i++) {
    if (probs[i] > maxProb) {
      maxProb = probs[i];
      predicted = i;
    }
  }

  int confidence_percent = (int)roundf(probs[predicted] * 100.0f);

  sendLine("=========== RESULTAT CNN ===========");

  for (int i = 0; i < 10; i++) {
    int p = (int)roundf(probs[i] * 10000.0f); // pour afficher 2 decimales approx
    int integerPart = p / 100;
    int fracPart = p % 100;

    String frac = String(fracPart);
    if (fracPart < 10) frac = "0" + frac;

    sendLine(String(i) + " : " + String(integerPart) + "." + frac + "%");
  }

  sendLine("Chiffre reconnu : " + String(predicted) + " --> " + String(confidence_percent) + "%");
  sendLine("Preprocessing : " + String(preprocessing_ms) + " ms");
  sendLine("Inference : " + String(inference_ms) + " ms");
  sendLine("Total : " + String(total_ms) + " ms");
  sendLine("====================================");

  showResultOnScreen(predicted, confidence_percent, preprocessing_ms, inference_ms, total_ms);
}

// =====================================================
// DEBUG GRID
// =====================================================
void printGrid() {
  sendLine("============= GRID =============");

  for (int y = 0; y < GRID_SIZE; y++) {
    String line = "";
    for (int x = 0; x < GRID_SIZE; x++) {
      line += String(grid[y][x]);
      if (x < GRID_SIZE - 1) line += " ";
    }
    sendLine(line);
  }

  sendLine("================================");
}

// =====================================================
// SETUP
// =====================================================
void setup() {
  Serial.begin(115200);
  delay(500);

  display.init();
  display.startWrite();
  display.setRotation(1);
  display.fillScreen(TFT_BLACK);
  display.setTextColor(TFT_WHITE, TFT_BLACK);

  resetGrid();
  showReadyScreen();

  WiFi.mode(WIFI_AP);
  WiFi.softAP(ssid, password);

  IPAddress ip = WiFi.softAPIP();
  Serial.print("AP IP: ");
  Serial.println(ip);

  server.begin();
}

// =====================================================
// LOOP
// =====================================================
void loop() {
  handleWifiClient();

  lgfx::touch_point_t tp[1];
  int nums = display.getTouchRaw(tp, 1);

  static bool drawing = false;
  static unsigned long preprocessing_us = 0;

  if (nums) {
    if (!drawing) {
      display.fillScreen(TFT_BLACK);
      drawing = true;
      preprocessing_us = 0;
    }

    unsigned long t0 = micros();

    int x = tp[0].x;
    int y = tp[0].y;

    display.fillCircle(x, y, 4, TFT_WHITE);

    int gx = map(x, 0, 320, 0, 27);
    int gy = map(y, 0, 240, 0, 27);

    if (gx >= 0 && gx < GRID_SIZE && gy >= 0 && gy < GRID_SIZE) {
      grid[gy][gx] = 255;
    }

    // Petit épaississement du trait en entier
    for (int dy = -1; dy <= 1; dy++) {
      for (int dx = -1; dx <= 1; dx++) {
        int nx = gx + dx;
        int ny = gy + dy;

        if (nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE) {
          if (dx != 0 || dy != 0) {
            if (grid[ny][nx] < 128) {
              grid[ny][nx] = 128;
            }
          }
        }
      }
    }

    unsigned long t1 = micros();
    preprocessing_us += (t1 - t0);
  }
  else if (drawing) {
    printGrid();
    runCNN(preprocessing_us);

    drawing = false;
    delay(2000);

    resetGrid();
    showReadyScreen();
  }
}