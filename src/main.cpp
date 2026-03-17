#include <M5GFX.h>
#include <../CNN/mnist_float32.h>
#include <math.h>

M5GFX display;

#define GRID_SIZE 28

int grid[GRID_SIZE][GRID_SIZE];

void printGrid();
void runCNN();


// =====================================================
// SOFTMAX (stable numériquement)
// =====================================================
void softmax(float *input, float *output, int size) {

  float maxVal = input[0];

  // trouver le max
  for(int i = 1; i < size; i++){
    if(input[i] > maxVal) maxVal = input[i];
  }

  // exp + somme
  float sum = 0.0f;
  for(int i = 0; i < size; i++){
    output[i] = expf(input[i] - maxVal);
    sum += output[i];
  }

  // normalisation
  for(int i = 0; i < size; i++){
    output[i] /= sum;
  }
}


// =====================================================
// SETUP
// =====================================================
void setup() {

  Serial.begin(115200);
  Serial.println("Programme démarre");

  display.init();
  display.startWrite();
  display.fillScreen(TFT_BLACK);

  // init grille
  for(int y = 0; y < GRID_SIZE; y++){
    for(int x = 0; x < GRID_SIZE; x++){
      grid[y][x] = 0;
    }
  }

  Serial.println("Grille initialisée");
}


// =====================================================
// LOOP : dessin tactile
// =====================================================
void loop() {

  lgfx::touch_point_t tp[1];
  int nums = display.getTouchRaw(tp, 1);

  static bool drawing = false;

  if(nums) {

    drawing = true;

    int x = tp[0].x;
    int y = tp[0].y;

    // dessiner sur écran
    display.fillCircle(x, y, 4, TFT_WHITE);

    // écran -> grille
    int gx = map(x, 0, 320, 0, 27);
    int gy = map(y, 0, 240, 0, 27);

    // épaissir le trait
    for(int dy = -1; dy <= 1; dy++){
      for(int dx = -1; dx <= 1; dx++){

        int nx = gx + dx;
        int ny = gy + dy;

        if(nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE){
          grid[ny][nx] = 1;
        }
      }
    }

  } 
  else if(drawing) {

    // doigt relâché

    printGrid();
    runCNN();

    drawing = false;

    // reset écran
    display.fillScreen(TFT_BLACK);

    // reset grille
    for(int y = 0; y < GRID_SIZE; y++){
      for(int x = 0; x < GRID_SIZE; x++){
        grid[y][x] = 0;
      }
    }

    Serial.println("Dessin terminé --> Doigt relâché");
  }
}


// =====================================================
//  EXECUTION CNN + SOFTMAX
// =====================================================
void runCNN() {

  input_t input;
  dense_47_output_type output;

  // grid -> input CNN
  for(int y = 0; y < 28; y++){
    for(int x = 0; x < 28; x++){
      input[y][x][0] = grid[y][x] ? 1.0f : 0.0f;
    }
  }

  // exécution réseau
  cnn(input, output);

  // Softmax
  float probs[10];
  softmax(output, probs, 10);

  // trouver classe la plus probable
  int predicted = 0;
  float maxVal = probs[0];

  for(int i = 1; i < 10; i++){
    if(probs[i] > maxVal){
      maxVal = probs[i];
      predicted = i;
    }
  }

  // ================= AFFICHAGE =================

  Serial.println("================ RESULTAT CNN ================");

  Serial.println("Probabilités (softmax) :");

  for(int i = 0; i < 10; i++){
    Serial.print(i);
    Serial.print(" : ");
    Serial.print(probs[i]*100, 5);
    Serial.println(" % ");
  }

  Serial.print("Chiffre reconnu : ");
  Serial.println(predicted);

  Serial.println("================================================");
}


// =====================================================
// DEBUG : affichage grille
// =====================================================
void printGrid() {

  Serial.println("================ GRID 28x28 ================");

  for(int y = 0; y < GRID_SIZE; y++){
    for(int x = 0; x < GRID_SIZE; x++){
      Serial.print(grid[y][x]);
      Serial.print(" ");
    }
    Serial.println();
  }

  Serial.println("==============================================");
}
