#include <M5GFX.h>
#include <../CNN/mnist_float32.h>
#include <math.h>

M5GFX display;

#define GRID_SIZE 28

// IMPORTANT : float et pas int !
float grid[GRID_SIZE][GRID_SIZE];

void printGrid();
void runCNN();


// =====================================================
// SOFTMAX (stable)
// =====================================================
void softmax(float *input, float *output, int size) {

  float maxVal = input[0];

  for(int i = 1; i < size; i++){
    if(input[i] > maxVal) maxVal = input[i];
  }

  float sum = 0.0f;
  for(int i = 0; i < size; i++){
    output[i] = expf(input[i] - maxVal);
    sum += output[i];
  }

  for(int i = 0; i < size; i++){
    output[i] /= sum;
  }
}


// =====================================================
// SETUP
// =====================================================
void setup() {

  Serial.begin(115200);

  display.init();
  display.startWrite();
  display.fillScreen(TFT_BLACK);

  // reset grille
  for(int y = 0; y < GRID_SIZE; y++){
    for(int x = 0; x < GRID_SIZE; x++){
      grid[y][x] = 0.0f;
    }
  }
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

    // Dessin écran
    display.fillCircle(x, y, 4, TFT_WHITE);

    // écran -> grille
    int gx = map(x, 0, 320, 0, 27);
    int gy = map(y, 0, 240, 0, 27);

    // ===============================
    // PIXEL CENTRAL = 1.0
    // ===============================
    if(gx >= 0 && gx < GRID_SIZE && gy >= 0 && gy < GRID_SIZE){
      grid[gy][gx] = 1.0f;
    }

    // ===============================
    // ÉPAISSIR = 0.5 AUTOUR
    // ===============================
    for(int dy = -1; dy <= 1; dy++){
      for(int dx = -1; dx <= 1; dx++){

        int nx = gx + dx;
        int ny = gy + dy;

        if(nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE){

          // ne pas toucher au centre
          if(dx != 0 || dy != 0){

            // ne pas écraser un pixel déjà plus fort
            if(grid[ny][nx] < 0.5f)
              grid[ny][nx] = 0.5f;
          }
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
        grid[y][x] = 0.0f;
      }
    }
  }
}


// =====================================================
// EXECUTION CNN + SOFTMAX
// =====================================================
void runCNN() {

  input_t input;
  dense_51_output_type output;

  // ⭐ IMPORTANT : on garde les valeurs float
  for(int y = 0; y < 28; y++){
    for(int x = 0; x < 28; x++){
      input[y][x][0] = grid[y][x];
    }
  }

  cnn(input, output);

  float probs[10];
  softmax(output, probs, 10);

  int predicted = 0;
  float maxVal = probs[0];

  for(int i = 1; i < 10; i++){
    if(probs[i] > maxVal){
      maxVal = probs[i];
      predicted = i;
    }
  }

  Serial.println("=========== RESULTAT CNN ===========");

  for(int i = 0; i < 10; i++){
    Serial.print(i);
    Serial.print(" : ");
    Serial.print(probs[i]*100, 4);
    Serial.println("%");
  }

  Serial.print("Chiffre reconnu : ");
  Serial.print(predicted);
  Serial.print(" --> ");
  Serial.print(probs[predicted]*100);
  Serial.println("%");

  Serial.println("====================================");
}


// =====================================================
// DEBUG : affichage grille
// =====================================================
void printGrid() {

  Serial.println("============= GRID =============");

  for(int y = 0; y < GRID_SIZE; y++){
    for(int x = 0; x < GRID_SIZE; x++){
      Serial.print(grid[y][x], 1); // affiche 0.0 / 0.5 / 1.0
      Serial.print(" ");
    }
    Serial.println();
  }

  Serial.println("================================");
}
