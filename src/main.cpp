#include <M5GFX.h>

M5GFX display;

#define GRID_SIZE 28

int grid[GRID_SIZE][GRID_SIZE];

void printGrid();  // prototype pour afficher la grille

void setup() {
  Serial.begin(115200);
  Serial.println("Programme démarre");

  display.init();
  display.startWrite();

  // Fond noir
  display.fillScreen(TFT_BLACK);

  // Initialiser la grille
  for(int y = 0; y < GRID_SIZE; y++){
    for(int x = 0; x < GRID_SIZE; x++){
      grid[y][x] = 0;
    }
  }

  Serial.println("Grille initialisée");
}

void loop() {
  lgfx::touch_point_t tp[1];
  int nums = display.getTouchRaw(tp, 1);

  static bool drawing = false; // savoir si on était en train de dessiner

  if(nums) { // le doigt est sur l'écran
    drawing = true;

    int x = tp[0].x;
    int y = tp[0].y;

    // dessiner sur l'écran en blanc
    display.fillCircle(x, y, 4, TFT_WHITE);

    // conversion 320x240 -> 28x28
    int gx = map(x, 0, 320, 0, 27);
    int gy = map(y, 0, 240, 0, 27);

    // --- remplissage "plus gras" ---
    for(int dy = -1; dy <= 1; dy++){
      for(int dx = -1; dx <= 1; dx++){
        int nx = gx + dx;
        int ny = gy + dy;

        if(nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE){
          grid[ny][nx] = 1;
        }
      }
    }

  } else if(drawing) {

    // afficher la grille dans le terminal à chaque point
    printGrid();

    // le doigt a été retiré → réinitialiser écran et grille
    drawing = false;

    display.fillScreen(TFT_BLACK); // écran noir
    for(int y = 0; y < GRID_SIZE; y++){
      for(int x = 0; x < GRID_SIZE; x++){
        grid[y][x] = 0; // grille remise à zéro
      }
    }

    Serial.println("Dessin terminé --> Doigt relâché");
  }
}

void printGrid() {
  Serial.println("===================== GRID 28x28 ======================");
  for(int y = 0; y < GRID_SIZE; y++){
    for(int x = 0; x < GRID_SIZE; x++){
      Serial.print(grid[y][x]);
      Serial.print(" ");
    }
    Serial.println();
  }
  Serial.println("=======================================================");
}
