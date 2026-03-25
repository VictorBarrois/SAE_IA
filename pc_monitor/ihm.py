import re
import socket
import threading
import queue
import serial
import serial.tools.list_ports
import customtkinter as ctk

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

GRID_SIZE = 28
TCP_PORT_DEFAULT = "12345"


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Reconnaissance de chiffres - M5Stack CoreS3")
        self.geometry("1220x760")
        self.minsize(1100, 700)

        self.serial_port = None
        self.tcp_socket = None
        self.reader_thread = None
        self.running = False
        self.queue = queue.Queue()

        self.current_digit = ctk.StringVar(value="-")
        self.current_conf = ctk.StringVar(value="0.00 %")
        self.current_time = ctk.StringVar(value="0.000000 s")
        self.status_text = ctk.StringVar(value="Déconnecté")

        self.connection_mode = ctk.StringVar(value="WiFi")
        self.selected_port = ctk.StringVar(value="")
        self.selected_baud = ctk.StringVar(value="115200")

        self.wifi_host = ctk.StringVar(value="192.168.4.1")
        self.wifi_port = ctk.StringVar(value=TCP_PORT_DEFAULT)

        self.prob_labels = []
        self.prob_values = []

        self.grid_data = [[0.0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.grid_rects = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

        self.collecting_grid = False
        self.current_grid_lines = []

        self._build_ui()
        self.refresh_ports()
        self.after(100, self.process_queue)

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(1, weight=1)

        # ---------------- Top frame ----------------
        top_frame = ctk.CTkFrame(self)
        top_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=16, pady=16)

        for i in range(12):
            top_frame.grid_columnconfigure(i, weight=1)

        ctk.CTkLabel(
            top_frame,
            text="Mode",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, padx=8, pady=8, sticky="w")

        self.mode_menu = ctk.CTkOptionMenu(
            top_frame,
            variable=self.connection_mode,
            values=["Série", "WiFi"],
            command=lambda _: self.update_connection_fields()
        )
        self.mode_menu.grid(row=0, column=1, padx=8, pady=8, sticky="ew")

        ctk.CTkLabel(
            top_frame,
            text="Port série",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=2, padx=8, pady=8, sticky="w")

        self.port_menu = ctk.CTkOptionMenu(top_frame, variable=self.selected_port, values=[""])
        self.port_menu.grid(row=0, column=3, padx=8, pady=8, sticky="ew")

        self.refresh_button = ctk.CTkButton(top_frame, text="Rafraîchir", command=self.refresh_ports)
        self.refresh_button.grid(row=0, column=4, padx=8, pady=8)

        ctk.CTkLabel(
            top_frame,
            text="Baudrate",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=5, padx=8, pady=8, sticky="w")

        self.baud_menu = ctk.CTkOptionMenu(
            top_frame,
            variable=self.selected_baud,
            values=["9600", "115200", "230400"]
        )
        self.baud_menu.grid(row=0, column=6, padx=8, pady=8, sticky="ew")

        ctk.CTkLabel(
            top_frame,
            text="IP ESP32",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=1, column=0, padx=8, pady=8, sticky="w")

        self.ip_entry = ctk.CTkEntry(top_frame, textvariable=self.wifi_host)
        self.ip_entry.grid(row=1, column=1, padx=8, pady=8, sticky="ew")

        ctk.CTkLabel(
            top_frame,
            text="Port TCP",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=1, column=2, padx=8, pady=8, sticky="w")

        self.tcp_port_entry = ctk.CTkEntry(top_frame, textvariable=self.wifi_port)
        self.tcp_port_entry.grid(row=1, column=3, padx=8, pady=8, sticky="ew")

        self.connect_button = ctk.CTkButton(top_frame, text="Connecter", command=self.toggle_connection)
        self.connect_button.grid(row=1, column=4, padx=8, pady=8)

        self.status_label = ctk.CTkLabel(
            top_frame,
            textvariable=self.status_text,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.status_label.grid(row=1, column=5, columnspan=4, padx=8, pady=8, sticky="ew")

        wifi_info = "WiFi carte : SSID = ESP32_IA_PROJECT | Mot de passe = GeiiTailscale2024$ | IP = 192.168.4.1 | Port = 12345"
        ctk.CTkLabel(
            top_frame,
            text=wifi_info,
            font=ctk.CTkFont(size=13)
        ).grid(row=2, column=0, columnspan=12, padx=8, pady=(4, 10), sticky="w")

        # ---------------- Left frame ----------------
        left_frame = ctk.CTkFrame(self)
        left_frame.grid(row=1, column=0, sticky="nsew", padx=(16, 8), pady=(0, 16))
        left_frame.grid_columnconfigure(0, weight=1)
        left_frame.grid_rowconfigure(3, weight=1)

        ctk.CTkLabel(left_frame, text="Résultat", font=ctk.CTkFont(size=24, weight="bold")).grid(
            row=0, column=0, padx=16, pady=(16, 10), sticky="w"
        )

        self.digit_card = ctk.CTkFrame(left_frame)
        self.digit_card.grid(row=1, column=0, sticky="ew", padx=16, pady=8)
        self.digit_card.grid_columnconfigure((0, 1, 2), weight=1)

        ctk.CTkLabel(
            self.digit_card, text="Chiffre reconnu", font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=0, padx=12, pady=(16, 6), sticky="w")

        ctk.CTkLabel(
            self.digit_card, text="Confiance", font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=1, padx=12, pady=(16, 6), sticky="w")

        ctk.CTkLabel(
            self.digit_card, text="Temps inférence", font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=2, padx=12, pady=(16, 6), sticky="w")

        self.digit_value = ctk.CTkLabel(
            self.digit_card, textvariable=self.current_digit, font=ctk.CTkFont(size=84, weight="bold")
        )
        self.digit_value.grid(row=1, column=0, padx=12, pady=(0, 18), sticky="n")

        self.conf_value = ctk.CTkLabel(
            self.digit_card, textvariable=self.current_conf, font=ctk.CTkFont(size=34, weight="bold")
        )
        self.conf_value.grid(row=1, column=1, padx=12, pady=(0, 18), sticky="n")

        self.time_value = ctk.CTkLabel(
            self.digit_card, textvariable=self.current_time, font=ctk.CTkFont(size=28, weight="bold")
        )
        self.time_value.grid(row=1, column=2, padx=12, pady=(0, 18), sticky="n")

        ctk.CTkLabel(left_frame, text="Grille 28 x 28", font=ctk.CTkFont(size=20, weight="bold")).grid(
            row=2, column=0, padx=16, pady=(18, 8), sticky="w"
        )

        grid_frame = ctk.CTkFrame(left_frame)
        grid_frame.grid(row=3, column=0, sticky="nsew", padx=16, pady=(0, 16))
        grid_frame.grid_rowconfigure(0, weight=1)
        grid_frame.grid_columnconfigure(0, weight=1)

        self.grid_canvas = ctk.CTkCanvas(
            grid_frame, width=420, height=420, bg="#d9d9d9", highlightthickness=0
        )
        self.grid_canvas.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

        self.init_grid_canvas()

        ctk.CTkLabel(
            left_frame,
            text="Rendu demandé : noir = 1.0 | gris = 0.5 | blanc = 0.0",
            font=ctk.CTkFont(size=13)
        ).grid(row=4, column=0, padx=16, pady=(0, 10), sticky="w")

        # ---------------- Right frame ----------------
        right_frame = ctk.CTkFrame(self)
        right_frame.grid(row=1, column=1, sticky="nsew", padx=(8, 16), pady=(0, 16))
        right_frame.grid_columnconfigure(0, weight=1)
        right_frame.grid_rowconfigure(3, weight=1)

        ctk.CTkLabel(right_frame, text="Probabilités", font=ctk.CTkFont(size=24, weight="bold")).grid(
            row=0, column=0, padx=16, pady=(16, 10), sticky="w"
        )

        probs_container = ctk.CTkFrame(right_frame)
        probs_container.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 16))
        probs_container.grid_columnconfigure(1, weight=1)

        for i in range(10):
            label = ctk.CTkLabel(
                probs_container, text=f"{i}", width=30, font=ctk.CTkFont(size=18, weight="bold")
            )
            label.grid(row=i, column=0, padx=(12, 8), pady=8, sticky="w")

            progress = ctk.CTkProgressBar(probs_container, height=18)
            progress.grid(row=i, column=1, padx=8, pady=8, sticky="ew")
            progress.set(0)

            value = ctk.CTkLabel(probs_container, text="0.00 %", width=90)
            value.grid(row=i, column=2, padx=(8, 12), pady=8, sticky="e")

            self.prob_labels.append(progress)
            self.prob_values.append(value)

        ctk.CTkLabel(right_frame, text="Logs", font=ctk.CTkFont(size=20, weight="bold")).grid(
            row=2, column=0, padx=16, pady=(8, 8), sticky="w"
        )

        self.log_box = ctk.CTkTextbox(right_frame, height=260)
        self.log_box.grid(row=3, column=0, sticky="nsew", padx=16, pady=(0, 16))
        self.log_box.insert("end", "Prêt.\n")
        self.log_box.configure(state="disabled")

        self.update_connection_fields()

    def update_connection_fields(self):
        mode = self.connection_mode.get()
        serial_state = "normal" if mode == "Série" else "disabled"
        wifi_state = "normal" if mode == "WiFi" else "disabled"

        self.port_menu.configure(state=serial_state)
        self.refresh_button.configure(state=serial_state)
        self.baud_menu.configure(state=serial_state)

        self.ip_entry.configure(state=wifi_state)
        self.tcp_port_entry.configure(state=wifi_state)

    def init_grid_canvas(self):
        self.grid_canvas.delete("all")
        canvas_size = 420
        cell = canvas_size / GRID_SIZE

        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                x0 = x * cell
                y0 = y * cell
                x1 = x0 + cell
                y1 = y0 + cell

                rect = self.grid_canvas.create_rectangle(
                    x0, y0, x1, y1, fill="#ffffff", outline="#cfcfcf"
                )
                self.grid_rects[y][x] = rect

    def color_for_value(self, value):
        if value >= 0.75:
            return "#000000"   # 1.0 -> noir
        if value >= 0.25:
            return "#808080"   # 0.5 -> gris
        return "#ffffff"       # 0.0 -> blanc

    def update_grid_canvas(self):
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                color = self.color_for_value(self.grid_data[y][x])
                self.grid_canvas.itemconfig(self.grid_rects[y][x], fill=color)

    def clear_probabilities(self):
        for i in range(10):
            self.prob_labels[i].set(0)
            self.prob_values[i].configure(text="0.00 %")

    def refresh_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        if not ports:
            ports = ["Aucun port détecté"]

        self.port_menu.configure(values=ports)

        if ports[0] != "Aucun port détecté":
            self.selected_port.set(ports[0])
        else:
            self.selected_port.set("Aucun port détecté")

    def toggle_connection(self):
        if self.running:
            self.disconnect_transport()
        else:
            if self.connection_mode.get() == "Série":
                self.connect_serial()
            else:
                self.connect_wifi()

    def connect_serial(self):
        port = self.selected_port.get()
        if not port or port == "Aucun port détecté":
            self.append_log("Aucun port série valide sélectionné.")
            return

        try:
            self.serial_port = serial.Serial(
                port=port,
                baudrate=int(self.selected_baud.get()),
                timeout=1
            )
            self.running = True
            self.reader_thread = threading.Thread(target=self.read_serial_loop, daemon=True)
            self.reader_thread.start()

            self.status_text.set(f"Connecté en série : {port}")
            self.connect_button.configure(text="Déconnecter")
            self.append_log(f"Connexion série ouverte sur {port} à {self.selected_baud.get()} bauds.")
        except Exception as e:
            self.append_log(f"Erreur connexion série : {e}")

    def connect_wifi(self):
        host = self.wifi_host.get().strip()
        port_text = self.wifi_port.get().strip()

        if not host:
            self.append_log("Adresse IP ESP32 manquante.")
            return

        try:
            port = int(port_text)
        except ValueError:
            self.append_log("Port TCP invalide.")
            return

        try:
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.settimeout(5)
            self.tcp_socket.connect((host, port))
            self.tcp_socket.settimeout(1.0)

            self.running = True
            self.reader_thread = threading.Thread(target=self.read_wifi_loop, daemon=True)
            self.reader_thread.start()

            self.status_text.set(f"Connecté en WiFi : {host}:{port}")
            self.connect_button.configure(text="Déconnecter")
            self.append_log(f"Connexion WiFi TCP ouverte sur {host}:{port}.")
        except Exception as e:
            self.append_log(f"Erreur connexion WiFi : {e}")
            try:
                if self.tcp_socket:
                    self.tcp_socket.close()
            except Exception:
                pass
            self.tcp_socket = None

    def disconnect_transport(self):
        self.running = False

        try:
            if self.serial_port:
                self.serial_port.close()
        except Exception:
            pass

        try:
            if self.tcp_socket:
                self.tcp_socket.close()
        except Exception:
            pass

        self.serial_port = None
        self.tcp_socket = None

        self.status_text.set("Déconnecté")
        self.connect_button.configure(text="Connecter")
        self.append_log("Connexion fermée.")

    def read_serial_loop(self):
        while self.running and self.serial_port and self.serial_port.is_open:
            try:
                line = self.serial_port.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    self.queue.put(line)
            except Exception as e:
                self.queue.put(f"[ERREUR SERIE] {e}")
                break

    def read_wifi_loop(self):
        buffer = ""

        while self.running and self.tcp_socket:
            try:
                data = self.tcp_socket.recv(4096)
                if not data:
                    self.queue.put("[INFO] Connexion WiFi fermée par l'ESP32.")
                    break

                buffer += data.decode("utf-8", errors="ignore")

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if line:
                        self.queue.put(line)

            except socket.timeout:
                continue
            except Exception as e:
                self.queue.put(f"[ERREUR WIFI] {e}")
                break

    def process_queue(self):
        while not self.queue.empty():
            line = self.queue.get()
            self.append_log(line)
            self.parse_line(line)

        self.after(100, self.process_queue)

    def append_log(self, text):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def parse_grid_line(self, line):
        tokens = line.split()
        if len(tokens) != GRID_SIZE:
            return False

        try:
            row = [float(tok) for tok in tokens]
        except ValueError:
            return False

        self.current_grid_lines.append(row)
        return True

    def finalize_grid(self):
        if len(self.current_grid_lines) == GRID_SIZE:
            self.grid_data = self.current_grid_lines
            self.update_grid_canvas()

        self.current_grid_lines = []
        self.collecting_grid = False

    def parse_line(self, line):
        if "============= GRID =============" in line:
            self.collecting_grid = True
            self.current_grid_lines = []
            return

        if "================================" in line and self.collecting_grid:
            self.finalize_grid()
            return

        if self.collecting_grid:
            if self.parse_grid_line(line):
                return

        m_digit = re.search(r"Chiffre reconnu\s*:\s*(\d)\s*-->\s*([\d.]+)%", line)
        if m_digit:
            digit = m_digit.group(1)
            conf = float(m_digit.group(2))
            self.current_digit.set(digit)
            self.current_conf.set(f"{conf:.2f} %")
            return

        m_time = re.search(r"Temps d['’]inference\s*:\s*([\d.]+)\s*s", line)
        if m_time:
            inf_time = float(m_time.group(1))
            self.current_time.set(f"{inf_time:.6f} s")
            return

        m_prob = re.search(r"^(\d)\s*:\s*([\d.]+)%$", line)
        if m_prob:
            idx = int(m_prob.group(1))
            val = float(m_prob.group(2))
            self.prob_labels[idx].set(max(0.0, min(1.0, val / 100.0)))
            self.prob_values[idx].configure(text=f"{val:.2f} %")
            return

    def on_close(self):
        self.disconnect_transport()
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()