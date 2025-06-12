import json

class GestorUsuarios:
    """Manages user registration, chat history association, and persistence of user data to a JSON file."""
    def __init__(self, archivo='usuarios.json'):
        self.usuarios = {}  # {nombre_usuario: historial_id}
        self.contador = 0
        self.archivo = archivo
        self.cargar() # Loads user data and counter from the JSON file on initialization.

    # Registers a new user, assigns a unique history ID, and saves to file.
    def registrar_usuario(self, nombre):
        self.contador += 1
        self.usuarios[nombre] = f"historial_{self.contador}"
        self.guardar() # Persists the current state of users and history ID counter to the JSON file.
        return self.usuarios[nombre]
    def obtener_historial(self, nombre):
        return self.usuarios.get(nombre)
    def listar_usuarios(self):
        return list(self.usuarios.keys())

    # Persists the current state of users and history ID counter to the JSON file.
    def guardar(self):
        with open(self.archivo, 'w', encoding='utf-8') as f:
            json.dump({'usuarios': self.usuarios, 'contador': self.contador}, f)

    # Loads user data and counter from the JSON file on initialization.
    def cargar(self):
        try:
            with open(self.archivo, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.usuarios = data.get('usuarios', {})
                self.contador = data.get('contador', 0)
        except (FileNotFoundError, json.JSONDecodeError):
            self.usuarios = {}
            self.contador = 0 