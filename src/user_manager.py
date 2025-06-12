import json

class GestorUsuarios:
    def __init__(self, archivo='usuarios.json'):
        self.usuarios = {}  # {nombre_usuario: historial_id}
        self.contador = 0
        self.archivo = archivo
        self.cargar()
    def registrar_usuario(self, nombre):
        self.contador += 1
        self.usuarios[nombre] = f"historial_{self.contador}"
        self.guardar()
        return self.usuarios[nombre]
    def obtener_historial(self, nombre):
        return self.usuarios.get(nombre)
    def listar_usuarios(self):
        return list(self.usuarios.keys())
    def guardar(self):
        with open(self.archivo, 'w', encoding='utf-8') as f:
            json.dump({'usuarios': self.usuarios, 'contador': self.contador}, f)
    def cargar(self):
        try:
            with open(self.archivo, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.usuarios = data.get('usuarios', {})
                self.contador = data.get('contador', 0)
        except (FileNotFoundError, json.JSONDecodeError):
            self.usuarios = {}
            self.contador = 0 