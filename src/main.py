# chatbot-langchain-anthropic/src/main.py

import os
from dotenv import load_dotenv
from src.config import GlobalConfig as Config
from src.chatbot import Chatbot
from src.rag.document_loader import DocumentLoader
import re
import json
from src.user_manager import GestorUsuarios
from src.services import ServiceContainer


class ChatSession:
    def __init__(self, user_id=None):
        self.user_id = user_id
        self.historial = []
    def set_user(self, user_id):
        self.user_id = user_id
    def add_message(self, message):
        self.historial.append(message)
    def clear(self):
        self.historial = []


def main():
    services = ServiceContainer()
    gestor = services.user_manager
    chatbot = services.chatbot
    document_loader = services.document_service
    session = ChatSession()
    
    print("Bienvenido al chatbot!")
    print("Comandos disponibles:")
    print("  - registro <nombre>: Registrar un nuevo usuario")
    print("  - usuarios: Ver lista de usuarios registrados")
    print("  - historial <nombre>: Cambiar a la conversación de un usuario")
    print("  - new: Iniciar una conversación temporal")
    print("  - upload <archivo>: Subir un documento")
    print("  - list_docs: Listar documentos cargados")
    print("  - clear_docs: Limpiar documentos")
    print("  - exit: Salir del chatbot")
    print("\nEscribe tu mensaje o comando:")

    while True:
        user_input = input("\n> ").strip()
        
        if user_input.lower() == 'exit':
            print("¡Hasta luego!")
            break
        
        elif user_input.lower().startswith('registro '):
            nombre = user_input[9:].strip()
            if nombre:
                if not re.match(r'^[\w\-]+$', nombre):
                    print("Nombre de usuario inválido. Solo se permiten letras, números, guiones y guion bajo.")
                    continue
                historial_id = gestor.registrar_usuario(nombre)
                print(f"Usuario {nombre} registrado exitosamente")
                session.set_user(historial_id)
            else:
                print("Por favor, proporciona un nombre de usuario")
            continue
        
        elif user_input.lower() == 'usuarios':
            usuarios = gestor.listar_usuarios()
            if usuarios:
                print("Usuarios registrados:")
                for usuario in usuarios:
                    print(f"- {usuario}")
            else:
                print("No hay usuarios registrados")
            continue
        
        elif user_input.lower().startswith('historial '):
            nombre = user_input[10:].strip()
            if nombre:
                historial_id = gestor.obtener_historial(nombre)
                if historial_id:
                    session.set_user(historial_id)
                    print(f"Cambiando a la conversación de {nombre}")
                else:
                    print(f"Usuario {nombre} no encontrado")
            else:
                print("Por favor, proporciona un nombre de usuario")
            continue
        
        elif user_input.lower() == 'new':
            session.set_user(None)
            session.clear()
            print("Iniciando conversación temporal")
            continue
        
        elif user_input.lower().startswith('upload '):
            try:
                archivo = user_input[7:].strip()
                if archivo:
                    document_loader.add_document(archivo)
                    print(f"Documento {archivo} cargado exitosamente")
                else:
                    print("Por favor, proporciona un archivo")
            except FileNotFoundError:
                print(f"Archivo no encontrado: {archivo}")
            except ValueError as ve:
                print(f"Error de valor: {ve}")
            except Exception as e:
                print(f"Error al cargar documento: {str(e)}")
            continue
        
        elif user_input.lower() == 'list_docs':
            try:
                docs = document_loader.list_documents()
                if docs:
                    print("Documentos cargados:")
                    for doc in docs:
                        print(f"- {doc}")
                else:
                    print("No hay documentos cargados")
            except Exception as e:
                print(f"Error al listar documentos: {str(e)}")
            continue
        
        elif user_input.lower() == 'clear_docs':
            try:
                document_loader.clear_documents()
                print("Documentos limpiados exitosamente")
            except Exception as e:
                print(f"Error al limpiar documentos: {str(e)}")
            continue
        
        # Enviar mensaje al chatbot con el historial actual
        response = chatbot.send_message(user_input, user_id=session.user_id)
        session.add_message(("user", user_input))
        session.add_message(("bot", response))
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()