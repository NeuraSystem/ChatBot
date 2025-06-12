# tests/test_chatbot.py

import unittest
from unittest.mock import patch, MagicMock
from src.services import ServiceContainer

# Usaremos un ServiceContainer para simular el entorno de la aplicación real
# Esto hace que las pruebas sean más de integración y menos propensas a romperse
# con cambios internos.

class TestChatbotFlow(unittest.TestCase):
    
    @classmethod
    @patch('src.rag.embeddings.SentenceTransformer') # Mock para no descargar el modelo real
    @patch('src.langgraph_service.ChatAnthropic') # Mock para no llamar a la API real
    def setUpClass(cls, mock_chat_anthropic, mock_sentence_transformer):
        """
        Configura el entorno de prueba una vez para toda la clase.
        Esto es más eficiente que hacerlo para cada test.
        """
        print("Configurando el entorno de prueba...")
        
        # Configuramos los mocks globales
        mock_chat_anthropic.return_value.invoke.return_value = MagicMock(content="Respuesta de prueba")
        # Mock devuelve embeddings válidos (lista de floats)
        mock_sentence_transformer.return_value.encode.return_value = [0.1, 0.2, 0.3]
        mock_sentence_transformer.return_value.embed_query.return_value = [0.1, 0.2, 0.3]
        
        # Creamos una instancia del contenedor de servicios, que inicializará todo
        cls.services = ServiceContainer()
        # Sobrescribimos el retriever con un mock para controlar sus respuestas
        cls.services.rag_retriever = MagicMock()
        print("Entorno de prueba configurado.")

    def test_01_user_registration(self):
        """Prueba que un usuario pueda registrarse."""
        manager = self.services.user_manager
        user_count_before = len(manager.usuarios)
        manager.registrar_usuario("test_user")
        user_count_after = len(manager.usuarios)
        self.assertEqual(user_count_after, user_count_before + 1)
        self.assertIn("test_user", manager.usuarios)

    def test_02_send_message_without_rag(self):
        """Prueba enviar un mensaje simple sin que RAG esté necesariamente involucrado."""
        chatbot = self.services.chatbot
        # Aseguramos que el retriever no devuelva nada para forzar una respuesta no-RAG
        self.services.rag_retriever.get_retriever.return_value.invoke.return_value = []
        
        response = chatbot.send_message("Hola", "historial_simple")
        self.assertEqual(response, "Respuesta de prueba")

    def test_03_document_service_add_and_clear(self):
        """Prueba que se puedan añadir y limpiar documentos a través del servicio."""
        doc_service = self.services.document_service
        # Aseguramos que el servicio esté disponible (si RAG_ENABLED es True)
        if doc_service:
            # Probamos añadir documentos
            doc_service.add_documents(["dummy_document.pdf"])
            # Verificamos que se llamó al método subyacente del retriever
            self.services.rag_retriever.add_documents.assert_called_once()

            # Probamos limpiar documentos
            doc_service.clear_documents()
            self.services.rag_retriever.clear_collection.assert_called_once()
        else:
            self.skipTest("El servicio de documentos está deshabilitado (RAG_ENABLED=False)")

# =============================================================================
# CAMBIO: Se eliminaron las clases de test individuales y se unificaron en un
# flujo de prueba más robusto y realista que utiliza el ServiceContainer.
# Esto evita errores cuando las clases dependen unas de otras y prueba la
# colaboración entre ellas.
# =============================================================================

if __name__ == "__main__":
    unittest.main()
