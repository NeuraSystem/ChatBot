# tests/test_integration.py

import unittest
from unittest.mock import patch, MagicMock
from src.services import ServiceContainer

class TestIntegration(unittest.TestCase):
    
    # =============================================================================
    # CAMBIO: Se añaden los 'patch' para simular las librerías externas.
    # Esto evita que se conecte a internet y soluciona el RepositoryNotFoundError.
    # =============================================================================
    @patch('src.rag.embeddings.SentenceTransformer')
    @patch('src.langgraph_service.ChatAnthropic')
    def setUp(self, mock_chat_anthropic, mock_sentence_transformer):
        """
        Configura un entorno de prueba limpio para cada test de integración,
        simulando las dependencias externas.
        """
        # Configuramos los mocks para que devuelvan valores predecibles
        mock_chat_anthropic.return_value.invoke.return_value = MagicMock(content="Respuesta integrada")
        mock_sentence_transformer.return_value.encode.return_value = [[0.1, 0.2, 0.3]]
        
        # Ahora creamos el contenedor de servicios con las dependencias ya mockeadas
        self.services = ServiceContainer()
        self.gestor = self.services.user_manager
        self.chatbot = self.services.chatbot
        self.document_service = self.services.document_service

    def test_flujo_completo(self):
        """
        Prueba un flujo completo: registrar usuario, enviar mensaje y gestionar documentos.
        """
        # 1. Registrar usuario
        historial_id = self.gestor.registrar_usuario("testuser_integracion")
        self.assertIn("testuser_integracion", self.gestor.usuarios)
        self.assertIsNotNone(historial_id)
        
        # 2. Enviar mensaje
        # Usamos el historial_id obtenido para la conversación
        respuesta = self.chatbot.send_message("Hola integración", historial_id)
        self.assertIsInstance(respuesta, str)
        self.assertEqual(respuesta, "Respuesta integrada")
        
        # 3. Gestionar documentos (si RAG está habilitado)
        if self.document_service:
            # Sobrescribimos el retriever interno con un mock para esta prueba
            self.document_service.rag_retriever = MagicMock()

            # Simular carga de documento
            resultado_add = self.document_service.add_documents(["test_integracion.txt"])
            self.assertIn("documentos solicitados", resultado_add)
            # Verificamos que el método subyacente fue llamado
            self.document_service.rag_retriever.add_documents.assert_called_once_with(["test_integracion.txt"])
            
            # Limpiar documentos
            resultado_clear = self.document_service.clear_documents()
            self.assertIn("Documentos limpiados", resultado_clear)
            self.document_service.rag_retriever.clear_collection.assert_called_once()
        else:
            self.skipTest("Prueba de documentos omitida porque RAG_ENABLED es False.")

if __name__ == "__main__":
    unittest.main()
