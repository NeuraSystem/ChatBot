# tests/test_integration.py

import unittest
from unittest.mock import patch, MagicMock
import numpy as np # Import numpy
from langchain_core.documents import Document # Added Document import
from src.services import ServiceContainer

class TestIntegration(unittest.TestCase):
    
    # =============================================================================
    # CAMBIO: Se añaden los 'patch' para simular las librerías externas.
    # Esto evita que se conecte a internet y soluciona el RepositoryNotFoundError.
    # =============================================================================
    @patch('sentence_transformers.SentenceTransformer')
    @patch('langchain_community.chat_models.ChatAnthropic')
    @patch('anthropic.Anthropic')  # Patch the actual Anthropic SDK client
    def setUp(self, mock_anthropic_sdk_client_class, mock_langchain_chat_anthropic_class, mock_sentence_transformer_class): # Order of args matches decorators
        """
        Configura un entorno de prueba limpio para cada test de integración,
        simulando las dependencias externas.
        """
        # 1. Configure the mock for the anthropic.Anthropic SDK client instance
        mock_sdk_client_instance = mock_anthropic_sdk_client_class.return_value
        mock_sdk_client_instance.count_tokens = MagicMock(return_value=5) # For Pydantic validation

        mock_completions_create_result = MagicMock()
        # This content should match what the test expects after Langchain adapter processing
        mock_completions_create_result.completion = "Respuesta integrada"
        mock_sdk_client_instance.completions = MagicMock()
        mock_sdk_client_instance.completions.create = MagicMock(return_value=mock_completions_create_result)

        # 2. Configure the mock for Langchain's ChatAnthropic instance
        mock_chat_model_instance = mock_langchain_chat_anthropic_class.return_value
        # mock_chat_model_instance.invoke.return_value = MagicMock(content="Respuesta integrada") # Not strictly needed if SDK mock is correct
        mock_chat_model_instance.get_num_tokens = MagicMock(return_value=10) # For MessageTrimmer

        # 3. Configure mock for SentenceTransformer instance
        mock_st_instance = mock_sentence_transformer_class.return_value
        def mock_encode_method(texts_or_text, *args, **kwargs):
            if isinstance(texts_or_text, str):
                return np.array([0.1, 0.2, 0.3])
            elif isinstance(texts_or_text, list):
                return np.array([[0.1, 0.2, 0.3] for _ in texts_or_text])
            return np.array([])
        mock_st_instance.encode = MagicMock(side_effect=mock_encode_method)
        
        # Ahora creamos el contenedor de servicios con las dependencias ya mockeadas
        self.services = ServiceContainer()
        self.gestor = self.services.user_manager
        self.chatbot = self.services.chatbot
        self.document_service = self.services.document_service # This holds the real RAGRetriever

        # If RAG is enabled, mock methods on the actual RAGRetriever instance
        # that self.document_service will use.
        if self.services.rag_retriever:
            self.rag_retriever_add_documents_mock = MagicMock()
            self.rag_retriever_clear_documents_mock = MagicMock()

            self.services.rag_retriever.add_documents = self.rag_retriever_add_documents_mock
            self.services.rag_retriever.clear_documents = self.rag_retriever_clear_documents_mock
            # For RAG part of send_message, if it tries to retrieve, ensure it gets empty list
            # Note: LangGraphService gets its retriever from RAGRetriever.get_retriever()
            # So, we mock RAGRetriever.get_retriever() to return a retriever mock.
            retriever_mock = MagicMock()
            retriever_mock.invoke.return_value = [] # No docs found
            self.services.rag_retriever.get_retriever = MagicMock(return_value=retriever_mock)


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
            # Mock DocumentLoader para esta sección de la prueba para evitar I/O real
            # y controlar los Documentos que se pasan al retriever
            mock_loaded_docs = [MagicMock(spec=Document)]
            self.services.document_loader.load_multiple_documents = MagicMock(
                return_value=mock_loaded_docs
            )
            # No need to replace self.document_service.rag_retriever

            # Simular carga de documento
            resultado_add = self.document_service.add_documents(["test_integracion.txt"])
            self.assertIn("documentos solicitados", resultado_add)
            # Check the mock on the actual retriever instance held by document_service
            self.rag_retriever_add_documents_mock.assert_called_once_with(mock_loaded_docs)
            
            # Limpiar documentos
            resultado_clear = self.document_service.clear_documents()
            self.assertIn("Documentos limpiados", resultado_clear)
            self.rag_retriever_clear_documents_mock.assert_called_once() # Assert on the mocked method
        else:
            self.skipTest("Prueba de documentos omitida porque RAG_ENABLED es False.")

if __name__ == "__main__":
    unittest.main()
