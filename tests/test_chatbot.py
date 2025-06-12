# tests/test_chatbot.py

import unittest
from unittest.mock import patch, MagicMock
import numpy as np # Import numpy
from langchain_core.documents import Document # Added Document import
from src.services import ServiceContainer

# Usaremos un ServiceContainer para simular el entorno de la aplicación real
# Esto hace que las pruebas sean más de integración y menos propensas a romperse
# con cambios internos.

class TestChatbotFlow(unittest.TestCase):
    
    @classmethod
    @patch('sentence_transformers.SentenceTransformer')
    @patch('langchain_community.chat_models.ChatAnthropic')
    @patch('anthropic.Anthropic')  # Patch the actual Anthropic SDK client
    def setUpClass(cls, mock_anthropic_sdk_client_class, mock_langchain_chat_anthropic_class, mock_sentence_transformer_class):
        """
        Configura el entorno de prueba una vez para toda la clase.
        """
        print("Configurando el entorno de prueba...")

        # 1. Configure the mock for the anthropic.Anthropic SDK client instance
        mock_sdk_client_instance = mock_anthropic_sdk_client_class.return_value
        mock_sdk_client_instance.count_tokens = MagicMock(return_value=5) # For Pydantic validation
        
        mock_completions_create_result = MagicMock()
        # This content should match what the test expects after Langchain adapter processing
        mock_completions_create_result.completion = "Respuesta de prueba"
        mock_sdk_client_instance.completions = MagicMock()
        mock_sdk_client_instance.completions.create = MagicMock(return_value=mock_completions_create_result)

        # 2. Configure the mock for Langchain's ChatAnthropic instance
        mock_chat_model_instance = mock_langchain_chat_anthropic_class.return_value
        # This .invoke will be called by LangGraphService. It should use the mocked SDK.
        # We don't strictly need to mock .invoke.return_value on mock_chat_model_instance itself anymore,
        # as the actual .invoke will run and use the mocked SDK.
        # However, keeping it doesn't hurt and provides a fallback if direct .invoke was tested.
        mock_chat_model_instance.invoke.return_value = MagicMock(content="Respuesta de prueba")
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
        
        # Initialize ServiceContainer. It should pick up the patched classes.
        cls.services = ServiceContainer()

        # Mock methods of the RAGRetriever instance obtained from ServiceContainer
        # Esto es para que DocumentService use el RAGRetriever real, pero con métodos controlados.
        if cls.services.rag_retriever:
            cls.services.rag_retriever.add_documents = MagicMock()
            cls.services.rag_retriever.clear_documents = MagicMock() # Mock clear_documents
            # Para test_02_send_message_without_rag, para asegurar que no haya documentos de RAG
            cls.services.rag_retriever.get_retriever = MagicMock(return_value=MagicMock(invoke=MagicMock(return_value=[])))
        else:
            # Si RAG_ENABLED es False, rag_retriever es None.
            # Para test_02, si retriever es None, LangGraphService no lo usará.
            # Para test_03, el test será skipeado por doc_service siendo None.
            pass

        # Mock DocumentLoader para evitar operaciones de archivo reales en tests
        if cls.services.document_loader: # Solo si RAG está habilitado
            mock_doc = MagicMock(spec=Document)
            # mock_doc.id = "dummy_doc_id_123" # ChromaDB genera IDs, no es necesario mockearlo en Document
            mock_doc.page_content = "Este es contenido de prueba."
            mock_doc.metadata = {"source": "dummy.pdf"}
            cls.services.document_loader.load_multiple_documents = MagicMock(
                return_value=[mock_doc]
            )
        print("Entorno de prueba configurado.")

    def test_01_user_registration(self):
        """Prueba que un usuario pueda registrarse."""
        manager = self.services.user_manager
        # Limpiar estado antes de la prueba para asegurar la independencia del test
        manager.usuarios = {}
        manager.contador = 0
        manager.guardar() # Asegurar que el archivo también se limpie/resetea

        user_count_before = len(manager.usuarios)
        manager.registrar_usuario("test_user")
        user_count_after = len(manager.usuarios)
        self.assertEqual(user_count_after, user_count_before + 1)
        self.assertIn("test_user", manager.usuarios)

    def test_02_send_message_without_rag(self):
        """Prueba enviar un mensaje simple sin que RAG esté necesariamente involucrado."""
        chatbot = self.services.chatbot
        # El mock de cls.services.rag_retriever.get_retriever en setUpClass
        # ya asegura que no se devuelvan documentos RAG.
        
        response = chatbot.send_message("Hola", "historial_simple")
        self.assertEqual(response, "Respuesta de prueba")

    def test_03_document_service_add_and_clear(self):
        """Prueba que se puedan añadir y limpiar documentos a través del servicio."""
        doc_service = self.services.document_service
        # Aseguramos que el servicio esté disponible (si RAG_ENABLED es True)
        if doc_service:
            # Probamos añadir documentos
            doc_service.add_documents(["dummy_document.pdf"]) # El path es ahora simbólico
            # Verificamos que se llamó al método subyacente del retriever
            # y que se llamó con la lista de documentos mockeados
            self.services.rag_retriever.add_documents.assert_called_once_with(
                self.services.document_loader.load_multiple_documents.return_value # Changed cls to self
            )

            # Probamos limpiar documentos
            doc_service.clear_documents()
            self.services.rag_retriever.clear_documents.assert_called_once() # Assert clear_documents
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
