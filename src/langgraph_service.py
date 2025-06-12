# src/langgraph_service.py

from langchain_core.messages import HumanMessage
# =============================================================================
# CAMBIO: Se importa ChatAnthropic desde 'langchain_community' para eliminar la advertencia.
# =============================================================================
from langchain_community.chat_models import ChatAnthropic
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import trim_messages
from .constants import MESSAGES
from src.config import config
from .rag.logging_config import logger

class LangGraphService:
    """
    Servicio que encapsula la configuración y ejecución de LangGraph.
    """
    def __init__(self, api_key, model, chat_history, retriever):
        self.api_key = api_key
        self.model = model
        self.chat_history = chat_history
        self.retriever = retriever
        self._setup_langgraph()

    def _setup_langgraph(self):
        try:
            self.llm = ChatAnthropic(
                anthropic_api_key=self.api_key,
                model_name=self.model,
                max_tokens=512
            )
            self.prompt_template = ChatPromptTemplate.from_messages([
                ("system", """Eres un asistente amigable y servicial. Responde de manera concisa y clara. Si hay contexto relevante, úsalo para dar respuestas más precisas, de lo contrario, responde con tu conocimiento general.""")
            ])
            self.trimmer = trim_messages(
                max_tokens=1000, strategy="last", token_counter=self.llm,
                include_system=True, allow_partial=False
            )
            
            workflow = StateGraph(dict)

            def call_model(state: dict):
                messages = state.get("messages", [])
                
                # Lógica RAG (si está habilitado y hay un retriever)
                if self.retriever and messages:
                    query = messages[-1].content
                    context_docs = self.retriever.invoke(query)
                    if context_docs:
                        context_str = "\n".join([doc.page_content for doc in context_docs])
                        context_msg = f"Contexto relevante:\n{context_str}"
                        # Insertamos el contexto al principio, después del mensaje de sistema si existe
                        messages.insert(-1, HumanMessage(content=context_msg))

                trimmed_messages = self.trimmer.invoke(messages)
                prompt = self.prompt_template.invoke({"messages": trimmed_messages})
                response = self.llm.invoke(prompt)
                
                return {"messages": [response]} # Solo devolvemos la nueva respuesta

            workflow.add_node("model", call_model)
            workflow.set_entry_point("model")
            workflow.set_finish_point("model")

            self.memory = MemorySaver()
            self.app = workflow.compile(checkpointer=self.memory)
        except Exception as e:
            logger.error(f"Error inesperado configurando LangChain: {str(e)}", exc_info=True)
            raise

    def send_message(self, message: str, historial_id: str = "default"):
        """
        Envía un mensaje al chatbot y retorna la respuesta.
        Ahora usa un diccionario 'configurable' para la memoria.
        """
        try:
            # =============================================================================
            # CAMBIO: LangGraph ahora requiere un diccionario 'configurable' para gestionar
            # la memoria por conversación. Usamos tu sugerencia 'historial_id'.
            # Esto soluciona el error "Checkpointer requires...".
            # =============================================================================
            config = {"configurable": {"thread_id": historial_id}}
            state = {"messages": [HumanMessage(content=message)]}
            
            response_stream = self.app.stream(state, config=config)
            final_response = None
            for chunk in response_stream:
                if "model" in chunk:
                    final_response = chunk["model"]["messages"][-1]

            return final_response.content if final_response else "No se pudo obtener una respuesta."

        except (ValueError, RuntimeError) as e:
            logger.error(f"Error de valor o ejecución procesando mensaje: {str(e)}", exc_info=True)
            return MESSAGES["ERROR"].format(error=str(e))
        except Exception as e:
            logger.error(f"Error inesperado procesando mensaje: {str(e)}", exc_info=True)
            return MESSAGES["ERROR"].format(error=str(e))