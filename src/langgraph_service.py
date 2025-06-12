# src/langgraph_service.py

from langchain_core.messages import HumanMessage
# =============================================================================
# Langchain model imports will be conditional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import trim_messages # Reverted import path
from .constants import MESSAGES
from src.config import GlobalConfig # Use GlobalConfig directly
from .rag.logging_config import logger

class LangGraphService:
    """
    Servicio que encapsula la configuración y ejecución de LangGraph.
    """
    def __init__(self, global_config: GlobalConfig, chat_history, retriever): # Takes GlobalConfig
        self.config = global_config # Store GlobalConfig
        self.chat_history = chat_history
        self.retriever = retriever
        self._setup_langgraph()

    def _setup_langgraph(self):
        try:
            # Dynamically load the appropriate LLM client based on the provider specified in the config.
            # This allows for flexible switching between different LLM backends.
            provider = self.config.LLM_PROVIDER.lower()
            model_name = self.config.LLM_MODEL_NAME

            if provider == "anthropic":
                from langchain_anthropic import ChatAnthropic # Use new package
                self.llm = ChatAnthropic(
                    model=model_name, # langchain_anthropic uses 'model'
                    anthropic_api_key=self.config.ANTHROPIC_API_KEY,
                    max_tokens=512 # Configurable max tokens for the LLM response. TODO: Consider making this configurable via GlobalConfig
                )
            elif provider == "openai":
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(
                    model_name=model_name,
                    openai_api_key=self.config.OPENAI_API_KEY,
                    max_tokens=512 # Configurable max tokens for the LLM response. TODO: Consider making this configurable via GlobalConfig
                )
            elif provider == "gemini":
                from langchain_google_genai import ChatGoogleGenerativeAI
                self.llm = ChatGoogleGenerativeAI(
                    model=model_name, # ChatGoogleGenerativeAI uses 'model' not 'model_name'
                    google_api_key=self.config.GEMINI_API_KEY,
                    # max_output_tokens=512 # Parameter name might differ for Gemini
                )
            elif provider == "deepseek":
                from langchain_openai import ChatOpenAI # DeepSeek uses OpenAI compatible API
                self.llm = ChatOpenAI(
                    model=model_name, # Corrected: model_name for ChatOpenAI
                    openai_api_key=self.config.DEEPSEEK_API_KEY,
                    base_url="https://api.deepseek.com/v1",
                    max_tokens=512
                )
            else:
                raise ValueError(f"Unsupported LLM_PROVIDER: {self.config.LLM_PROVIDER}")

            self.prompt_template = ChatPromptTemplate.from_messages([
                ("system", """Eres un asistente amigable y servicial. Responde de manera concisa y clara. Si hay contexto relevante, úsalo para dar respuestas más precisas, de lo contrario, responde con tu conocimiento general.""")
            ])
            # Ensure token_counter is correctly passed if needed by trim_messages
            # For Langchain >0.2.14, trim_messages can take BaseLanguageModel directly for token counting
            # Pass the initialized LLM instance to MessageTrimmer for accurate token counting.
            self.trimmer = trim_messages(
                max_tokens=self.config.MAX_CHUNK_SIZE, # Using MAX_CHUNK_SIZE from config. Ensure the trimmer uses the same token counting mechanism as the LLM.
                strategy="last",
                token_counter=self.llm, # Pass the LLM instance
                include_system=True,
                allow_partial=False,
                # start_on_system=False # Optional: depending on desired trimming behavior
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