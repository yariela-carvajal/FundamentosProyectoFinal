import os
from typing import List, Tuple

from dotenv import load_dotenv
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

IMPORTANT_NOTE = (
    "Antes de comenzar cualquier suplementación, consulta con un profesional de la salud, "
    "especialmente si tienes problemas renales o estás tomando medicación para la presión arterial o antibióticos."
)

DOCUMENTS = [
    """Citrato de Magnesio:El más equilibrado\nEs uno de los tipos más populares debido a su alta biodisponibilidad y precio accesible. 
    Se absorbe con relativa facilidad en el tracto digestivo. Es ideal para mejorar la digestión y aliviar el estreñimiento ocasional.
    También ayuda a la relajación muscular y a reponer niveles bajos de magnesio. Su biodisponibilidad es alta, con un efecto laxante moderado. 
    Su uso principal es para el estreñimiento/general. La forma de consumo suele venir en polvo o cápsulas. Se recomienda tomarlo por la noche si se busca un efecto laxante suave por la mañana.""",

    """Glicinato/Bisglicinato de Magnesio: El relajante\nEn esta forma, el magnesio está unido a la glicina, un aminoácido que favorece el sueño. 
    Es la opción que menos efectos laxantes produce. Reduce la ansiedad, mejora de la calidad del sueño, alivia el estrés y apoya en casos de fibromialgia o dolores crónicos. 
    Su biodisponibilidad es muy Alta con un efecto laxante muy bajo. Su uso principal es para regular el sueño y disminuir la ansiedad. 
    Generalmente se consume en cápsulas y debe tomarlo entre 30 y 60 minutos antes de dormir es lo ideal.""",

    """Malato de Magnesio: Para la energía\nUnido al ácido málico, que es clave en el ciclo de producción de energía en las células. 
    Se usa para combatir la fatiga crónica, mejorar el rendimiento físico y aliviar dolores musculares. Es excelente para personas que se sienten \"agotadas\" durante el día. 
    Su biodisponibilidad es alta con un efectoLaxante bajo. Su uso principal es para ayudar a producir energía y reducir el dolor muscular. 
    Se consume preferiblemente por la mañana con el desayuno para aprovechar el impulso de energía.""",

    """Treonato de Magnesio: El 'alimento' para el cerebro.\nEs la única forma que atraviesa eficazmente la barrera hematoencefálica, llegando directamente al sistema nervioso central. 
    Se usa para mejorar la memoria, el enfoque, la función cognitiva y prevenir el deterioro mental relacionado con la edad. 
    Su biodisponibilidad es media/alta con un efecto laxante casi nulo. Su uso principal es para ayudar al cerebro/memoria. 
    Se puede tomar en cualquier momento, pero suele recomendarse una dosis dividida (mañana y tarde).""",

    """Cloruro de Magnesio: El clásico 'multiusos' \nEs una sal inorgánica que suele encontrarse en aceites para la piel o en suplementos económicos. 
    Se utiliza para la desintoxicación de tejidos, apoyo al metabolismo y salud renal. Por vía tópica (aceite), es excelente para calambres musculares localizados. 
    Su biodisponibilidad es media, con un efecto laxante alto (si se ingiere). Su uso principal es la desintoxicación / uso tópico., 
    Su forma de consumo es si es en polvo (para preparar en agua), tiene un sabor muy amargo. Se absorbe bien, pero puede causar diarrea si la dosis es alta. 
    Vía oral: Es muy eficaz para \"limpiar\" el organismo y estimular la función renal, pero su sabor metálico y amargo es un reto para muchos. 
    Vía tópica: Es el rey de los \"aceites de magnesio\". Si te dan calambres después de hacer ejercicio, aplicar cloruro de magnesio diluido directamente sobre el músculo es una de las formas más rápidas de obtener alivio sin pasar por el sistema digestivo.""",

    """Óxido de Magnesio: Su biodisponibilidad es muy baja, con un efecto laxante muy alto. 
    Su uso principal es la acidez estomacal. Evita el Óxido de Magnesio, es el más común en farmacias baratas, pero su absorción es bajísima (cerca del 4%). 
    Básicamente, actúa como un laxante fuerte porque el cuerpo no lo puede procesar bien.""",

    """En el consumo de magnesio la comida importa: Por lo general, tomarlo con alimentos ayuda a reducir la posibilidad de malestar estomacal.""",

    """Dosis: La dosis diaria recomendada de magnesio suele oscilar entre los 300mg y 450 mg para adultos, pero esto varía según la dieta y necesidades individuales.""",

    """Aunque el magnesio es generalmente muy seguro y el cuerpo suele eliminar el exceso a través de la orina, consumirlo de forma inadecuada o en dosis excesivas puede traer complicaciones.""",

    """Efectos Gastrointestinales del Magnesio (Los más frecuentes)\nEs el efecto secundario número uno, especialmente con formas de baja absorción como el óxido o el cloruro. 
    Diarrea: El magnesio atrae agua hacia los intestinos (efecto osmótico), lo que acelera el tránsito intestinal. 
    Náuseas y calambres abdominales: Pueden ocurrir si se toma el suplemento con el estómago vacío. 
    Gases y distensión: Comunes cuando el cuerpo no está acostumbrado a la dosis.""",

    """Toxicidad por Magnesio (Hipermagnesemia)\nEs poco común en personas sanas, pero puede ocurrir si se ingieren dosis masivas (generalmente superiores a 2,500 - 5,000 mg al día) o si los riñones no funcionan correctamente. 
    Los síntomas incluyen: Letargo y debilidad muscular, Hipotensión, Dificultad para respirar y Arritmias.""",

    """Interacciones con Medicamentos\nEl magnesio puede \"bloquear\" o disminuir la eficacia de ciertos fármacos: Antibióticos, Medicamentos para la osteoporosis (Bifosfonatos), Medicamentos para la presión arterial y Diuréticos. 
    "Debe tomarse con cuidado y respetando un espacio de tiempo adecuado con otros medicamentos.""",

    """Efectos Neurológicos\nSomnolencia excesiva: Dado que formas como el glicinato son muy relajantes, algunas personas pueden sentirse \"atontadas\" o demasiado sedadas al día siguiente si la dosis es alta. 
    Confusión mental: Solo ocurre en niveles de toxicidad elevados.""",

    """¿Cómo minimizar estos riesgos?\nEscala la dosis: Empieza con una dosis baja y auméntala gradualmente para que tu intestino se adapte. 
    Divide las tomas: En lugar de tomar 400 mg de golpe, toma 200 mg en la mañana y 200 mg en la noche. 
    Elige la forma correcta: Si tienes el estómago sensible, evita el cloruro y el óxido; opta por el bisglicinato. 
    Ojo con los riñones: Si tienes insuficiencia renal, no debes tomar magnesio sin supervisión médica estricta, ya que tus riñones podrían no ser capaces de filtrar el exceso."""
]

DEMO_QUESTIONS = [
    "¿Cuál es el mejor magnesio para mejorar el sueño?",
    "¿Cuál es la dosis de magnesio recomendada?",
    "¿Cómo puedo mejorar la memoria?",
    "¿Cómo puedo mejorar la energía?",
    "¿Si consumo un antibiótico puedo consumir magnesio?"
]


def validate_question(question: str) -> str:
    """Valida y limpia la pregunta del usuario.

    Verifica que la entrada sea una cadena válida y no esté vacía.

    Args:
        question: Texto raw del usuario.

    Returns:
        Pregunta limpia sin espacios en blanco al inicio/final.

    Raises:
        ValueError: Si la pregunta no es texto válido o está vacía.
    """
    if not isinstance(question, str):
        raise ValueError("La pregunta debe ser un texto válido.")

    question_text = question.strip()
    if not question_text:
        raise ValueError("Por favor ingresa una pregunta sobre el magnesio.")

    return question_text


def build_document_store(documents: List[str]) -> FAISS:
    """Construye un almacén vectorial FAISS a partir de documentos.

    Procesa la colección de documentos usando CharacterTextSplitter,
    genera embeddings con HuggingFace y crea un índice FAISS.

    Args:
        documents: Lista de documentos de texto sin procesar.

    Returns:
        Almacén vectorial FAISS listo para búsqueda y recuperación.
    """
    cleaned_docs = [doc.strip() for doc in documents if isinstance(doc, str) and doc.strip()]
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = []
    for index, document in enumerate(cleaned_docs, start=1):
        for chunk in text_splitter.split_text(document):
            chunks.append(Document(page_content=chunk, metadata={"source": f"Documento {document}"}))

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    return FAISS.from_documents(chunks, embeddings)


def build_qa_chain() -> object:
    """Construye la cadena de recuperación con OpenAI y el almacén vectorial.

    Inicializa el modelo GPT-4o-mini, configura el retriever con k=3,
    crea el prompt template y retorna una cadena RetrievalChain.

    Returns:
        Objeto RetrievalChain con método invoke(query_dict) para QA.

    Raises:
        RuntimeError: Si OPENAI_API_KEY no está definida o la inicialización falla.
    """
    vector_store = build_document_store(DOCUMENTS)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "La variable de entorno OPENAI_API_KEY no está definida. "
            "Establece la clave de OpenAI antes de ejecutar la aplicación."
        )

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=api_key)
    except Exception as exc:
        raise RuntimeError(
            "No se pudo inicializar el modelo de OpenAI. Verifica que OPENAI_API_KEY sea válida."
        ) from exc

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Create the prompt template
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the provided context. If you don't find the answer in the context, say so.

Context: {context}

Question: {question}

Answer:"""
    )

    # Create a simple chain using LCEL (LangChain Expression Language)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Wrapper to maintain compatibility
    class RetrievalChain:
        def __init__(self, retriever, chain):
            self.retriever = retriever
            self.chain = chain

        def invoke(self, query_dict):
            question = query_dict.get("input", "")
            docs = self.retriever.invoke(question)
            context = format_docs(docs)
            
            result = self.chain.invoke({
                "context": context,
                "question": question
            })
            
            return {
                "output_text": result,
                "source_documents": docs
            }

    return RetrievalChain(retriever, chain)


def format_sources(source_documents: List[Document]) -> str:
    """Formatea documentos fuente para mostrar en el output del chat.

    Extrae nombres únicos de fuentes y los presenta en formato de lista.
    Si no hay fuentes, retorna mensaje por defecto.

    Args:
        source_documents: Lista de documentos con metadata de origen.

    Returns:
        Cadena formateada con las fuentes encontradas o mensaje por defecto.
    """
    if not source_documents:
        return "Fuentes: Ninguna fuente específica encontrada."

    unique_sources = []
    for source_doc in source_documents:
        source_name = source_doc.metadata.get("source") if source_doc.metadata else None
        if source_name and source_name not in unique_sources:
            unique_sources.append(source_name)

    if not unique_sources:
        return "Fuentes: Documentos internos."

    formatted = "\n".join(f"- {source}" for source in unique_sources)
    return f"Fuentes:\n{formatted}"


QA_CHAIN = build_qa_chain()


def generate_response(question: str) -> str:
    """Genera una respuesta usando la cadena de recuperación basada en RAG.

    Valida la pregunta, la envía al QA_CHAIN, y formatea la respuesta
    con referencias a las fuentes utilizadas.

    Args:
        question: Pregunta del usuario sobre magnesio.

    Returns:
        Respuesta formateada con fuentes encontradas.

    Raises:
        ValueError: Si la pregunta es inválida.
        RuntimeError: Si la llamada al modelo falla o hay problema con API de OpenAI.
    """
    validated_question = validate_question(question)
    try:
        response = QA_CHAIN.invoke({"input": validated_question})
        if isinstance(response, dict):
            answer = (
                response.get("output_text")
                or response.get("result")
                or response.get("answer")
                or response.get("text")
                or "No se obtuvo una respuesta clara."
            )
            sources = response.get("source_documents", []) or []
        else:
            answer = str(response)
            sources = []

        return f"{answer.strip()}\n\n{format_sources(sources)}"
    except ValueError:
        raise
    except Exception as exc:
        if "api" in str(exc).lower() or "key" in str(exc).lower():
            error_message = "Error de API de OpenAI. Verifica tu clave API y créditos."
        else:
            error_message = f"Error al generar la respuesta: {str(exc)}"
        print(f"DEBUG: {error_message}")
        print(f"DEBUG: Exception details: {exc}")
        raise RuntimeError(error_message) from exc


def handle_demo_question(question: str) -> Tuple[str, str]:
    """Maneja la selección de una pregunta de demostración.

    Valida la pregunta y genera la respuesta, retornando la pregunta
    validada y la respuesta con fuentes.

    Args:
        question: Pregunta de demostración preseleccionada.

    Returns:
        Tupla (pregunta_validada, respuesta_con_fuentes).

    Raises:
        ValueError: Si la pregunta es inválida.
        RuntimeError: Si la generación de respuesta falla.
    """
    validated_question = validate_question(question)
    answer = generate_response(validated_question)
    return validated_question, answer


def clear_fields() -> Tuple[str, str]:
    """Limpia los campos de pregunta y respuesta.

    Retorna tupla de cadenas vacías para limpiar la interfaz.

    Returns:
        Tupla de dos cadenas vacías ("", "").
    """
    return "", ""


def build_interface() -> gr.Blocks:
    """Construye la interfaz Gradio para la aplicación RAG de Magnesio.

    Crea una interfaz interactiva con campos de entrada/salida,
    botones de envío/limpieza y preguntas de demostración.

    Returns:
        Objeto gr.Blocks configurado y listo para lanzar (launch).
    """
    with gr.Blocks(title="Tipos de magnesio y sus propiedades") as demo:
        gr.Markdown("## Tipos de magnesio y sus propiedades")
        gr.Markdown(f"**Nota importante:** {IMPORTANT_NOTE}")

        with gr.Row():
            question_input = gr.Textbox(
                label="Pregunta",
                placeholder="Escribe tu pregunta sobre magnesio aquí... \nSi deseas enviar la pregunta, sin dar click en el botón, presiona 'Shift+Enter' después de escribir tu pregunta.",
                lines=3,
                interactive=True,
            )
            response_output = gr.Textbox(
                label="Respuesta",
                placeholder="La respuesta aparecerá aquí...",
                lines=12,
                interactive=False,
            )

        with gr.Row():
            submit_button = gr.Button(value="Enviar pregunta")
            clear_button = gr.Button(value="Limpiar")

        with gr.Accordion("Preguntas de demostración", open=True):
            demo_buttons = []
            for demo_question in DEMO_QUESTIONS:
                button = gr.Button(value=demo_question)
                demo_buttons.append(button)

        submit_button.click(
            fn=generate_response,
            inputs=[question_input],
            outputs=[response_output],
        )

        question_input.submit(
            fn=generate_response,
            inputs=[question_input],
            outputs=[response_output],
        )

        clear_button.click(
            fn=clear_fields,
            inputs=[],
            outputs=[question_input, response_output],
        )

        for button, demo_question in zip(demo_buttons, DEMO_QUESTIONS):
            button.click(
                fn=lambda q=demo_question: handle_demo_question(q),
                inputs=[],
                outputs=[question_input, response_output],
            )

    return demo


if __name__ == "__main__":
    interface = build_interface()
    interface.launch(server_name="0.0.0.0", share=False)
