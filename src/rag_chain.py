from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from .config import config
from .vector_store import VectorStoreManager

class RAGChain:
    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store
        self.retriever = self.vector_store.get_retriever()
        
        # Load local HuggingFace model
        tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(config.LLM_MODEL)
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=100
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        self.prompt = PromptTemplate.from_template(
            """Answer the question based only on the following context:
{context}

Question: {question}

Answer:"""
        )
        
        self.chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def answer(self, question: str) -> dict:
        docs = self.vector_store.similarity_search(question)
        if not docs:
            return {
                "answer": "I don't have enough context to answer this question.",
                "sources": [],
                "num_docs_used": 0
            }
        
        answer_text = self.chain.invoke(question)
        sources = [doc.page_content for doc in docs]
        
        return {
            "answer": answer_text,
            "sources": sources,
            "num_docs_used": len(sources)
        }
