import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.vectorstores import VectorStore
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY


SYSTEM_PROMPT = """You are a RAG-powered assistant that assists users with their questions about microwave usage.
            
## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
"""

USER_PROMPT = """##RAG CONTEXT:
{context}


##USER QUESTION: 
{query}"""


class MicrowaveRAG:

    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = self._setup_vectorstore()

    def _setup_vectorstore(self) -> VectorStore:
        """Initialize the RAG system"""
        print("🔄 Initializing Microwave Manual RAG System...")
        # Check if `microwave_faiss_index` folder exists
        current_dir = os.path.dirname(os.path.abspath(__file__))
        index_path = os.path.join(current_dir, "microwave_faiss_index")
        if os.path.exists(index_path):
            print("✅ Found existing FAISS index - loading...")
            vectorstore = FAISS.load_local(
                folder_path=index_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("✅ Vectorstore loaded successfully!")
        else:
            print("📝 No existing index found - creating new one...")
            vectorstore = self._create_new_index()
            print("✅ Vectorstore created and saved successfully!")
        return vectorstore

    def _create_new_index(self) -> VectorStore:
        print("📖 Loading text document...")
        # 1. Create Text loader
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "microwave_manual.txt")
        loader = TextLoader(
            file_path=file_path,
            encoding="utf-8"
        )
        # 2. Load documents with loader
        documents = loader.load()
        print(f"📄 Loaded {len(documents)} document(s)")
        
        # 3. Create RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", "."]
        )
        # 4. Split documents into chunks
        chunks = text_splitter.split_documents(documents)
        print(f"✂️  Split into {len(chunks)} chunks")
        
        # 5. Create vectorstore from documents
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        # 6. Save indexed data locally
        current_dir = os.path.dirname(os.path.abspath(__file__))
        index_path = os.path.join(current_dir, "microwave_faiss_index")
        vectorstore.save_local(index_path)
        print("💾 Index saved locally")
        
        # 7. Return created vectorstore
        return vectorstore

    def retrieve_context(self, query: str, k: int = 4, score=0.3) -> str:
        """
        Retrieve the context for a given query.
        Args:
              query (str): The query to retrieve the context for.
              k (int): The number of relevant documents(chunks) to retrieve.
              score (float): The similarity score between documents and query. Range 0.0 to 1.0.
        """
        print(f"{'=' * 100}\n🔍 STEP 1: RETRIEVAL\n{'-' * 100}")
        print(f"Query: '{query}'")
        print(f"Searching for top {k} most relevant chunks with similarity score {score}:")

        # Make similarity search with relevance scores
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query=query,
            k=k,
            score_threshold=score
        )

        context_parts = []
        # Iterate through results
        for i, (doc, score) in enumerate(results, 1):
            context_parts.append(doc.page_content)
            print(f"\n📄 Chunk {i} (Score: {score:.4f}):")
            print(f"{doc.page_content}")

        print("=" * 100)
        return "\n\n".join(context_parts) # will join all chunks ion one string with `\n\n` separator between chunks

    def augment_prompt(self, query: str, context: str) -> str:
        print(f"\n🔗 STEP 2: AUGMENTATION\n{'-' * 100}")

        # Format USER_PROMPT with context and query
        augmented_prompt = USER_PROMPT.format(context=context, query=query)

        print(f"{augmented_prompt}\n{'=' * 100}")
        return augmented_prompt

    def generate_answer(self, augmented_prompt: str) -> str:
        print(f"\n🤖 STEP 3: GENERATION\n{'-' * 100}")

        # 1. Create messages array
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=augmented_prompt)
        ]
        # 2. Invoke llm client with messages
        response = self.llm_client.invoke(messages)
        # 3. Print response content
        print(f"📝 Answer: {response.content}")
        # 4. Return response content
        return response.content


def main(rag: MicrowaveRAG):
    print("🎯 Microwave RAG Assistant")

    while True:
        user_question = input("\n> ").strip()
        # Step 1: make Retrieval of context
        context = rag.retrieve_context(user_question)
        # Step 2: Augmentation
        augmented_prompt = rag.augment_prompt(user_question, context)
        # Step 3: Generation
        answer = rag.generate_answer(augmented_prompt)



if __name__ == "__main__":
    main(
        MicrowaveRAG(
            # 1. pass embeddings
            embeddings=AzureOpenAIEmbeddings(
                deployment="text-embedding-3-small-1",
                azure_endpoint=DIAL_URL,
                api_key=SecretStr(API_KEY)
            ),
            # 2. pass llm_client
            llm_client=AzureChatOpenAI(
                temperature=0.0,
                azure_deployment="gpt-4o",
                azure_endpoint=DIAL_URL,
                api_key=SecretStr(API_KEY),
                api_version=""
            )
        )
    )