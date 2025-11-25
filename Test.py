"""
RAG System Test Script
Tests the complete Microwave RAG implementation with sample queries
"""
import os
import sys
from pathlib import Path

# Add task directory to path
sys.path.insert(0, str(Path(__file__).parent / "task"))

from task.app import MicrowaveRAG
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY


def print_separator():
    print("\n" + "=" * 100 + "\n")


def run_test(rag: MicrowaveRAG, query: str, test_num: int):
    print(f"\n{'#' * 100}")
    print(f"# TEST {test_num}")
    print(f"{'#' * 100}\n")
    print(f"❓ Question: {query}")
    
    # Step 1: Retrieval
    context = rag.retrieve_context(query)
    
    # Step 2: Augmentation
    augmented_prompt = rag.augment_prompt(query, context)
    
    # Step 3: Generation
    answer = rag.generate_answer(augmented_prompt)
    
    print_separator()
    input("Press Enter to continue to next test...")


def main():
    print("🎯 Microwave RAG System - Automated Testing")
    print("=" * 100)
    print("This script will test the RAG system with various queries")
    print("=" * 100)
    
    # Initialize RAG system
    print("\n🔧 Initializing RAG System...")
    rag = MicrowaveRAG(
        embeddings=AzureOpenAIEmbeddings(
            deployment="text-embedding-3-small-1",
            azure_endpoint=DIAL_URL,
            api_key=SecretStr(API_KEY)
        ),
        llm_client=AzureChatOpenAI(
            temperature=0.0,
            azure_deployment="gpt-4o",
            azure_endpoint=DIAL_URL,
            api_key=SecretStr(API_KEY),
            api_version=""
        )
    )
    
    print("✅ RAG System initialized successfully!")
    input("\nPress Enter to start tests...")
    
    # Test 1: Valid query - Safety precautions
    run_test(
        rag,
        "What safety precautions should be taken to avoid exposure to excessive microwave energy?",
        1
    )
    
    # Test 2: Valid query - Cooking time
    run_test(
        rag,
        "What is the maximum cooking time that can be set on the DW 395 HCG microwave oven?",
        2
    )
    
    # Test 3: Valid query - Cleaning
    run_test(
        rag,
        "How should you clean the glass tray of the microwave oven?",
        3
    )
    
    # Test 4: Valid query - Safe materials
    run_test(
        rag,
        "What materials are safe to use in this microwave during both microwave and grill cooking modes?",
        4
    )
    
    # Test 5: Invalid query - Out of scope
    run_test(
        rag,
        "What do you know about the DIALX community?",
        5
    )
    
    # Test 6: Invalid query - Out of scope
    run_test(
        rag,
        "What do you think about the dinosaur era? Why did they die?",
        6
    )
    
    print("\n" + "=" * 100)
    print("🎉 All tests completed successfully!")
    print("=" * 100)


if __name__ == "__main__":
    # Change to task directory for proper file paths
    os.chdir(Path(__file__).parent / "task")
    main()

