import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from chains.extraction_chain import get_extraction_chain
from chains.matching_chain import get_matching_chain
from chains.scoring_chain import get_scoring_chain
from chains.explanation_chain import get_explanation_chain

# Load environment variables
load_dotenv()

# Initialize the pipeline
def run_pipeline():
    print("Initializing LLM...")
    # Using Qwen/Qwen2.5-7B-Instruct as a reliable free HuggingFace model
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        max_new_tokens=512,
        do_sample=False,
        temperature=0.01,
        return_full_text=False
    )
    chat_llm = ChatHuggingFace(llm=llm)
    
    # Initialize the 4 individual chains
    extraction_chain = get_extraction_chain(chat_llm)
    matching_chain = get_matching_chain(chat_llm)
    scoring_chain = get_scoring_chain(chat_llm)
    explanation_chain = get_explanation_chain(chat_llm)

    # Load Job Description
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "data", "job_description.txt"), "r") as f:
        job_description = f.read()

    # Define candidates
    candidates = [
        {"file": "resume_strong.txt", "tag": "strong"},
        {"file": "resume_average.txt", "tag": "average"},
        {"file": "resume_weak.txt", "tag": "weak"}
    ]

    for candidate in candidates:
        print(f"\n--- Processing Candidate: {candidate['tag'].upper()} ---")
        
        # Load Resume
        with open(os.path.join(base_dir, "data", candidate['file']), "r") as f:
            resume_text = f.read()

        # Config tags for LangSmith
        config = {"tags": [candidate['tag']]}
        
        try:
            # STEP 1: EXTRACTION
            extracted_data = extraction_chain.invoke({"resume": resume_text}, config=config)
            
            # STEP 2: MATCHING
            # Passing output from Step 1 as input to Step 2
            matched_data = matching_chain.invoke({
                "extracted_data": str(extracted_data),
                "job_description": job_description
            }, config=config)
            
            # STEP 3: SCORING
            # Passing output from Step 2
            score_data = scoring_chain.invoke({
                "extracted_data": str(extracted_data),
                "matched_data": str(matched_data)
            }, config=config)
            
            # STEP 4: EXPLANATION
            # Passing output from Step 3 and 2
            explanation_data = explanation_chain.invoke({
                "matched_data": str(matched_data),
                "score_data": str(score_data)
            }, config=config)

            # FINAL OUTPUT Formatting
            final_output = {
                "score": score_data.get("score"),
                "explanation": explanation_data.get("explanation")
            }
            
            import json
            print(json.dumps(final_output, indent=2))
        
        except Exception as e:
            print(f"Pipeline Failed for {candidate['tag']}. Error: {e}")

if __name__ == "__main__":
    run_pipeline()
