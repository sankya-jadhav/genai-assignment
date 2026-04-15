from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

# Define the examples for few-shot prompting
examples = [
    {
        "resume": "Software engineer with 5 years experience. Skills: Java, Python. Tools: Git, Docker.",
        "output": '{{"skills": ["Java", "Python"], "experience": "5 years", "tools": ["Git", "Docker"]}}'
    },
    {
        "resume": "Recent grad looking for a data role. Know SQL and Tableau.",
        "output": '{{"skills": ["SQL", "Tableau"], "experience": "0 years", "tools": []}}'
    }
]

# Create a prompt template for the examples
example_prompt = PromptTemplate(
    input_variables=["resume", "output"],
    template="Resume:\n{resume}\n\nOutput:\n{output}"
)

# Create the few-shot prompt template
extract_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="""You are an expert HR data extractor. Your task is to extract skills, experience (in years), and tools from the provided resume.
You must adhere to the following very strict rules:
- Do NOT assume any information not present in the input
- Return output in JSON format only
- Do not add extra text outside JSON
- Output keys must be exactly "skills", "experience", and "tools".
""",
    suffix="Resume:\n{resume}\n\nOutput:\n",
    input_variables=["resume"],
)
