from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
from prompts.extract_prompt import extract_prompt

class ExtractionOutput(BaseModel):
    skills: List[str] = Field(description="A list of skills extracted from the resume")
    experience: str = Field(description="The extracted years of experience")
    tools: List[str] = Field(description="A list of tools extracted from the resume")

extraction_parser = JsonOutputParser(pydantic_object=ExtractionOutput)

def get_extraction_chain(llm):
    # For FewShotPromptTemplate, appending to the suffix is effective.
    prompt_with_instructions = extract_prompt
    prompt_with_instructions.suffix += "\n{format_instructions}"
    prompt_with_instructions = prompt_with_instructions.partial(
        format_instructions=extraction_parser.get_format_instructions()
    )
    return prompt_with_instructions | llm | extraction_parser
