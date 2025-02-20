from typing import cast
from langgraph.func import entrypoint,task
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv,find_dotenv
from pydantic import BaseModel,Field
import os

load_dotenv(find_dotenv())

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"))


class InstructionsGenerator(BaseModel):
    worker_instructions:list[str] = Field(description="List of instructions for each worker keep workers count under 3")





@task
def generate_instructions(query:str):
    instructions = cast(InstructionsGenerator,llm.with_structured_output(InstructionsGenerator).invoke(
        f"""
        Generate instructions for the worker to generate an idea validation report based on the following query:
        {query}
        """
    ))
    return instructions


@task
def call_worker(instruction:str)->str:
    response = llm.invoke(instruction).content
    # print("Worker Response: ""\n\n",response)
    return response


@task
def combine_results(results:list[str])->str:
    return "\n\n".join(results)


@entrypoint()
def call_orchestrator(query:str):
    # step 1: Generate instructions for the worker
    instructions = generate_instructions(query).result()

    # step 2: Call the worker
    workers = [call_worker(instruction) for instruction in instructions.worker_instructions]

    # step 3: Resolve the futures in parallel
    results = [result.result() for result in workers]

    # step 4: Combine all results into a single response
    final_result = combine_results(results).result()
    return final_result





def main ():
    final_report = call_orchestrator.invoke("I want to build a Lead Generation Agent")
    print("Final Report: ""\n\n",final_report)
    
    with open("report.md","w") as f:
        f.write(final_report)
        print("Report saved as report.md")











