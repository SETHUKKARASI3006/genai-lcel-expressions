## Design and Implementation of LangChain Expression Language (LCEL) Expressions

### AIM:
To design and implement LangChain Expression Language (LCEL) expressions using multiple prompt parameters and core components—prompt, model, and output parser—and evaluate their functionality through examples such as joke generation, retrieval-augmented QA, and function calling.

### PROBLEM STATEMENT:
Build LCEL-based chains that demonstrate modularity, composability, and real-world utility. The system should support dynamic prompts, integrate retrieval mechanisms, and optionally invoke external tools via function calling.

### DESIGN STEPS:

#### STEP 1:
Create a prompt template with parameters

#### STEP 2:
Use ChatOpenAI() and StrOutputParser() to form a chain:
chain = prompt | model | output_parser

#### STEP 3:
Use DocArrayInMemorySearch for embedding-based document retrieval

#### STEP 4:
Use RunnableMap to dynamically inject context and questions

#### STEP 5:
Use fallback chains and streaming for robustness and interactivity

### PROGRAM:

```
import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

#!pip install pydantic==1.10.8

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    "tell me a short joke about {topic}"
)
model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | model | output_parser

chain.invoke({"topic": "rabbits"})

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch

vectorstore = DocArrayInMemorySearch.from_texts(
    ["harrison worked at kensho", "rabbits like to eat carrots"],
    embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

retriever.get_relevant_documents("where did harrison work?")

retriever.get_relevant_documents("what do rabbits like to eat")

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

from langchain.schema.runnable import RunnableMap


chain = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | model | output_parser

chain.invoke({"question": "where did harrison work?"})

inputs = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
})

inputs.invoke({"question": "where did harrison work?"})

functions = [
    {
      "name": "weather_search",
      "description": "Search for weather given an airport code",
      "parameters": {
        "type": "object",
        "properties": {
          "airport_code": {
            "type": "string",
            "description": "The airport code to get the weather for"
          },
        },
        "required": ["airport_code"]
      }
    }
  ]

prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}")
    ]
)
model = ChatOpenAI(temperature=0).bind(functions=functions)

runnable = prompt | model

runnable.invoke({"input": "what is the weather in sf"})

functions = [
    {
      "name": "weather_search",
      "description": "Search for weather given an airport code",
      "parameters": {
        "type": "object",
        "properties": {
          "airport_code": {
            "type": "string",
            "description": "The airport code to get the weather for"
          },
        },
        "required": ["airport_code"]
      }
    },
        {
      "name": "sports_search",
      "description": "Search for news of recent sport events",
      "parameters": {
        "type": "object",
        "properties": {
          "team_name": {
            "type": "string",
            "description": "The sports team to search for"
          },
        },
        "required": ["team_name"]
      }
    }
  ]

model = model.bind(functions=functions)

runnable = prompt | model

runnable.invoke({"input": "how did the patriots do yesterday?"})

from langchain.llms import OpenAI
import json

simple_model = OpenAI(
    temperature=0, 
    max_tokens=1000, 
    model="gpt-3.5-turbo-instruct"
)
simple_chain = simple_model | json.loads

challenge = "write three poems in a json blob, where each poem is a json blob of a title, author, and first line"

simple_chain.invoke(challenge)

simple_chain.invoke(challenge)

model = ChatOpenAI(temperature=0)
chain = model | StrOutputParser() | json.loads

chain.invoke(challenge)

final_chain = simple_chain.with_fallbacks([chain])

final_chain.invoke(challenge)

prompt = ChatPromptTemplate.from_template(
    "Tell me a short joke about {topic}"
)
model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | model | output_parser

chain.invoke({"topic": "rabbits"})

chain.batch([{"topic": "rabbits"}, {"topic": "frogs"}])

for t in chain.stream({"topic": "rabbits"}):
    print(t)

response = await chain.ainvoke({"topic": "rabbits"})
response
```

### OUTPUT:
<img width="960" height="116" alt="image" src="https://github.com/user-attachments/assets/4de3a0bb-0335-4a88-b6d5-cbede8e90131" />

<img width="633" height="117" alt="image" src="https://github.com/user-attachments/assets/d57f9c9d-1e72-4b60-bc06-d5fa3018d033" />

<img width="1368" height="109" alt="image" src="https://github.com/user-attachments/assets/dbfefd3f-3f22-452b-8adc-3bde4b66bfc7" />

<img width="738" height="316" alt="image" src="https://github.com/user-attachments/assets/2f23e338-2315-42e2-9317-bd3cfcfc1678" />

<img width="991" height="149" alt="image" src="https://github.com/user-attachments/assets/6c63765a-5b14-4f34-8a49-62a81bee770c" />

### RESULT:
The implementation showcases the flexibility and power of LCEL for building intelligent, context-aware applications. 
