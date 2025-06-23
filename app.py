#importing necessary libraries
import os
import re
import time
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
import sqlite3
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker, scoped_session
import faiss
from sentence_transformers import SentenceTransformer
from sqlalchemy.sql import text
import json
import sqlparse
import logging

MAX_RETRIES = 3  # Maximum number of retries for generating a valid SQL query

# Load environment variables
load_dotenv()

# Initialize SQL database
sqldb_path = r"smarthome.db"
engine = create_engine(f"sqlite:///{sqldb_path}", connect_args={"check_same_thread": False})
Session = scoped_session(sessionmaker(bind=engine))

# Inspect database schema and get table information
inspector = inspect(engine)


# Initialize LLaMA model and Sql Coder with ChatOllama
llama_model_name = "llama3.1:latest"
sql_agent_llm1_name="pxlksr/defog_sqlcoder-7b-2:F16"
sql_agent_llm1=ChatOllama(model=sql_agent_llm1_name)
sql_agent_llm = ChatOllama(model=llama_model_name)

# Load vector database (FAISS index and metadata)
vector_db_path = r"vector_db.index"
metadata_path = r"metadata.txt"
ind = faiss.read_index(vector_db_path)


# Load metadata
with open(metadata_path, "r") as f:
    metadata = f.read().splitlines()

# Load the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
# Helper function to interact with the LLaMA model

# Database schema information   
schema_info = {
    "Bathroom_Brightness": ["timestamp", "brightness"],
    "Bathroom_Humidity": ["timestamp", "humidity"],
    "Bathroom_Temperature": ["timestamp", "temperature"],
    "Kitchen_Brightness": ["timestamp", "brightness"],
    "Kitchen_Humidity": ["timestamp", "humidity"],
    "Kitchen_Temperature": ["timestamp", "temperature"],
    "Room1_Brightness": ["timestamp", "brightness"],
    "Room1_Humidity": ["timestamp", "humidity"],
    "Room1_Temperature": ["timestamp", "temperature"],
    "Room2_Brightness": ["timestamp", "brightness"],
    "Room2_Humidity": ["timestamp", "humidity"],
    "Room2_Temperature": ["timestamp", "temperature"],
    "Room3_Brightness": ["timestamp", "brightness"],
    "Room3_Humidity": ["timestamp", "humidity"],
    "Room3_Temperature": ["timestamp", "temperature"],
    "Toilet_Brightness": ["timestamp", "brightness"],
    "Toilet_Humidity": ["timestamp", "humidity"],
    "Toilet_Temperature": ["timestamp", "temperature"],
    "ac": ["device_id", "temperature", "status", "energy_consumption", "minutes_used", "timestamp"],
    "device_information": ["device_id", "device_type", "device_location"],
    "fan": ["device_id", "speed", "status", "energy_consumption", "minutes_used", "timestamp"],
    "light": ["device_id","status", "energy_consumption", "minutes_used", "timestamp"],
    "oven": ["device_id", "mode", "status", "energy_consumption", "minutes_used", "timestamp"],
    "tv": ["device_id", "playback", "status", "energy_consumption", "minutes_used", "timestamp"],
    "washing_machine": ["device_id", "mode", "status", "energy_consumption", "water_consumption", "minutes_used", "timestamp"],
}

# Device-location information
device_location_info = [
    ('tv', 'Room1'), 
    ('fan', 'Room1'), 
    ('light', 'Room1'),
    ('washing machine', 'Room2'), 
    ('fan', 'Room2'),
    ('light', 'Room2'), 
    ('ac', 'Room3'), 
    ('fan', 'Room3'),
    ('light', 'Room3'), 
    ('oven', 'Kitchen'), 
    ('light', 'Kitchen'),
    ('light', 'Bathroom'), 
    ('light', 'Toilet')
]

# Database schema information
database_schema = {
    "Bathroom_Brightness": {
        "columns": {
            "timestamp": "TEXT",
            "brightness": "FLOAT"
        },
        "example_entry": [('2017-03-08 23:58:47', 0.0)]
    },
    "Bathroom_Humidity": {
        "columns": {
            "timestamp": "TEXT",
            "humidity": "BIGINT"
        },
        "example_entry": [('2017-03-08 23:58:47', 47)]
    },
    "Bathroom_Temperature": {
        "columns": {
            "timestamp": "TEXT",
            "temperature": "FLOAT"
        },
        "example_entry": [('2017-03-08 23:58:47', 19.21)]
    },
    "Kitchen_Brightness": {
        "columns": {
            "timestamp": "TEXT",
            "brightness": "FLOAT"
        },
        "example_entry": [('2017-03-09 06:31:25', 12.82)]
    },
    "Kitchen_Humidity": {
        "columns": {
            "timestamp": "TEXT",
            "humidity": "BIGINT"
        },
        "example_entry": [('2017-03-09 01:32:39', 47)]
    },
    "Kitchen_Temperature": {
        "columns": {
            "timestamp": "TEXT",
            "temperature": "FLOAT"
        },
        "example_entry": [('2017-03-09 01:12:35', 17.48)]
    },
    "Room1_Brightness": {
        "columns": {
            "timestamp": "TEXT",
            "brightness": "FLOAT"
        },
        "example_entry": [('09-03-2017 06:22', 1.83)]
    },
    "Room1_Humidity": {
        "columns": {
            "timestamp": "TEXT",
            "humidity": "BIGINT"
        },
        "example_entry": [('2017-03-09 01:11:35', 44)]
    },
    "Room1_Temperature": {
        "columns": {
            "timestamp": "TEXT",
            "temperature": "FLOAT"
        },
        "example_entry": [('2017-03-09 00:51:30', 19.53)]
    },
    "Room2_Brightness": {
        "columns": {
            "timestamp": "TEXT",
            "brightness": "FLOAT"
        },
        "example_entry": [('2017-03-09 06:34:26', 0.92)]
    },
    "Room2_Humidity": {
        "columns": {
            "timestamp": "TEXT",
            "humidity": "BIGINT"
        },
        "example_entry": [('2017-03-09 00:24:24', 44)]
    },
    "Room2_Temperature": {
        "columns": {
            "timestamp": "TEXT",
            "temperature": "FLOAT"
        },
        "example_entry": [('2017-03-09 00:04:19', 17.8)]
    },
    "Room3_Brightness": {
        "columns": {
            "timestamp": "TEXT",
            "brightness": "FLOAT"
        },
        "example_entry": [('2017-03-08 23:56:17', 0.0)]
    },
    "Room3_Humidity": {
        "columns": {
            "timestamp": "TEXT",
            "humidity": "BIGINT"
        },
        "example_entry": [('2017-03-09 00:46:29', 44)]
    },
    "Room3_Temperature": {
        "columns": {
            "timestamp": "TEXT",
            "temperature": "FLOAT"
        },
        "example_entry": [('2017-03-09 00:16:22', 17.8)]
    },
    "Toilet_Brightness": {
        "columns": {
            "timestamp": "TEXT",
            "brightness": "FLOAT"
        },
        "example_entry": [('2017-03-09 06:27:53', 0.92)]
    },
    "Toilet_Humidity": {
        "columns": {
            "timestamp": "TEXT",
            "humidity": "BIGINT"
        },
        "example_entry": [('2017-03-09 01:10:04', 49)]
    },
    "Toilet_Temperature": {
        "columns": {
            "timestamp": "TEXT",
            "temperature": "FLOAT"
        },
        "example_entry": [('2017-03-09 00:20:23', 16.06)]
    },
    "fan": {
        "columns": {
            "device_id": "INTEGER",
            "speed": "INTEGER",
            "status": "TEXT",
            "energy_consumption": "REAL",
            "minutes_used": "INTEGER",
            "timestamp": "TEXT"
        },
        "example_entry": [(102, 3, 'on', 0.0, 0, '2017-03-03 00:00:00')]
    },
    "light": {
        "columns": {
            "device_id": "INTEGER",
            "status": "TEXT",
            "energy_consumption": "REAL",
            "minutes_used": "INTEGER",
            "timestamp": "TEXT"
        },
        "example_entry": [(103,'on', 0.0, 0, '2017-03-03 00:00:00')]
    },
    "ac": {
        "columns": {
            "device_id": "INTEGER",
            "temperature": "INTEGER",
            "status": "TEXT",
            "energy_consumption": "REAL",
            "minutes_used": "INTEGER",
            "timestamp": "TEXT"
        },
        "example_entry": [(301, 17, 'on', 0.0, 0, '2017-03-03 00:00:00')]
    },
    "oven": {
        "columns": {
            "device_id": "INTEGER",
            "mode": "TEXT",
            "status": "TEXT",
            "energy_consumption": "REAL",
            "minutes_used": "INTEGER",
            "timestamp": "TEXT"
        },
        "example_entry": [(401, 'Grill', 'on', 0.0, 0, '2017-03-03 00:00:00')]
    },
    "washing_machine": {
        "columns": {
            "device_id": "INTEGER",
            "mode": "TEXT",
            "status": "TEXT",
            "energy_consumption": "REAL",
            "water_consumption": "REAL",
            "minutes_used": "INTEGER",
            "timestamp": "TEXT"
        },
        "example_entry": [(201, 'Heavy Duty', 'on', 0.0, 0.0, 0, '2017-03-03 00:00:00')]
    },
    "tv": {
        "columns": {
            "device_id": "INTEGER",
            "playback": "TEXT",
            "status": "TEXT",
            "energy_consumption": "REAL",
            "minutes_used": "INTEGER",
            "timestamp": "TEXT"
        },
        "example_entry": [(101, 'Hulu', 'on', 0.0, 0, '2017-03-03 00:00:00')]
    },
    "device_information": {
        "columns": {
            "device_id": "INTEGER",
            "device_type": "TEXT",
            "device_location": "TEXT"
        },
        "example_entry": [(101, 'tv', 'Room1')]
    }
}

# Function to interact with LLM for generating sub-queries
def generate_response(prompt_text):
    try:
        response = sql_agent_llm.invoke(prompt_text)
        return response.content.strip()
    except Exception as e:
        return f"Error in generating response: {e}"
    
# Function to interact with LLM for generating SQL query
def generate_response1(prompt_text):
    try:
        response = sql_agent_llm1.invoke(prompt_text)
        return response.content.strip()
    except Exception as e:
        return f"Error in generating response: {e}"
    
# Identify relevant tables based on the user query
def identify_relevant_tables(user_question):
    """
    Identify relevant tables for the given user question.
    """
    prompt = f"""
    Based on the user's question, identify ALL relevant SQL tables from the database. Use the following information:

    Schema:
        {schema_info}

    Device-Location Information:
        {device_location_info}

    
    Rules:
    - use the Schema to know the tables and columns and what is stored in them
    - Match tables based on their names and column contents.
    - If brightness, temperature, or humidity are mentioned, include the respective table.
    - If unsure, include all potentially relevant tables.
    - IF A QUESTION IS ASKED ON ANY DEVICE , INCLUDE DEVICE_INFORMATION TABLE IN THE RELEVANT TABLES
    - if a question is asked on only devices , dont include the tables with the entries of brightness, temperature, humidity
    - ONLY RETURN THE TABLE NAMES . NO EXTRA TEXTS .
    - DONT GIVE THE REASONS OF WHY YOU INCLUDED A TABLE . JUST GIVE THE TABLE NAMES.
    User question: "{user_question}"

    ONLY return the names of the relevant tables as a comma-separated list. No additional information or explanations are needed.
    """
    response = generate_response(prompt)
    return [table.strip() for table in response.split(",")]

# Generate SQL query based on user question and relevant tables
def generate_sql_query(user_question, relevant_tables):
    """
    Generate a SQL query for the given user question and relevant tables using few-shot examples.
    """

    few_shot_examples = """
    Examples:

    - User question: "What is the energy consumption of the washing machine during the 2nd week of May 2024?"  
  SQL query:  
  "SELECT SUM(energy_consumed) AS total_energy  
  FROM washing_machine  
  WHERE device_id = (  
      SELECT device_id FROM device_information WHERE device_type = 'washing_machine'  
  )  
  AND DATE(timestamp) BETWEEN '2024-05-06' AND '2024-05-12'; "
    - User question: "How many hours was the fan used in Room2 during the 3rd week of June 2023?"  
  SQL query:  
  "SELECT SUM(minutes_used) / 60.0 AS total_hours  
  FROM fan  
  WHERE device_id = (  
      SELECT device_id FROM device_information WHERE device_location = 'Room2' AND device_type = 'fan'  
  )  
  AND DATE(timestamp) BETWEEN '2023-06-12' AND '2023-06-18';" 

    - User question: "what is the total energy consumption of the oven in February 2024 ?"
      SQL query: "SELECT SUM(energy_consumption) AS total_energy_feb
                    FROM oven
                    WHERE strftime('%Y-%m', timestamp) = '2024-02';
                    "
    -User question: "How many hours was the air conditioner in Room3 used during summer 2024?"
    SQL query:"SELECT SUM(minutes_used) / 60.0 AS total_hours
FROM ac
WHERE device_id = (
    SELECT device_id FROM device_information WHERE device_location = 'Room3' AND device_type = 'ac'
) 
AND strftime('%Y', timestamp) = '2024' 
AND strftime('%m', timestamp) IN ('03', '04', '05');
"
    - User question: "What is the average temperature of Room1 on 27th October 2024?"  
  SQL query:  
  "SELECT AVG(temperature) AS avg_temp  
  FROM Room1_Temperature  
  AND DATE(timestamp) = '2024-10-27';"  


    -User question:"What was the total energy consumption of the air conditioner in Room3 during summer 2024?"
    SQL query:"SELECT SUM(energy_consumption) AS total_energy
FROM ac
WHERE device_id = (
    SELECT device_id FROM device_information WHERE device_location = 'Room3' AND device_type = 'ac'
) 
AND strftime('%Y', timestamp) = '2024' 
AND strftime('%m', timestamp) IN ('03', '04', '05');
"


    -User question:"What was the average temperature set on the air conditioner in Room3 during summer 2024?"
    SQL query:"SELECT AVG(temperature) AS avg_temperature
FROM ac
WHERE device_id = (
    SELECT device_id FROM device_information WHERE device_location = 'Room3' AND device_type = 'ac'
) 
AND strftime('%Y', timestamp) = '2024' 
AND strftime('%m', timestamp) IN ('03', '04', '05');
"


    -User question:"On which day was the air conditioner used the most in Room3 during summer 2024?"
     SQL query:"SELECT DATE(timestamp) AS usage_day, SUM(minutes_used) AS total_minutes
FROM ac
WHERE device_id = (
    SELECT device_id FROM device_information WHERE device_location = 'Room3' AND device_type = 'ac'
) 
AND strftime('%Y', timestamp) = '2024' 
AND strftime('%m', timestamp) IN ('03', '04', '05')
GROUP BY usage_day
ORDER BY total_minutes DESC
LIMIT 1;
"
    """

    prompt = f"""
    role:
        you are a expert sqllite3 query generator who knows complete sqlLite3syntax.
        Generate a SQL query to answer the user's question based on the provided table names and column details.

    Schema:
        {schema_info}

    Device-Location Information:
        {device_location_info}

    Rules:
    - Use only the tables in {relevant_tables}
    - use the Schema to know the tables and columns and what is stored in them and generate the sql query accordingly
    - use the correct table and column names
    - Use correct SQLLite3 syntax that can be executed directly.
    - Only include the necessary tables and columns to answer the question.
    - Brightness, temperature, and humidity are independent of each other.
    - NOTE THAT THE TIMESTAMP IS OF THE FORMAT YYYY-MM-DD HH:MM:SS
    - For filtering by specific months or time periods, use `strftime('%m', timestamp)` for months , `strftime('%Y-%m-%d', timestamp)` for dates and strftime('%Y-%m-%d %H:%M:%S', timestamp) for whole timestamp.
    - If a time period like "last week" is mentioned, filter rows using the `timestamp` column.
    - For calculations, use SQL functions like `AVG`, `MAX`, `MIN`, or `COUNT` where appropriate.
    - Ensure no extra text or explanation is included in the output—ONLY the SQL query.
    - Give only a single query.
    - IF A QUESTION IS ASKED ABOUT A DEVICE(LIGHT/FAN/AC/WASHING_MACHINE/TV) CHECK THE DEVICE_INFORMATION TABLE TO LOOK OUT FOR THE LOCATION AND DEVICE TYPE
    - THE TABLE AND COLUMN NAMES ARE CASE SENSITIVE (USE IT CORRECTLY FROM THE DATABASE SCHEMA PROVIDED)
    - END THE QUERY WITH A SEMI-COLON ;

    {few_shot_examples}

    User question: "{user_question}"
    """
    response = generate_response1(prompt)
    return response

# Function to extract the SQL query from the LLaMA response
def validate_sql_with_llama(sql_query, sub_question):
    """
    Validates whether an LLM-generated SQL query is logically correct.

    Args:
        sql_query (str): The generated SQL query.
        sub_question (str): The corresponding sub-question.

    Returns:
        dict: Validation results including logical correctness and LLM feedback.
    """

    validation_results = {
        "syntax_valid": False,
        "execution_success": False,
        "logical_correctness": False,
        "llm_feedback": "",
        "error": None
    }

    # **1. Validate SQL Syntax**
    try:
        parsed_query = sqlparse.parse(sql_query)
        if not parsed_query:
            validation_results["error"] = "Invalid SQL syntax."
            return validation_results
        validation_results["syntax_valid"] = True
    except Exception as e:
        validation_results["error"] = f"SQL Parsing Error: {e}"
        return validation_results

    # **2. Execute the Query in a Safe Mode**
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql_query)).fetchall()
            validation_results["execution_success"] = True
    except Exception as e:
        validation_results["error"] = f"Query Execution Error: {e}"
        return validation_results

    # **3. Ask LLaMA to Validate Logical Correctness**
    llm_prompt = f"""
You are an SQL expert. Validate if the given SQL query correctly answers the user's question.

### **Database Schema:**
    Schema:
        {schema_info}

    Device-Location Information:
        {device_location_info}

### **User Question:**
{sub_question}

### **Generated SQL Query:**
{sql_query}

### **Validation Steps:**
1. Check if the query logic correctly answers the user's question.
2. Ensure the correct tables and columns are used.
3. Verify if aggregate functions (AVG, MAX, COUNT) match the question type.
4. Ensure filtering by time, device type, or location is correctly implemented.
5. If correct, return: **"VALID QUERY"**.
6. If incorrect, explain the error.

**Expected Output Format:**
- VALID QUERY (if correct)
- ERROR: Explanation of the mistake.
    """

    llm_response = sql_agent_llm.invoke(llm_prompt).content.strip()
    
    if "VALID QUERY" in llm_response:
        validation_results["logical_correctness"] = True
    else:
        validation_results["error"] = "Logical error detected."
        validation_results["llm_feedback"] = llm_response  # Store LLaMA's explanation

    return validation_results

# Function to run the SQL query with retries
def run_with_retries(question, retries=3):
    """
    Generate, validate, and execute an SQL query with retries if validation fails.

    Args:
        question (str): User question.
        retries (int): Number of retries allowed.

    Returns:
        str or list: Query result if valid, otherwise an error message.
    """
    attempt = 0
    success = False
    result_df = None

    while attempt < retries and not success:
        try:
            attempt += 1
            print(f"\nAttempt {attempt}: Generating SQL query for '{question}'")

            # Generate SQL query
            sql_query = correct_case_in_query(
                extract_sql_from_response(
                    generate_sql_query(question, identify_relevant_tables(question))
                )
            )

            print("\nGenerated SQL Query:")
            print(sql_query)

            # Validate SQL query using LLaMA
            validation_results = validate_sql_with_llama(sql_query, question)

            # Check all validation criteria
            if validation_results["error"] == None:
                print("\n SQL query passed all validations. Executing query...")
                result_df = execute_sql_query(sql_query)
                print("\nQuery Execution Result:", result_df)
                success = True  # Stop retrying if the query is correct
            else:
                print("\nRetrying with a new SQL query...\n")

        except Exception as e:
            print("\n Error during SQL query generation or execution:", e)
            time.sleep(1)  # Wait before retrying
    
    if success:
        return result_df
    else:
        return f"\nError: SQL query could not be validated after {retries} attempts."

# Function to extract the SQL query from the LLaMA response    
def extract_sql_from_response(response):
    queries = re.findall(r"SELECT.*?;", response, re.IGNORECASE | re.DOTALL)
    if queries:
        return queries[0].strip().rstrip(';')
    else:
        raise ValueError("No valid SQL query found.")

# Function to correct the case of room and device names in the SQL query
def correct_case_in_query(sql_query, schema_info=schema_info, device_location_info=device_location_info):
    """
    Corrects the case of room and device names in the SQL query based on the provided schema and device-location information.

    Args:
    sql_query (str): The SQL query with potential case mismatches.w
    schema_info (dict): A dictionary containing table names and their columns.
    device_location_info (list): A list of tuples with device types and locations.

    Returns:
    str: The corrected SQL query with the proper case for room and device names.
    """
    # Create a mapping of devices and rooms to their correct case
    device_location_dict = {device.lower(): (device, location) for device, location in device_location_info}
    table_names = [table.lower() for table in schema_info.keys()]

    # Function to replace device and room names with correct cases
    def replace_case(match):
        word = match.group(0).lower()
        if word in device_location_dict:
            return device_location_dict[word][0]  # Return correct case for device
        elif word in table_names:
            for table in schema_info.keys():
                if table.lower() == word:
                    return table  # Return correct case for table/room
        return match.group(0)

    # Use regex to find and replace device/room names in the SQL query
    import re
    corrected_query = re.sub(r'\b\w+\b', replace_case, sql_query)

    return corrected_query

# Query vector database
def query_vector_db(query, top_k=3):
    """
    Query the vector database for relevant insights.
    """
    global ind
    print(type(ind))
    query_embedding = embedding_model.encode([query]).astype('float32')
    distances, indices = ind.search(query_embedding, top_k)
    results = [metadata[idx] for idx in indices[0]]
    return results

# Function to execute an SQL query
def execute_sql_query(query,db_file=r"data\smarthome.db"):
    try:
        # Connect to SQLite database
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # Execute the query
        cursor.execute(query)

        # If it's a SELECT query, fetch and return the results
        if query.strip().upper().startswith("SELECT"):
            result = cursor.fetchall()
            return result
        else:
            # Commit changes for INSERT, UPDATE, DELETE queries
            conn.commit()
            print("Query executed successfully.")
        
        # Close connection
        conn.close()
    
    except sqlite3.Error as e:
        print("Error:", e)
        return f"Error executing query: {e}"

# Function to interact with LLM for generating sub-queries
def generate_sub_queries(user_query):
    """
    Use LLM to generate sub-questions dynamically based on the input query.
    """
    prompt = f"""
User Query: "{user_query}"

Schema:
    {schema_info}

Device-Location Information:
    {device_location_info}

Role : Your job is to break down a user question into smaller, **specific questions** to retrieve **detailed data** for all relevant rooms and devices based on the Schema and Device-Location Information provided.

Task:
    Focus on Relevance: Break down the given user query into specific sub-questions to fetch only the relevant information based on the schema.
    Device and Location Specificity: Use the device-location information to ensure sub-questions are limited to the specific devices and locations mentioned in the user query.
    Targeted Metrics: Ensure the sub-questions focus solely on the relevant metrics for the devices and locations specified in the query. Include aspects like energy consumption, usage hours, playback details, etc., only for the mentioned devices.
    Structured and Concise: Maintain a clean and structured format. Each sub-question must be concise, focusing on a specific device, metric, or usage aspect within the given time range (e.g., March).
    Avoid Irrelevance: Do not include sub-questions related to devices, metrics, or locations not explicitly mentioned in the user query.
    Comprehensive Coverage: Cover all possible aspects of the specified devices and metrics comprehensively without duplication.
    Output Format: Return only the sub-questions as a list. Do not include extra explanations, headers, or the original user query in the response.
    IF SQL QUERY CAN BE DIRECTLY CREATED FOR THE USER QUESTION THEN JUST RETURN THE USER QUESTION IN A LIST
Few-shot Examples:

User Query:"Analyze the air conditioner usage in Room3 during summer 2024."
Generated Sub-Questions: 
[
"How many hours was the air conditioner in Room3 used during summer 2024?",
"What was the total energy consumption of the air conditioner in Room3 during summer 2024?",
"What was the average temperature set on the air conditioner in Room3 during summer 2024?",
"On which day was the air conditioner used the most in Room3 during summer 2024?"
]



User Query: "Summarize the events of March."
Generated Sub-Questions:
[
    "How many hours was the TV in Room1 used during March?",
    "What was the energy consumed by the TV in Room1 during March?",
    "What were the most-played applications on the TV in Room1 during March?",
    "How many hours was the fan in Room1 used during March?",
    "What was the energy consumed by the fan in Room1 during March?",
    "What was the average brightness level in Room1 during March?",
    "What was the average humidity level in Room1 during March?",
    "What was the average temperature in Room1 during March?",
    "How many hours was the washing machine in Room2 used during March?",
    "How much water did the washing machine in Room2 consume during March?",
    "What was the energy consumed by the washing machine in Room2 during March?",
    "What was the average brightness level in Room2 during March?",
    "What was the average humidity level in Room2 during March?",
    "What was the average temperature in Room2 during March?",
    "How many hours was the AC in Room3 used during March?",
    "What was the energy consumed by the AC in Room3 during March?",
    "What was the average temperature set on the AC in Room3 during March?",
    "How many hours was the oven in the kitchen used during March?",
    "What was the energy consumed by the oven in the kitchen during March?",
    "What was the average brightness level in the kitchen during March?",
    "What was the average humidity level in the kitchen during March?",
    "What was the average temperature in the kitchen during March?",
    "How many hours was the light in the bathroom used during March?",
    "What was the average brightness level in the bathroom during March?",
    "What was the average humidity level in the bathroom during March?",
    "What was the average temperature in the bathroom during March?",
    "How many hours was the light in the toilet used during March?",
    "What was the average brightness level in the toilet during March?",
    "What was the average humidity level in the toilet during March?",
    "What was the average temperature in the toilet during March?"
]

User Query: "Summarize the device usage in the kitchen during june 2024."
Generated Sub-Questions:
[
    "How many hours was the oven in the kitchen used during june 2024?",
    "What was the energy consumption of the oven in the kitchen during june 2024?",
    "What is the most frequently used mode of the oven in the kitchen during june 2024?",
    "How many hours was the light in the kitchen turned on during june 2024?",
    "What was the energy consumption of the light in the kitchen during june 2024?",
]

User Query: "How does the energy consumption of the oven in March 2024 compare to February 2024?"
Generated Sub-Questions:
[
    "what is the total energy consumption of the oven in February 2024 ?",
    "what is the total energy consumption of the oven in March 2024 ?"
]

User Query: "Which months had the highest and lowest energy consumption during 2024?"
Generated Sub-Questions:
[
    "What was the total energy consumption of lights for each month in 2024?"
    "What was the total energy consumption of fans for each month in 2024?"
    "What was the total energy consumption of ac for each month in 2024?"
    "What was the total energy consumption of oven for each month in 2024?"
    "What was the total energy consumption of washing machine for each month in 2024?"
    "What was the total energy consumption of tv for each month in 2024?"
]

Generated Sub-Questions:

    """
    # Use the LLM to generate sub-questions
    response = generate_response(prompt)
    return response.splitlines()

# Function to validate the generated sub-questions using an LLM
def validate_sub_questions_llm(user_question, generated_sub_questions):
    """
    Use an LLM to validate whether the generated sub-questions are relevant and complete.
    """

    prompt = f"""
    **Role**: You are an expert in SQL and smart home data analysis. 
    Your task is to validate whether the **entire set** of generated sub-questions correctly aligns with the user's original query.  

    **User Question**: "{user_question}"

    **Generated Sub-Questions (Analyze Together)**:
    {generated_sub_questions}

    **Schema**:
    {schema_info}

    **Device-Location Information**:
    {device_location_info}

    ## **Validation Rules**:
    - Consider **ALL** sub-questions together.
    - Ensure that the **complete** information is covered.
    - If any question is missing, add it under `"missing_questions"`.
    - If any question is irrelevant, add it under `"extra_questions"`.
    - **Only return a single, final JSON object** after reviewing all questions.

    **Output Format:**
    ```json
    {{
        "status": "Valid" OR "Needs Improvement",
        "missing_questions": ["QUESTION 1", "QUESTION 2"],
        "extra_questions": ["QUESTION 3", "QUESTION 4"]
    }}
    ```

    ---

    ## **Few-Shot Examples for LLM Guidance**

    ### **Example 1: Correct Sub-Questions**
    **User Query:** "Summarize the device usage in Room1 during April."
    
    **Generated Sub-Questions:**
    [
        "How many hours was the TV in Room1 used during April?",
        "What was the energy consumed by the TV in Room1 during April?",
        "What were the most-played applications on the TV in Room1 during April?",
        "How many hours was the fan in Room1 used during April?",
        "What was the energy consumed by the fan in Room1 during April?",
        "How many hours was the light in Room1 used during April?",
        "What was the energy consumed by the light in Room1 during April?"
    ]

    **Validation Output:**
    ```json
    {{
        "status": "Valid",
        "missing_questions": [],
        "extra_questions": []
    }}
    ```

    ---

    ### **Example 2: Missing a Necessary Question**
    **User Query:** "Summarize the events of the kitchen during summer."

    **Generated Sub-Questions:**
    [
        "How many hours was the oven in the kitchen used during summer?",
        "What was the energy consumption of the oven in the kitchen during summer?",
        "How many hours was the light in the kitchen used during summer?",
        "What was the energy consumption of the light in the kitchen during summer?",
        "What was the average brightness level in the kitchen during summer?",
        "What was the average humidity level in the kitchen during summer?"
    ]

    **Validation Output:**
    ```json
    {{
        "status": "Needs Improvement",
        "missing_questions": ["what is the most used mode of the oven in the kitchen during summer", "What was the average temperature in the kitchen during summer?"],
        "extra_questions": []
    }}
    ```

    ---

    ### **Example 3: Irrelevant Questions Included**
    **User Query:** "What was the highest recorded temperature in the kitchen during summer 2024?"

    **Generated Sub-Questions:**
    [
        "What was the highest temperature recorded in the kitchen during summer 2024?",
        "What was the highest brightness level in the kitchen during summer 2024?",
        "What was the highest humidity level in the kitchen during summer 2024?"
    ]

    **Validation Output:**
    ```json
    {{
        "status": "Needs Improvement",
        "missing_questions": [],
        "extra_questions": ["What was the highest brightness level in the kitchen during summer 2024?", "What was the highest humidity level in the kitchen during summer 2024?"]
    }}
    ```

    ---

    ### **Example 4: Both Missing & Extra Questions**
    **User Query:** "Analyze the air conditioner usage in Room3 during summer 2024."

    **Generated Sub-Questions:**
    [
        "How many hours was the air conditioner in Room3 used during summer 2024?",
        "What was the total energy consumption of the air conditioner in Room3 during summer 2024?",
        "What was the average temperature set on the air conditioner in Room2 during summer 2024?",
        "On which day was the air conditioner used the most in Room3 during Winter 2024?"
    ]

    **Validation Output:**
    ```json
    {{
        "status": "Needs Improvement",
        "missing_questions": ["What was the average temperature set on the air conditioner in Room3 during summer 2024?", "On which day was the air conditioner used the most in Room3 during summer 2024?"],
        "extra_questions": ["What was the average temperature set on the air conditioner in Room2 during summer 2024?", "On which day was the air conditioner used the most in Room3 during Winter 2024?"]
    }}
    ```

    ---

    ## **Now validate the following user question and its generated sub-questions:**
    **User Query:** "{user_question}"

    **Generated Sub-Questions:**
    {generated_sub_questions}

    **Validation Output (Strictly JSON Format):**
    """

    # Call the LLM to evaluate the sub-questions
    raw_response = generate_response(prompt)

    # Extract JSON from response
    try:
        json_start = raw_response.find("{")  # Find first `{`
        json_end = raw_response.rfind("}")   # Find last `}`
        if json_start == -1 or json_end == -1:
            raise ValueError("No valid JSON found in response")

        json_string = raw_response[json_start : json_end + 1]  # Extract JSON part
        validation_result = json.loads(json_string)  # Convert to dictionary
        return validation_result

    except json.JSONDecodeError as e:
        print(" Error: Failed to parse JSON response from LLM:", e)
        print("Raw LLM Response:\n", raw_response)
        return {"status": "Error", "missing_questions": [], "extra_questions": [], "error": "Invalid JSON output from LLM"}

    except Exception as e:
        print(" Unexpected Error:", e)
        return {"status": "Error", "missing_questions": [], "extra_questions": [], "error": str(e)}

# Function to extract a list of sub-questions from the LLM response
def extract_json(response_text):
    """
    Extracts a JSON object from the LLM response text.
    If the response is not valid JSON, it attempts to fix common formatting errors.
    """
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        try:
            # Handle cases where the response has extra text before/after JSON
            json_str = response_text[response_text.index("{"):response_text.rindex("}") + 1]
            return json.loads(json_str)
        except:
            print("Error: LLM returned invalid JSON.")
            return {"category": "direct_query"}  # Default to direct_query if parsing fails

#Function to classify the user question
def classify_query(user_question):
    """
    Classifies the user question into:
    - "summary_query": Requires sub-questions
    - "direct_query": Can be answered with a single SQL query
    """
    prompt = f"""
    You are an expert in SQL and data retrieval. Classify the given user question into one of the following categories:
    - "summary_query": If the question requires summarizing multiple aspects or events over a period of time.
    - "direct_query": If the question can be answered using a single SQL query without generating sub-questions.

    Here are some examples:

    Example 1:
    User Question: "What is the average temperature in Room1 for the past 6 months?"
    Output: {{"category": "direct_query"}}

    Example 2:
    User Question: "What is the current temperature in Room1?"
    Output: {{"category": "direct_query"}}

    Example 3:
    User Question: "How has the temperature trend changed in Room1 over the last 5 years?"
    Output: {{"category": "summary_query"}}

    Example 4:
    User Question: "Give me the summary of device usage in Room2 in october 2024?"
    Output: {{"category": "summary_query"}}

    Example 5:
    User Question: "What was the maximum power consumption of the oven in January 2024?"
    Output: {{"category": "direct_query"}}

    Example 6:
    User Question: "List all device activations in Room1 between 8 PM and 10 PM yesterday."
    Output: {{"category": "direct_query"}}

    Now, classify the following question:
    User Question: "{user_question}"

    Output format:
    {{"category": "summary_query" OR "direct_query"}}
    """

    response = generate_response(prompt)  # Call LLM
    return extract_json(response).get("category", "direct_query")  # Default to direct_query if uncertain

# Function to generate sub-questions for a user question
def generate_validated_sub_questions(user_question):
    """
    Generate sub-questions for a user question **only if needed**.
    If incorrect, retry generating new ones up to MAX_RETRIES times.
    """

    # 🔹 Classify the query type before proceeding
    query_type = classify_query(user_question)
  

    if query_type == "direct_query":
        print("Skipping sub-question generation. Direct SQL query required.")
        return [user_question]  # No sub-questions needed

    attempt = 0
    while attempt < MAX_RETRIES:
        attempt += 1
        print(f"Attempt {attempt}: Generating sub-questions...")

        # Generate sub-questions
        sub_questions_response = generate_sub_queries(user_question)
        sub_questions = extract_list_from_response(sub_questions_response)

        # Validate sub-questions
        validation_result = validate_sub_questions_llm(user_question, sub_questions)
        print(validation_result)
        print("---------" * 15)
        print(validation_result["status"])

        if validation_result["status"] == "Valid":
            print("Sub-questions are valid.")
            print(sub_questions)
            return sub_questions

        # Otherwise, print the issues and retry
        print(f"Validation failed. Missing Questions: {validation_result.get('missing_questions', [])}")
        print(f"Extra Questions: {validation_result.get('extra_questions', [])}")
        time.sleep(1)  # Add delay before retrying

    print("Failed to generate correct sub-questions after multiple attempts.")
    return None  # Return None if all retries fail

# Function to execute sub-questions and validate the results
def execute_sub_prompts(sub_prompts):
    """
    Execute each sub-prompt using the provided execution function and return results.

    Args:
        sub_prompts (list): A list of sub-prompts (questions).
        execution_function (callable): A function that takes a sub-prompt and returns its result.

    Returns:
        dict: A dictionary with sub-prompts as keys and their corresponding results as values.
    """
    results = {}
    
    for sub_prompt in sub_prompts:
        try:
            # Execute the sub-prompt using the provided function
            #print("-"*15)
            result = run_with_retries(sub_prompt)
            #print("-"*15) 
            results[sub_prompt] = result
        except Exception as e:
            # Handle errors gracefully
            results[sub_prompt] = f"Error: {e}"

    return results

# Function to extract numerical values from a text
def extract_numbers(text):
    """
    Extracts all numerical values from a given text.
    """
    return [float(num) for num in re.findall(r'\d+\.\d+|\d+', text)]

# Function to check for hallucinations in the summary Generation using LLM
def check_hallucinations(summary, sub_prompt_results, vector_insights, tolerance=0.01):
    """
    Enhanced function to detect hallucinations with tolerance for rounding errors.

    Args:
        summary (str): The summary generated by the LLM.
        sub_prompt_results (dict): Dictionary with sub-questions as keys and their corresponding numerical results.
        vector_insights (list): List of historical insights containing numerical data.
        tolerance (float): Tolerance for rounding differences between numbers.

    Returns:
        dict: Contains details of matched, extra, and missing values.
    """
    # Extract numbers from the summary
    summary_numbers = extract_numbers(summary)
    print("summary nos : ",summary_numbers)
    key_numbers = []
    for res in sub_prompt_results.keys():
        key_numbers.extend(extract_numbers(res))
    key_numbers = extract_numbers(summary)
    print("key_numbers nos : ",key_numbers)
    # Extract numbers from sub-prompt results (SQL results)
    actual_numbers = []
    for result in sub_prompt_results.values():
        if isinstance(result, (list, tuple)):
            for value in result:
                if isinstance(value, (list, tuple)):
                    actual_numbers.extend([float(v) for v in value if isinstance(v, (int, float))])
                elif isinstance(value, (int, float)):
                    actual_numbers.append(float(value))
        elif isinstance(result, (int, float)):
            actual_numbers.append(float(result))
    print("act nos : ",actual_numbers)
    # Extract numbers from vector insights (historical data)
    vector_numbers = []
    for insight in vector_insights:
        vector_numbers.extend(extract_numbers(insight))
    print("valid nos : ",vector_numbers)
    # Combine both actual numbers and vector numbers
    actual_numbers = actual_numbers + key_numbers
    valid_numbers = set(round(n, 2) for n in (actual_numbers + vector_numbers))

    # Round summary numbers for tolerance comparison
    summary_numbers_rounded = {round(n, 2) for n in summary_numbers}

    # Identify inconsistencies
    extra_values = summary_numbers_rounded - valid_numbers
    missing_values = valid_numbers - summary_numbers_rounded
    matched_values = summary_numbers_rounded & valid_numbers

    return {
        "matched_values": sorted(matched_values),
        "extra_values_in_summary": sorted(extra_values),
        "missing_values_in_summary": sorted(missing_values),
        "hallucination_detected": bool(extra_values),
    }

# Function to generate a user-friendly summary using an LLM
def generate_user_friendly_summary(user_question, sub_prompt_results, vector_insights, max_retries=3):
    """
    Generate a user-friendly, concise summary using an LLM based on the results of sub-prompts.
    If hallucinations are detected, it retries up to `max_retries` times.

    Args:
        sub_prompt_results (dict): Dictionary with sub-prompts as keys and their corresponding results.
        vector_insights (list): List of historical insights containing numerical data.
        max_retries (int): Maximum number of retries if hallucinations are detected.

    Returns:
        str: A concise summary generated by the LLM without hallucinations.
    """
    # Format results and vector insights into a prompt-friendly string
    formatted_results = "\n".join([f"{sub_prompt}: {result}" for sub_prompt, result in sub_prompt_results.items()])
    formatted_insights = "\n".join(vector_insights)

    attempt = 0
    while attempt < max_retries:
        attempt += 1
        print(f" Attempt {attempt}: Generating summary...")

        # Define the LLM prompt
        prompt = f"""
user's original question : {user_question}

The following are the detailed results from various sub-queries related to a user's original question:

{formatted_results}

Additionally, here are some historical insights based on past data:

{formatted_insights}

role:
    Please generate a concise, user-friendly summary of these results in accordance with user's original question. Integrate both the detailed results and historical insights to provide context, highlight trends, and offer relevant comparisons. Focus on key insights, avoid repetition, and ensure clarity.

Instructions:
    - Use only the provided numerical data.
    - Avoid making assumptions or fabricating numbers.
    - Just return the summary without any explanations or extra text.
    - Use the SQL result as the definitive answer to the user's question. This result is accurate and should not be altered or calculated based on the vector insights.
    - Use the vector insights only to compare the current answer with past trends or provide additional context. Do not use the vector insights to predict or modify the SQL result.
    - Generate a response that clearly answers the user's original question using the SQL result while providing additional insights based on the vector database.
    - JUST GIVE THE ANSWER AND THE INSIGHTS 

Few-shot Examples:
1. Sub-Prompts and Results:
   - "How many hours was the oven in the kitchen used during March?": "[(13.15,)]"
   - "What was the energy consumed by the oven in the kitchen during March?": "[(129.18,)]"
   - "What was the average brightness level in the kitchen during March?": "[(4.17,)]"
   - "What was the average humidity level in the kitchen during March?": "[(46.89,)]"
   - "What was the average temperature in the kitchen during March?": "[(18.2265,)]"

   Historical Insights:
   ['In March 2016, the Oven consumed approximately 161.0 kWh of energy and operated for an average of 7.5 hours per day.', 
    'In March 2015, the Oven consumed approximately 190.9 kWh of energy and operated for an average of 7.8 hours per day.']

   Generated Summary:
   "In March, the oven in the kitchen was used for 13.15 hours, consuming 129.18 kWh of energy. The kitchen had an average brightness level of 4.17 lux, a humidity level of 46.89%, and a temperature of 18.23°C. Historically, in March 2016, the oven operated for an average of 7.5 hours per day and consumed 161.0 kWh of energy, showing a decrease in both usage and energy consumption over time."

Now, generate a similar summary for the following results:

Sub-Prompts and Results:
{formatted_results}

Generated Summary:
        """

        # Generate the summary
        summary = generate_response(prompt).strip()
        print(f"Generated Summary:\n{summary}\n")

        # Validate hallucinations
        hallucination_check = check_hallucinations(summary, sub_prompt_results, vector_insights)

        if not hallucination_check["hallucination_detected"]:
            print("No hallucinations detected. Returning final summary.\n")
            return summary
        else:
            print(f"Hallucinations detected in attempt {attempt}. Retrying...\n")
            print(f"Extra values: {hallucination_check['extra_values_in_summary']}")
            print(f"Missing values: {hallucination_check['missing_values_in_summary']}")

    print("Failed to generate a hallucination-free summary after retries.")
    return "Error: Unable to generate an accurate summary after multiple attempts."

# Function to extract a list from a response
def extract_list_from_response(response_list):
    """
    Extract the content between the first `[` and the last `]` from a list of strings.
    
    Args:
        response_list (list): A list of strings containing sub-questions and other text.

    Returns:
        list: A list of sub-questions extracted from between the brackets.
    """
    try:
        # Ensure response_list is a valid list of strings
        if not isinstance(response_list, list) or not all(isinstance(item, str) for item in response_list):
            logging.error(f"Invalid response format: {response_list}")
            return []

        # Join all list items into a single string
        combined_text = ''.join(response_list)

        # Find the first occurrence of '[' and the last occurrence of ']'
        start_idx = combined_text.find('[')
        end_idx = combined_text.rfind(']')

        # Ensure both brackets exist in the response
        if start_idx == -1 or end_idx == -1 or start_idx > end_idx:
            logging.error(f"Malformed response: Missing brackets in {combined_text}")
            return []

        # Extract content inside brackets
        content_between_brackets = combined_text[start_idx + 1:end_idx]

        # Split by commas, remove unnecessary quotes & strip whitespace
        sub_questions = [item.strip().strip('"').strip("'") for item in content_between_brackets.split(',')]

        # Filter out empty items
        sub_questions = [q for q in sub_questions if q]

        logging.info(f"Extracted sub-questions: {sub_questions}")
        return sub_questions

    except Exception as e:
        logging.error(f"Error extracting list from response: {response_list}. Exception: {e}")
        return []

#Functiion to ensure the encoding is correct
def fix_encoding(text):
    """
    Fix encoding issues related to temperature symbols.
    """
    return text.encode('latin1').decode('utf-8') if "ï¿½" in text else text

#Function to run the Flask app
from flask import Flask, request, render_template, jsonify

# Initialize the Flask application
app = Flask(__name__)

# Define a route for the default URL
@app.route('/')
def index():
    return render_template('index.html')

# Define a route for the prediction
@app.route('/process_query', methods=['POST'])

# Function to process the user query
def process_query():
              
    user_question = request.form['question']                          # Get the user question from the form
    sub_questions = generate_validated_sub_questions(user_question)   # Generate sub-questions
    print(sub_questions)
    sub_prompt_results = execute_sub_prompts(sub_questions)           # Execute sub-questions
    historical_insights = query_vector_db(user_question)              # Query vector database
    summary = generate_user_friendly_summary(user_question,sub_prompt_results,historical_insights)      # Generate user-friendly summary
    sub_prompt_results_str = {str(k): str(v) for k, v in sub_prompt_results.items()}                    # Convert results to string
    historical_insights_str = [str(insight) for insight in historical_insights]                         # Convert insights to string
    summary_str = str(summary)                                        # Convert summary to string
    finalized_summary=fix_encoding(summary_str)                       # Fix encoding issues in the summary
    
    # Return the results as a JSON response
    return jsonify({
        "sql_queries": sub_prompt_results_str,
        "vector_insights": historical_insights_str,
        "final_summary": finalized_summary
    })

# Run the app
if __name__ == '__main__':
    app.run(debug=False)

