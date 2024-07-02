import pandas as pd
import re
from neo4j import GraphDatabase
from typing import List, Tuple
import tiktoken
from openai import OpenAI
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HELICONE_API_KEY=os.getenv("HELICONE_API_KEY")
user=os.getenv("user")
uri=os.getenv("uri")
password=os.getenv("password")
client = OpenAI(
    base_url="https://api.together.xyz/v1", api_key=TOGETHER_API_KEY
)
MAX_TOKENS_PER_REQUEST = 512

def clean_text(text):
    text = str(text)  # Ensure it's a string
    text = text.lower()  # Convert to lowercase
    #text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special characters
    return text

df = pd.read_csv('Scrapped_data.csv', encoding='latin-1')

# Step 2: Filter rows for 'Python Engineer' or 'Python Developer'
python_jobs = df[df['Job Name'].str.contains("Python Engineer|Python Developer", case=False, na=False)]

# Display the first 10 rows of these filtered results
python_jobs_first_10 = python_jobs.head(10)

# Step 3: Identify unique job titles
unique_job_titles = df['Job Name'].str.lower().unique()
print(unique_job_titles)

first_unique_jobs = df[df['Job Name'].str.contains("Java Developer", case=False, na=False)]
first_unique_jobs_first_10 = first_unique_jobs.head(10)

# Step 5: Combine both DataFrames
combined_df = pd.concat([python_jobs_first_10, first_unique_jobs_first_10], ignore_index=True)
combined_df.drop(['Unnamed: 4','Unnamed: 7','Unnamed: 8'],axis=1,inplace=True)
combined_df = combined_df.applymap(clean_text)

def create_knowledge_graph(df, uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))

    def add_job(tx, job_name, company_name, jd, skills, yoe, industry):
        # Create job node and related nodes/relationships
        create_query = """
        MERGE (j:Job {name: $job_name, jd: $jd})
        MERGE (c:Company {name: $company_name})
        MERGE (j)-[:JOB_DESCRIPTION_FOR]->(c)
        MERGE (e:Experience {years: $yoe})
        MERGE (j)-[:EXPERIENCE_REQUIRED]->(e)
        WITH j
        UNWIND $skills AS skill
        MERGE (s:Skill {name: skill})
        MERGE (j)-[:REQUIRES]->(s)
        WITH j
        UNWIND $industry AS ind
        MERGE (i:Industry {name: ind})
        MERGE (j)-[:INDUSTRY]->(i)
        RETURN j
        """
        result = tx.run(create_query, job_name=job_name, company_name=company_name, jd=jd, skills=skills.split(','), yoe=yoe, industry=industry.split(', '))
        job_node = result.single()["j"]
        job_element_id = job_node.element_id

        # Update the job node with its element_id
        update_query = """
        MATCH (j:Job {name: $job_name})
        SET j.element_id = $element_id
        RETURN j
        """
        tx.run(update_query, job_name=job_name, element_id=job_element_id)
        return job_element_id

    def add_similar_to(tx, job_name):
        query = """
        MATCH (j:Job)
        WHERE j.name CONTAINS $job_name
        WITH COLLECT(j) AS jobs
        FOREACH (x IN jobs |
            FOREACH (y IN jobs |
                MERGE (x)-[:SIMILAR_TO]->(y)))
        """
        tx.run(query, job_name=job_name)

    with driver.session() as session:
        job_ids = {}
        for index, row in df.iterrows():
            job_id = session.execute_write(add_job, row['Job Name'], row['Company Name'], row['JD'], row['Skills'], row['YOE'], row['Industry:'])
            job_ids[row['Job Name']] = job_id

        for job_name in job_ids.keys():
            session.execute_write(add_similar_to, job_name)

    driver.close()

def number_of_tokens(texts: List[str]) -> List[int]:
    model = tiktoken.encoding_for_model("gpt-3.5-turbo")
    encodings = model.encode_batch(texts)
    num_of_tokens = [len(encoding) for encoding in encodings]
    return num_of_tokens

# Function to split text into chunks of a specified number of tokens
def split_text_into_chunks(text: str, max_tokens: int) -> List[str]:
    model = tiktoken.encoding_for_model("gpt-3.5-turbo")
    encoding = model.encode(text)
    chunks = []

    for i in range(0, len(encoding), max_tokens):
        chunk_encoding = encoding[i:i + max_tokens]
        chunk_text = model.decode(chunk_encoding)
        chunks.append(chunk_text)

    return chunks

# Function to split all texts into chunks and respect the total token limit per request
def prepare_chunks_for_api(texts: List[str], chunk_size: int, max_request_tokens: int) -> List[List[str]]:
    all_chunks = []
    current_batch = []
    current_batch_tokens = 0

    for text in texts:
        text_chunks = split_text_into_chunks(text, chunk_size)
        for chunk in text_chunks:
            chunk_tokens = len(tiktoken.encoding_for_model("gpt-3.5-turbo").encode(chunk))

            if current_batch_tokens + chunk_tokens > max_request_tokens:
                all_chunks.append(current_batch)
                current_batch = [chunk]
                current_batch_tokens = chunk_tokens
            else:
                current_batch.append(chunk)
                current_batch_tokens += chunk_tokens

    if current_batch:
        all_chunks.append(current_batch)

    return all_chunks

# Function to fetch job data from Neo4j
def fetch_job_data(uri: str, user: str, password: str) -> List[Tuple[str, str, str, str, str]]:
    driver = GraphDatabase.driver(uri, auth=(user, password))

    def get_job_data(tx):
        query = """
        MATCH (j:Job)
        MATCH (s:Skill)
        MATCH (i:Industry)
        RETURN elementId(j) AS id, j.name AS job_name, j.jd AS jd,
               COALESCE(s.name, '') AS skills, COALESCE(i.name, '') AS industry
        """
        result = tx.run(query)
        return [(record['id'], record['job_name'], record['jd'], record['skills'], record['industry']) for record in result]

    with driver.session() as session:
        job_data = session.execute_read(get_job_data)

    driver.close()
    return job_data

# Fetch job data from Neo4j
#job_data = fetch_job_data(uri, user, password)


# Clean the job data to replace None with empty strings
#cleaned_job_data = [
#    (id, job_name or "", jd or "", skills or "", industry or "")
 #   for id, job_name, jd, skills, industry in job_data
#]


# Generate combined texts
#combined_texts = [
 #   " ".join([job_name, jd, skills, industry])
 #   for _, job_name, jd, skills, industry in cleaned_job_data
#]

# Prepare the text chunks for the API
#text_chunks = prepare_chunks_for_api(combined_texts, chunk_size=512, max_request_tokens=MAX_TOKENS_PER_REQUEST)

# Function to generate embeddings for each chunk
def generate_embeddings(chunks: List[str]):
    embeddings = []
    for chunk in chunks:
        response = client.embeddings.create(input=chunk, model="BAAI/bge-large-en-v1.5")
        chunk_embeddings = [item.embedding for item in response.data]
        embeddings.extend(chunk_embeddings)
    return embeddings


