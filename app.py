import streamlit as st
import json
from graphrag import GraphDatabase,GROQ_API_KEY,OpenAI,HELICONE_API_KEY,re,uri,user,password
SysPromptDefault = "You are now in the role of an expert AI."
def fetch_job_descriptions(uri, user, password, company_name, job_title):
    driver = GraphDatabase.driver(uri, auth=(user, password))

    def get_job_descriptions(tx, company_name, job_title):
        query = """
        MATCH (c:Company {name: $company_name})<-[:JOB_DESCRIPTION_FOR]-(j:Job {name: $job_title})-[:EXPERIENCE_REQUIRED]->(e:Experience)
        RETURN j.jd AS jd, e.years AS experience
        """
        result = tx.run(query, company_name=company_name, job_title=job_title)
        return [(record['experience'], record['jd']) for record in result]

    with driver.session() as session:
        job_descriptions = session.execute_read(get_job_descriptions, company_name, job_title)

    driver.close()
    return job_descriptions
def response(message: object, model: object = "llama3-8b-8192", SysPrompt: object = SysPromptDefault, temperature: object = 0.2) -> object:
    """

    :rtype: object
    """
    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://gateway.hconeai.com/openai/v1",
        default_headers={
            "Helicone-Auth": f"Bearer {HELICONE_API_KEY}",
            "Helicone-Target-Url": "https://api.groq.com"
        }
    )

    messages = [{"role": "system", "content": SysPrompt}, {"role": "user", "content": message}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        frequency_penalty=0.2,
    )
    return response.choices[0].message.content

extraction_prompt="""
**You are an expert in natural language processing and information extraction. You have over 20 years of experience in developing and refining algorithms to accurately identify and extract structured data from unstructured text.**

**Objective:** Extract specific information from a given job QUERY. The information needed includes `company_name`, `job_name`, `new_requirements`, and `years_of_experience`. The output should be formatted as valid JSON.

**Steps to complete the task:**

1. **Identify the `company_name`:** Look for the name of the organization or company mentioned in the query. This is typically a proper noun or an entity representing a business.

2. **Identify the `job_name`:** Look for the title of the job or position being referred to in the query. This can include terms like "Engineer," "Manager," "Developer," etc.

3. **Identify the `new_requirements`:** Extract any specific requirements listed for the job. These might include skills, certifications, or other qualifications that are explicitly mentioned.

4. **Identify the `years_of_experience`:** Look for numerical values or phrases that indicate the required or preferred years of experience for the job. This can include terms like "years of experience," "minimum experience," etc.

5. **Format the extracted information:** Ensure the extracted information is formatted as valid JSON with the keys `company_name`, `job_name`, `new_requirements`, and `years_of_experience`.

**Example Input:**
"TechCorp is hiring a Senior Software Engineer. Candidates must have 5+ years of experience in software development and be proficient in Python and JavaScript. Experience with cloud technologies is a plus."

**Example Output:**
```json
{
  "company_name": "TechCorp",
  "job_name": "Senior Software Engineer",
  "new_requirements": ["proficient in Python", "proficient in JavaScript", "Experience with cloud technologies"],
  "years_of_experience": "5+ years"
}
```

Take a deep breath and work on this problem step-by-step.
"""

jd_prompt="""You are a senior HR manager with 20 years of experience in creating job descriptions for various roles across industries. You are tasked with revising an existing JOB_DESCRIPTION by incorporating NEW_REQUIREMENTS and adjusting the YEARS_OF_EXPERIENCE needed. Your objective is to ensure the revised job description is comprehensive, clearly outlines the role's responsibilities, and attracts the right candidates. 

Please follow these steps:

1. **Review the Existing JOB_DESCRIPTION:**
   - Analyze the responsibilities, qualifications, and experience required in the current job description.
   
2. **Incorporate NEW_REQUIREMENTS:**
   - Integrate the new requirements provided into the relevant sections of the job description.

3. **AdjustYEARS_OF_EXPERIENCE:**
   - Modify the years of experience required for the role as specified.

4. **Structure the Job Description in Markdown Format:**
   - Ensure the job description is formatted properly in Markdown, making it easy to read and professional.

**Example Format:**

```markdown
# Job Description

## Job Title: [Updated Job Title]

### Responsibilities
- [Responsibility 1]
- [Responsibility 2]
- [Responsibility 3]

### Skills 
- [New Requirement 1]
- [New Requirement 2]
- [New Requirement 3]

### Years of Experience
- [Updated years of experience required]
```

Use this template to create a revised job description using the provided inputs:  JOB_DESCRIPTION, NEW_REQUIREMENTS, and YEARS_OF_EXPERIENCE

Take a deep breath and work on this problem step-by-step."""

def extract_json(response_str):
    """Extract the JSON part from the response string."""
    match = re.search(r"\{.*}", response_str, re.DOTALL)
    if match:
        json_part = match.group()
        try:
            json.loads(json_part)  # Check if it's valid JSON
            return json_part
        except json.JSONDecodeError:
            print("Invalid JSON detected.")
    return None
def extraction(query):
    model = "llama3-70b-8192"
    message_query = f"QUERY:\n\n{query}"
    response_str = response(message=message_query, model=model, SysPrompt=extraction_prompt, temperature=0)
    json_part = extract_json(response_str)
    return json_part

def convert_to_sentence(job_descriptions):
    sentences = []
    for experience, jd in job_descriptions:
        sentence = f"{jd} Required experience: {experience}."
        sentences.append(sentence)
    return sentences

def jd(json_part):
    json_part=json.loads(json_part)
    company_name = json_part.get("company_name", "").lower()
    job_name = json_part.get("job_name", "").lower()
    new_requirements = json_part.get("new_requirements", "")
    years_of_experience = json_part.get("years_of_experience","").lower()
    previous_jd=fetch_job_descriptions(uri, user, password, company_name=company_name, job_title=job_name)
    sentences = convert_to_sentence(previous_jd)
    print(company_name)
    print(job_name)
    print(sentences)
    model = "llama3-70b-8192"
    message=f"JOB_DECRIPTION:\n\n{sentences}\n\nNEW_REQUIREMENTS:\n\n{new_requirements}\n\nYEARS_OF_EXPERIENCE:\n\n{years_of_experience}\n\n"
    output = response(message=message, model=model, SysPrompt=jd_prompt, temperature=0)
    print(output)
    return output
def run_conversation(prompt):
    json_part = extraction(prompt)
    output = jd(json_part)
    return output

# Streamlit app implementation
def main():
    st.title("Job Descriptor")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Prompt for legal query
    if prompt := st.chat_input("What is your new requirement?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = run_conversation(prompt)

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Clear the cache after processing the query
        st.cache_data.clear()

if __name__ == "__main__":
    main()
