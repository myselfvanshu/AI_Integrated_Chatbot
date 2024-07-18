### AI_Integrated_Chatbot

This project demonstrates a chatbot using Streamlit and various AI tools. The chatbot can embed a PDF document and answer questions based on the document's content.


### Setup and Run
1. Create a directory. 

2. Open terminal and create a virtual environment with python version 3.10 because integrating OpenAI and LLM is more suitable for python version 3.10 .
            >> conda create -p venv python==3.10 -y

3. Make a “.env” file and in it put your GROQ_API_KEY as
    GROQ_API_KEY=”your_groq_api_key”

4. Activate the environment
            >>conda activate venv/

5. Make a file “requirements.txt” and put the necessary libraries needed and install them as well.
            >>pip install -r requirements.txt

6. Make a python file for example “app.py” and write the code there.

7. After completing the code, open terminal and run the code.
            >>streamlit run app.py

### Step 1: Clone the Repository

Clone the repository to your local machine:

```sh
git clone <repository-url>
cd <repository-directory>

