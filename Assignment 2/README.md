# Autonomous Research Agent (LangChain + Gemini)

## Overview

This project presents an Autonomous Research Agent built using LangChain and the Google Gemini API.
The system is capable of performing automated research by:

•⁠  ⁠Retrieving information from the web
•⁠  ⁠Extracting relevant insights
•⁠  ⁠Generating a structured report

---

## Objective

The goal of this project is to design an AI-based system that:

•⁠  ⁠Accepts a topic as input
•⁠  ⁠Conducts automated research using external sources
•⁠  ⁠Produces a well-organized report

---

## Technologies Used

•⁠  ⁠Python
•⁠  ⁠LangChain
•⁠  ⁠Google Gemini API
•⁠  ⁠DuckDuckGo Search
•⁠  ⁠Wikipedia API

---

## Features

•⁠  ⁠Web-based information retrieval using DuckDuckGo
•⁠  ⁠Knowledge extraction from Wikipedia
•⁠  ⁠Report generation using Gemini LLM
•⁠  ⁠Structured output including:

  * Cover Page
  * Introduction
  * Key Findings
  * Challenges
  * Future Scope
  * Conclusion

---

## Project Structure


Autonomous-Research-Agent/
│── main.py
│── requirements.txt
│── README.md
│── sample_outputs/
│     ├── report1.txt
│     ├── report2.txt


---

## How to Run

### 1. Clone the repository


git clone https://github.com/Kumkum-Mishra/Autonomous-Research-Agent.git
cd Autonomous-Research-Agent


### 2. Install required packages


pip install -r requirements.txt


### 3. Configure API key


export GOOGLE_API_KEY="your_api_key"


(Alternatively, you can set it directly inside the script if using Colab.)

### 4. Execute the program


python main.py


---

## Sample Topics

•⁠  ⁠AI in Education
•⁠  ⁠Impact of AI in Healthcare
•⁠  ⁠Climate Change Impact

---

## Output Format

The generated report follows a structured format consisting of:

•⁠  ⁠Cover Page
•⁠  ⁠Introduction
•⁠  ⁠Key Findings
•⁠  ⁠Challenges
•⁠  ⁠Future Scope
•⁠  ⁠Conclusion

---

## Key Learnings

•⁠  ⁠Understanding LangChain tools and integrations
•⁠  ⁠Working with large language models (Gemini API)
•⁠  ⁠Handling dependency and environment-related issues
•⁠  ⁠Building simple autonomous AI systems

---

## Conclusion

This project highlights how AI agents can automate research workflows by combining external data sources with language models to generate meaningful and structured outputs.

---

## Author

Devesh Kumar Gola
2023399094
CSH-G2