# Copyright 2025 FIZ-Karlsruhe (Mustafa Sofean)

import google.generativeai as genai
import time
from dotenv import load_dotenv
import os

load_dotenv()

os.environ['HTTP_PROXY'] = os.getenv('HTTP_PROXY')
os.environ['HTTPS_PROXY'] = os.getenv('HTTPS_PROXY')

gemini_api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=gemini_api_key)
gemini_model = os.getenv('GEMINI_API_MODEL')


def is_plasma(text: str):
    """

    :param text:
    :return:
    """
    prompt = f"""
    You will be provided with patent texts.\n
              your task is to identify if the texts relate to Plasma Physics or not. 
              Just answer with YES or NO. \n
              Use the following abstract to understand the Plasma Physics domain: \n
              Plasma technology involves the use of a partially or fully ionized gas known as plasma for 
              various industrial, medical, and scientific applications. one application of plasma physics 
              is low-temperature plasmas which include surface treatment and modification of materials, 
              such as improving adhesion, creating surface patterns, or depositing thin films. 
              It is also used in biomedical applications, such as sterilization and wound healing. 
              Another application of plasma technology are Plasma medicine which involves the use of plasma 
              to treat various medical conditions and diseases, and Plasma decontamination which is 
              a process that uses plasma to remove or inactivate biological or chemical contaminants from 
              a surface or environment. \n
            

              Text:  '''{text}'''\n
              Answer:
              """

    model = genai.GenerativeModel(gemini_model)
    response = model.generate_content(prompt)

    print(response.text.strip())
    time.sleep(5)

    return response.text.strip()
