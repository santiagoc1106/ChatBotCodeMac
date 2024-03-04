import speech_recognition as sr
import os
import os
import datetime
import numpy as np
from gtts import gTTS
import re 
import json
import time 
import python_weather
import asyncio
import pickle
import selenium.webdriver as webdriver
from selenium import webdriver
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import librosa as lb
import os
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

spanish = None
name = None
x = None
# x = [
    
#     "my name is", "Mi nombre es", "me llamo", "soy", "temperature", "what is the date", "what day is it", "what's today",
#      "fecha", "que dia es hoy", "tiempo", "hora", "show my voice", "I live", "what is my name", "what's my name", 
#      "es Mi nombre", "es mi nombre", "where am I from", "time", "thank","thanks","I appreciate", "gracias","aprecio", "bye","goodbye","shut down", "adios","hasta luego","apagate", 
#      "repeat", "say", "Calculate", "calculate", "what is", "what's", 
     
#      ]
class ChatBot():

    def __init__(self, name):
        print("...", name, "...")
        self.name = name

    def speech_to_text(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as mic:
            
            if spanish == None:
                print("Listening/Escuchando...")
            if spanish == True:
                print("Escuchando...")
            if spanish == False:
                print("Listening...")
            audio = recognizer.listen(mic)
            self.text="ERROR"
        try:
            self.text = recognizer.recognize_google(audio)
            print("User --> ", self.text)
        except:
            print("User -->  ERROR")

    @staticmethod
    def text_to_speech(text):
        print("JARVIS--> ", text)
        if spanish == False:
            speaker = gTTS(text=text, lang="en", slow=False)
            speaker.save("res.mp3")
            statbuf = os.stat("res.mp3")
            mbytes = statbuf.st_size / 1024
            duration = mbytes / 200
            os.system('start res.mp3')  #if you are using mac->afplay or else for windows->start
            #os.system("close res.mp3")
            time.sleep(int(7*duration))
            os.remove("res.mp3")

        if spanish == True:
            speaker = gTTS(text=text, lang="es", slow=False)
            speaker.save("res.mp3")
            statbuf = os.stat("res.mp3")
            mbytes = statbuf.st_size / 1024
            duration = mbytes / 200
            os.system('afplay res.mp3')
            #os.system("close res.mp3")
            time.sleep(int(7*duration))
            os.remove("res.mp3")

    def wake_up(self, text):
        return True if self.name in text.lower() else False
    
    @staticmethod
    def action_time():
        return datetime.datetime.now().time().strftime('%I:%M %p')
    
    @staticmethod
    def date_action():
        return str(datetime.date.today())
    
    def timer_action():
        seconds = time.time()
        print("Time:", seconds)
        local_time = time.ctime(seconds)
        print("Local time:", local_time)
    
    def evaluate_math_expression(self, expr):
        try:
            # Replace words with mathematical symbols
            expr = expr.lower()
            expr = re.sub(r'(\bplus\b|\bmas\b)', '+', expr)
            expr = re.sub(r'(\bminus\b|\bmenos\b)', '-', expr)
            expr = re.sub(r'(\bmultiply\b|\btimes\b|\bpor\b)', '*', expr)
            expr = re.sub(r'(\bdivide\b|\bdividido por\b|\bsobre\b)', '/', expr)

            # Ensure proper spacing around operators
            expr = re.sub(r'(\d)([+\-*/])', r'\1 \2 ', expr)
            expr = re.sub(r'([+\-*/])(\d)', r' \1 \2', expr)

            # Evaluate the expression
            result = eval(expr)
            return result
        except Exception as e:
            return f"Error: {str(e)}"


# Running the AI

if __name__ == "__main__":
    ai = ChatBot(name="jarvis")
    t = True
    while t:
            ai.speech_to_text()
            if ai.wake_up(ai.text) is True:
                res = "Hello, I'm JARVIS, your personal assistant. How may I help you?"
                spanish = False

            if ai.text == "hola Jarvis":
                spanish = True
                res = "Hola, soy JARVIS, su asistente personal. Como le puedo ayudar?"

            elif any(i in ai.text for i in ["my name is"]):
                x = True
                spanish = False
                words = ai.text.split(" ")
                ind = words.index('is')
                name = words[ind + 1]
                with open ('name.pickle', 'wb') as F:
                    pickle.dump(name, F)
                with open('name.pickle', 'rb') as F:
                    loaded_name = pickle.load(F)
                res = f"Hello {loaded_name}, I look forward in helping you."
            #your name
            elif any(i in ai.text for i in ["Mi nombre es", "me llamo", "soy"]):
                x = True
                spanish = True
                words = ai.text.split(" ")
                ind = words.index('is')
                name = words[ind + 1]
                with open ('name.pickle', 'wb') as F:
                    pickle.dump(name, F)
                with open('name.pickle', 'rb') as F:
                    loaded_name = pickle.load(F)
                res = f"Hola {loaded_name}, espero que te pueda ayudar."

            elif any (i in ai.text for i in ["what is my name", "what's my name"]):
                x = True
                spanish = False
                with open('name.pickle', 'rb') as F:
                    loaded_name = pickle.load(F)
                res = f"Your name is {loaded_name}"

            elif any (i in ai.text for i in ["es Mi nombre", "es mi nombre"]):
                x = True
                spanish = True
                with open('name.pickle', 'rb') as F:
                    loaded_name = pickle.load(F)
                res = f"Tu nombre es {loaded_name}"
            #repeat phrase
            elif any (i in ai.text for i in ["repeat", "say"]):
                x = True
                words = ai.text.split("this")
                if len(words) == 2:
                    phrase = words[1].strip()
                res = f"{phrase}"
        # Use the internet
            # elif any(i in ai.text for i in []):
            #     # words = ai.text.split("up")
            #     # if len(words)==2:
            #     #     search_term = words[1].strip()
            #     search_term = ai.text 
            #     def get_results(search_term):
            #         url = "https://www.google.com"
            #         chrome_options = Options()
            #         chrome_options.add_experimental_option("detach", True)  
            #         browser = Chrome(options = chrome_options)
            #         browser.get(url)
            #         search_box = browser.find_element(By.CLASS_NAME, "gLFyf")
            #         search_box.send_keys(search_term)
            #         search_box.submit()
            #         try:
            #             links = browser.find_elements(By.XPATH, "//ol[@class='web_regular_results']//h3//a")
            #         except:
            #             links = browser.find_elements(By.XPATH, "//h3//a")
            #         results = []
            #         for link in links:
            #             href = link.get_attribute("href")
            #             print(href)
            #             results.append(href)
            #         return results
            #     get_results(search_term)
            #     res = "This is what I got"

        #     where you live 
            elif any(i in ai.text for i in ["I live" ]):
                x = True
                words = ai.text.split(" ")
                ind = words.index('in')
                city = words[ind + 1]
                with open ('city.pickle', 'wb') as F:
                    pickle.dump(city, F)
                with open('city.pickle', 'rb') as F:
                    loaded_city = pickle.load(F)
                res = "Good to know."
            elif any (i in ai.text for i in ["where am I from"]):
                x = True
                with open('city.pickle', 'rb') as F:
                    loaded_city = pickle.load(F)
                res = f"You live in {loaded_city}"
            #weather
            elif any( i in ai.text for i in ["temperature"]):
                x = True
                words = ai.text.split(" ")
                ind = words.index("in")
                loc = words[ind + 1]
                async def getweather(loc):
                # declare the client. the measuring unit used defaults to the metric system (celcius, km/h, etc.)
                    async with python_weather.Client(unit=python_weather.IMPERIAL) as client:
                    # fetch a weather forecast from a city
                        weather = await client.get(loc)
                        return weather.current.temperature
                temp = asyncio.run(getweather(loc))
                res = f'The temperature is {temp} degrees'
            #Show spectrograms of saved audio files
            elif any(i in ai.text for i in ["show my voice"]):
                x = True
                def show_voice():
                    audio_files= ['heyjarvis.mp3', 'name.mp3', 'repeat.mp3', 'time.mp3']

                    mfccs_list = []

                    for filename in audio_files:
                        audio_data, sample_rate = lb.load(filename)

                        mfccs = lb.feature.mfcc(y = audio_data, sr = sample_rate)

                        mfccs_list.append(mfccs)

                        # Plot waveform
                        plt.figure(figsize=(10, 4))
                        plt.subplot(2, len(audio_files), audio_files.index(filename) + 1)
                        plt.plot(np.arange(len(audio_data))/sample_rate, audio_data)
                        plt.title(f'Waveform - {filename}')
                        plt.xlabel('Time (s)')
                        plt.ylabel('Amplitude')
                        
                        # Visualize spectrogram
                        plt.subplot(2, len(audio_files), len(audio_files) + audio_files.index(filename) + 1)
                        stft_data = lb.stft(audio_data)
                        magnitude_spec = np.abs(stft_data)  # Take the absolute value to retain only magnitude
                        log_magnitude_spec = lb.amplitude_to_db(magnitude_spec, ref=np.max)
                        lb.display.specshow(log_magnitude_spec, sr=sample_rate, x_axis='time', y_axis='hz')
                        plt.colorbar(format='%+2.0f dB')
                        plt.title(f'Spectrogram - {filename}')

                    # Define your audio files
                    audio_files = ['heyjarvis.mp3', 'name.mp3', 'repeat.mp3', 'time.mp3']
                    plt.tight_layout()
                    plt.show()
                show_voice()

            #image
            
            #elif any(i in ai.text for i in ["generate", "image", "photo"]):
                #client =  OpenAI(api_key = api_key)
                #words = ai.text.split(" ")
                #ind = words.index("of")
                #req = words[ind + 1]
                #response = client.images.generate(
                #    size = "1024x1024",
                #    prompt = f"{req}",
                #    quality = "standard",
                #    n = 1,
                #)
                #image_url = response.data[0].url
                #res = f"Here is a photo of a {req}: {image_url}"

            ##action time
            elif any (i in ai.text for i in ["tiempo", "hora"] ):
                x = True
                spanish = True
                res =  ai.action_time()
            elif any (i in ai.text for i in ["time"] ):
                spanish = False
                res = ai.action_time()
            #date
            elif any( i in ai.text for i in ["fecha", "que dia es hoy"]):
                x = True
                spanish = True
                res = ai.date_action()
            elif any( i in ai.text for i in ["what is the date", "what day is it", "what's today"]):
                x = True
                spanish = False
                res = ai.date_action()
            ##respond politely
            elif any(i in ai.text for i in ["thank","thanks","I appreciate"]):
                x = True
                spanish = False
                res = np.random.choice(["You're welcome.","My pleasure.","Always happy to help.","I'm always here if you need me."])
            elif any(i in ai.text for i in ["gracias","aprecio"]):
                x = True
                spanish = True
                res = np.random.choice(["De nada.","Es mi placer.","Estoy aqui si me necesita."])
            elif any(i in ai.text for i in ["bye","goodbye","shut down"]):
                x = True
                spanish = False
                res = np.random.choice(["Have a good day.","Goodbye.","Come back if you need me"])
                t= False
            elif any(i in ai.text for i in ["adios","hasta luego","apagate"]):
                x = True
                spanish = True
                res = np.random.choice(["Ten un buen dia.","Adios.","Hasta luego."])
                t = False
            #do math
            elif any(i in ai.text for i in ["Calculate", "calculate", "what is", "what's"]):
                x = True
                # Extract the mathematical expression from the user's input
                expr_match = re.search(r'(\d+(\.\d+)?\s*[-+*/]\s*\d+(\.\d+)?)', ai.text)
                if expr_match:
                    expr = expr_match.group()
                    result = ai.evaluate_math_expression(expr)
                    res = f"The result is: {result}" if not isinstance(result, str) else result
                else:
                    res = "Sorry, I couldn't understand the mathematical expression."
            
            elif ai.text == "ERROR":
                if spanish == False:
                    res= np.random.choice(["Sorry, come again?", "I couldn't understand you.", "Could you repeat that?", "I am having trouble understanding.", "My apologies, I cannot understand"])
                if spanish == True:
                    res = np.random.choice(["Disculpe, no te puedo entender", "Me lo puede repetir por favor?", "No te puedo escuchar"])
            ai.text_to_speech(res)

            if x == False: 
                search_term = ai.text 
                def get_results(search_term):
                    url = "https://www.google.com"
                    chrome_options = Options()
                    chrome_options.add_experimental_option("detach", True)  
                    browser = Chrome(options = chrome_options)
                    browser.get(url)
                    search_box = browser.find_element(By.CLASS_NAME, "gLFyf")
                    search_box.send_keys(search_term)
                    search_box.submit()
                    try:
                        links = browser.find_elements(By.XPATH, "//ol[@class='web_regular_results']//h3//a")
                    except:
                        links = browser.find_elements(By.XPATH, "//h3//a")
                    results = []
                    for link in links:
                        href = link.get_attribute("href")
                        print(href)
                        results.append(href)
                    return results
                get_results(search_term)
                res = "This is what I got"

    if spanish == False:
        print("Program shutting down...")
    if spanish == True:
        print("Apagando el programa...")


