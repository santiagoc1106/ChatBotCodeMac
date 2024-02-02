# for language model
import speech_recognition as sr
import os
# for data
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


spanish = None
name = None

# Building the AI
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
            print("me --> ", self.text)
        except:
            print("me -->  ERROR")

    @staticmethod
    def text_to_speech(text):
        print("JARVIS--> ", text)
        if spanish == False:
            speaker = gTTS(text=text, lang="en", slow=False)
            speaker.save("res.mp3")
            statbuf = os.stat("res.mp3")
            mbytes = statbuf.st_size / 1024
            duration = mbytes / 200
            os.system('afplay res.mp3')  #if you are using mac->afplay or else for windows->start
            #os.system("close res.mp3")
            time.sleep(int(10*duration))
            os.remove("res.mp3")

        if spanish == True:
            speaker = gTTS(text=text, lang="es", slow=False)
            speaker.save("res.mp3")
            statbuf = os.stat("res.mp3")
            mbytes = statbuf.st_size / 1024
            duration = mbytes / 200
            os.system('afplay res.mp3')  #if you are using mac->afplay or else for windows->start
            #os.system("close res.mp3")
            time.sleep(int(10*duration))
            os.remove("res.mp3")

    def wake_up(self, text):
        return True if self.name in text.lower() else False
    
    @staticmethod
    def action_time():
        return datetime.datetime.now().time().strftime('%I:%M %p')
    
    @staticmethod
    def date_action():
        return datetime.date.today()
    
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
            words = ai.text.split(" ")
            ind = words.index('is')
            name = words[ind + 1]
            with open ('name.pickle', 'wb') as F:
                pickle.dump(name, F)
            with open('name.pickle', 'rb') as F:
                loaded_name = pickle.load(F)
            res = f"Hola {loaded_name}, espero que te pueda ayudar."

        elif any (i in ai.text for i in ["what is my name", "what's my name"]):
            with open('name.pickle', 'rb') as F:
                loaded_name = pickle.load(F)
            res = f"Your name is {loaded_name}"

        elif any (i in ai.text for i in ["es Mi nombre", "es mi nombre"]):
            res = f"Tu nombre es {loaded_name}"
        #repeat
        elif any (i in ai.text for i in ["repeat", "say"]):
            words = ai.text.split("this")
            if len(words) == 2:
                phrase = words[1].strip()
            res = f"{phrase}"
        #where you live 
        elif any(i in ai.text for i in ["I live" ]):
            words = ai.text.split(" ")
            ind = words.index('in')
            city = words[ind + 1]
            with open ('city.pickle', 'wb') as F:
                pickle.dump(city, F)
            with open('city.pickle', 'rb') as F:
                loaded_city = pickle.load(F)
            res = "Good to know."
        elif any (i in ai.text for i in ["where am I from"]):
            with open('city.pickle', 'rb') as F:
                loaded_city = pickle.load(F)
            res = f"You live in {loaded_city}"
        #weather
        elif any( i in ai.text for i in ["temperature"]):
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
        ## action time
        elif any (i in ai.text for i in ["tiempo", "hora"] ):
            spanish = True
            res =  ai.action_time()
        elif any (i in ai.text for i in ["time"] ):
            spanish = False
            res = ai.action_time()
        #date
        elif any( i in ai.text for i in ["fecha", "que dia es hoy"]):
            spanish = True
            res = ai.date_action()
        elif any( i in ai.text for i in ["date", "what day is it", "what's today"]):
            spanish = False
            res = ai.date_action()
        ## respond politely
        elif any(i in ai.text for i in ["thank","thanks","I appreciate"]):
            spanish = False
            res = np.random.choice(["You're welcome.","My pleasure.","Always happy to help.","I'm always here if you need me."])
        elif any(i in ai.text for i in ["gracias","aprecio"]):
            spanish = True
            res = np.random.choice(["De nada.","Es mi placer.","Estoy aqui si me necesita."])
        elif any(i in ai.text for i in ["bye","goodbye","shut down"]):
            spanish = False
            res = np.random.choice(["Have a good day.","Goodbye.","Come back if you need me"])
            t= False
        elif any(i in ai.text for i in ["adios","hasta luego","apagate"]):
            spanish = True
            res = np.random.choice(["Ten un buen dia.","Adios.","Hasta luego."])
            t = False
        #do math
        elif any(i in ai.text for i in ["Calculate", "calculate", "what is", "what's"]):
            # Extract the mathematical expression from the user's input
            expr_match = re.search(r'(\d+(\.\d+)?\s*[-+*/]\s*\d+(\.\d+)?)', ai.text)
            if expr_match:
                expr = expr_match.group()
                result = ai.evaluate_math_expression(expr)
                res = f"The result is: {result}" if not isinstance(result, str) else result
            else:
                res = "Sorry, I couldn't understand the mathematical expression."
        elif any(i in ai.text for i in ["computa", "cula"]):
            spanish = True
            # Extract the mathematical expression from the user's input
            expr_match = re.search(r'(\d+(\.\d+)?\s*[-+*/]\s*\d+(\.\d+)?)', ai.text)
            span_num = ("Uno", "Dos", "Tres", "Cuatro", "Cinco", "Seis", "Siete", "Ocho", "Nueve", "Diez")
            span_to_num = {"Uno": 1, "Dos": 2, "Tres": 3, "Cuatro": 4, "Cinco":5, "Seis":6, "Siete":7, "Ocho":8, "Nueve":9, "Diez":10}
            numbers = [span_to_num[word] for word in span_num]
            span_add = [("mas")]
            span_mult = [("por")]
            span_sub = [("menos")]
            span_div = [("dividido por")]

            if expr_match:
                expr = expr_match.group() in ai.text.lower
                result = ai.evaluate_math_expression(expr)
                res = f"El resultado es: {result}" if not isinstance(result, str) else result
            else:
                res = "No entendi la expresion matematica."
        
        elif ai.text == "ERROR":
            if spanish == False:
                res= np.random.choice(["Sorry, come again?", "I couldn't understand you.", "Could you repeat that?", "I am having trouble understanding.", "My apologies, I cannot understand"])
            if spanish == True:
                res = np.random.choice(["Disculpe, no te puedo entender", "Me lo puede repetir por favor?", "No te puedo escuchar"])
       
        ai.text_to_speech(res)
    if spanish == False:
        print("Program shutting down...")
    if spanish ==True:
        print("Apagando el programa...")