# for language model
import speech_recognition as sr
import os
# for data
import os
import datetime
import numpy as np
from gtts import gTTS
import requests
import re 


# Building the AI
class ChatBot():
    def __init__(self, name):
        print("...", name, "...")
        self.name = name
    def speech_to_text(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as mic:
            print("Listening/Escuchando...")
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
        #speaker = gTTS(text=text, lang="en", slow=False)
        #speaker.save("res.mp3")
        #statbuf = os.stat("res.mp3")
        #mbytes = statbuf.st_size / 1024
        #duration = mbytes / 200
        #os.system('start res.mp3')  #if you are using mac->afplay or else for windows->start
        #os.system("close res.mp3")
        #time.sleep(int(50*duration))
        #os.remove("res.mp3")
    def wake_up(self, text):
        return True if self.name in text.lower() else False
    @staticmethod
    def action_time():
        return datetime.datetime.now().time().strftime('%I:%M %p')
    @staticmethod
    def date_action():
        return datetime.date.today()
    def evaluate_math_expression(self, expr):
        try:
            # Replace words with mathematical symbols
            expr = expr.lower()
            expr = re.sub(r'(\bplus\b|\bmas\b)', '+', expr)
            expr = re.sub(r'(\bminus\b|\bmenos\b)', '-', expr)
            expr = re.sub(r'(\bmultiply\b|\btimes\b|\bpor\b)', '*', expr)
            expr = re.sub(r'(\bdivide\b|\bdivididopor\b|\bsobre\b)', '/', expr)

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
        ## action time
        elif any (i in ai.text for i in ["tiempo", "hora"] ):
            spanish = True
            res = ai.action_time()
        elif any (i in ai.text for i in ["time"] ):
            spanish = False
            res = ai.action_time()
        elif any( i in ai.text for i in ["fecha", "que dia es hoy"]):
            spanish = True
            res = ai.date_action()
        elif any( i in ai.text for i in ["date", "what day is it", "what's today"]):
            spanish = False
            res = ai.date_action()
        ## respond politely
        elif any(i in ai.text for i in ["thank","thanks",]):
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
        elif any(i in ai.text for i in ["Calculate", "calculate"]):
            # Extract the mathematical expression from the user's input
            expr_match = re.search(r'(\d+(\.\d+)?\s*[-+*/]\s*\d+(\.\d+)?)', ai.text)
            if expr_match:
                expr = expr_match.group()
                result = ai.evaluate_math_expression(expr)
                res = f"The result is: {result}" if not isinstance(result, str) else result
            else:
                res = "Sorry, I couldn't understand the mathematical expression."
        elif any(i in ai.text for i in ["computa"]):
            spanish = True
            # Extract the mathematical expression from the user's input
            expr_match = re.search(r'(\d+(\.\d+)?\s*[-+*/]\s*\d+(\.\d+)?)', ai.text)
            if expr_match:
                expr = expr_match.group()
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
    print("Program shutting down...")