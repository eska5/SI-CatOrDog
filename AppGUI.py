from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import tensorflow

modelCDO = tensorflow.keras.models.load_model("CDO.model")
modelCH = tensorflow.keras.models.load_model("BinaryCatHiro.model")
modelDC = tensorflow.keras.models.load_model("BinaryDogCezar.model")

window = Tk()
window.title("Projekt SI, Wykrywanie Obrazow")
lbl = Label(window, text="Wybierz zdjęcie: ", font=("Arial Bold", 50))
font = ("Arial Bold", 50)
lbl.grid(column=0, row=0)


def loadFile(filepath, size):
    imageR = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new = cv2.resize(imageR, (size, size))
    return new.reshape(-1, size, size, 1)


def toText(predictCDO, predictCH, predictDC):
    whatIsItPRC = 0
    whatIsIt = -1
    napis = []
    for result in predictCDO:
        if whatIsItPRC < result[0]:
            whatIsItPRC = result[0]
            whatIsIt = "Kot"
        if whatIsItPRC < result[1]:
            whatIsItPRC = result[1]
            whatIsIt = "Pies"
        if whatIsItPRC < result[2]:
            whatIsItPRC = result[2]
            whatIsIt = "Inna opcja"
    napis.append('To jest: ' + whatIsIt + '!\n\n' + 'Kot: ' + str(round((result[0]*100), 2)) + '%\n' + 'Pies: ' + str(
        round((result[1]*100), 2)) + '%\n' + ' Inne: ' + str(round((result[2]*100), 2)) + '%')
    if whatIsIt == 'Kot':
        for result in predictCH:
            if result == 0.0:
                whatIsIt = 'Hiro'
            else:
                whatIsIt = 'Inny Kot'
        napis.append('To jest: ' + whatIsIt + '!\n')
    elif whatIsIt == 'Pies':
        for result in predictDC:
            if result == 0.0:
                whatIsIt = 'Cezar'
            else:
                whatIsIt = 'Inny Pies'
        napis.append('To jest: ' + whatIsIt + '!\n')

    return napis


def clicked():
    global img
    global file_path
    global panel
    global text
    global textHC
    global textCDO
    text = Label(window)

    textHC.destroy()
    textCDO.destroy()

    file_path = filedialog.askopenfilename()

    predictCDO = modelCDO.predict([loadFile(file_path, 200)])
    predictCH = modelCH.predict([loadFile(file_path, 200)])
    predictDC = modelDC.predict([loadFile(file_path, 100)])

    img = Image.open(file_path)
    img = img.resize((500, 500), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(window, image=img)
    panel.grid(column=0, row=2)

    display_text = toText(predictCDO, predictCH, predictDC)
    textCDO = Label(window, text=display_text[0], font=("Arial", 24))
    textCDO.place(x=600, y=250)
    textHC = Label(window, text=display_text[1], font=("Arial", 24))
    textHC.place(x=900, y=250)


textHC = Label(window, text="", font=("Arial", 24))
textCDO = Label(window, text="", font=("Arial", 24))
textHC.place(x=900, y=250)
textCDO.place(x=600, y=250)
btn = Button(window, text="Wybierz", bg="yellow",
             font=("Arial Bold", 30), command=clicked)
btn.grid(column=1, row=0)
window.geometry("1920x720")
window.resizable(width=True, height=True)
window.mainloop()
