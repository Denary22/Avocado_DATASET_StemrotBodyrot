import tkinter as tk
from tkinter import filedialog  as fd #Ventanas de dialogo
from PIL import Image
from PIL import ImageTk
import cv2
import numpy as np

#Bibliotecas aplicación
import cv2
import math
import numpy as np
from rembg import remove
from PIL import Image
import matplotlib.pyplot as plt

url_imagen = ""
class Aplication:
    
    def __init__(self):
        self.raiz = tk.Tk() 
        self.raiz.title("Prototipo 1") #Cambiar el nombre de la ventana 
        self.raiz.geometry("1024x768") #Configurar tamaño
        self.raiz.resizable(width=0, height=0)
        #self.raiz.iconbitmap("./Imagenes/ant.ico") #Cambiar el icono
        self.imagen= tk.PhotoImage(file="./fondo1.png")
        tk.Label(self.raiz, image=self.imagen, bd=0).pack()

        #Labels
        self.image_label = tk.Label(self.raiz, bg="#F1F0F0")
        self.image_label.place(x=60, y=145, width=280, height=390)

        #Labels
        self.image_label2 = tk.Label(self.raiz, bg="#F1F0F0")
        self.image_label2.place(x=375, y=145, width=280, height=390)

        #Labels
        self.text_label = tk.Label(self.raiz, bg="#F1F0F0")
        self.text_label.place(x=30, y=600, width=730, height=150)


        #Botones
        self.boton = tk.Button(text="Elegir imagen", bg="#73B731", fg="#ffffff",font=("Verdana", 9, "bold"), borderwidth = 0, command=self.seleccionar)
        self.boton.place(x=340, y=70, width=120)
        self.boton_area = tk.Button(self.raiz, text="ÁREA DAÑADA", width=2, height=2, bg="#73B731", fg="#ffffff",font=("Verdana", 9, "bold"), borderwidth = 0, command=self.area_dañada, state="disabled")
        self.boton_area.place(x= 704, y= 166, width=260, height=47)
        self.boton_clasific= tk.Button(self.raiz, text="CLASIFICACIÓN ENFERMEDAD", width=2, height=2, bg="#73B731", fg="#ffffff",font=("Verdana", 9, "bold"), borderwidth = 0, state="disabled")
        self.boton_clasific.place(x= 704, y= 242, width=260, height=47)
        self.boton_maduracion = tk.Button(self.raiz, text="NIVEL DE MADURACIÓN", width=2, height=2, bg="#73B731", fg="#ffffff",font=("Verdana", 9, "bold"), borderwidth = 0, state="disabled")
        self.boton_maduracion.place(x= 704, y= 318, width=260, height=47)
        self.boton_reset = tk.Button(self.raiz, text="REINICIAR", width=2, height=2, bg="#73B731", fg="white",font=("Verdana", 9, "bold"), borderwidth = 0, command=self.reset_images, state="disabled")
        self.boton_reset.place(x= 704, y= 393, width=260, height=47)

        self.raiz.mainloop() 

    def seleccionar(self):
            global url_imagen
            print(url_imagen)
            nomArchivo = fd.askopenfilename(initialdir= "C:/Users/USER/OneDrive/Escritorio/AImages", title= "Seleccionar Archivo", filetypes= (("Image files", "*.png; *.jpg; *.gif"),("todos los archivos", "*.*")))
            url_imagen = nomArchivo
            print(url_imagen)
            if nomArchivo!='':
                self.image = Image.open(nomArchivo)
                rotated_image = self.image.rotate(90, expand=True)
                # Redimensionar la imagen PIL al tamaño estándar manteniendo la relación de aspecto
                rotated_image.thumbnail((270, 390))
                # Convertir la imagen PIL a un objeto PhotoImage de Tkinter
                self.photo = ImageTk.PhotoImage(rotated_image)
                self.image_label.configure(image=self.photo)
            
                # Habilitar botones
                self.boton_area.configure(state="normal")
                self.boton_clasific.configure(state="normal")
                self.boton_maduracion.configure(state="normal")
                self.boton_reset.configure(state="normal")


    def area_dañada(self):
        print(url_imagen)
        #img = Image.open(file_path)
        img = cv2.imread(url_imagen)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #Removemos el fondo
        img_remove = remove(img)

        #Filtro Gaussiano
        img_gaussiana = cv2.GaussianBlur(img_remove, (15,15), 0)

        #Escala de grises
        img_gris = cv2.cvtColor(img_gaussiana, cv2.COLOR_BGR2GRAY)

        #Umbralización area del fruto
        ret, th = cv2.threshold(img_gris, 1, 255, cv2.THRESH_BINARY)

        #Contar pixeles blancos del área del fruto
        pixeles_blancos = np.where(th == 255, 1, 0)
        numero_pixeles_blancos = pixeles_blancos.sum()

        img_height, img_width, channels = img.shape
        total_pixels = img_height * img_width
        print("Total de pixeles = ",total_pixels)
        print("Total de pixeles blancos = ", numero_pixeles_blancos)

        #Dibujar contorno
        #Algoritmo de Canny
        canny_img_ext = cv2.Canny(th, 5, 15)
        img_contCanny = th.copy()
        cont, ret = cv2.findContours(canny_img_ext, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_contCanny, cont, -1, (0,0,0), 200)

        #Operación OR
        op_or_ext = cv2.bitwise_and(img_gaussiana , img_gaussiana, mask=img_contCanny)

        #Segmetación por color
        # Se seleccionan los rangos de HSV
        #stem end rot
        rango_min = np.array([18, 0, 0], np.uint8)
        rango_max = np.array([23, 255, 255], np.uint8)

        #body rot
        #rango_min = np.array([18, 0, 0], np.uint8)
        #rango_max = np.array([25, 255, 255], np.uint8)

        # Lectura de la imagen y conversión a HSV
        imagen_HSV = cv2.cvtColor(op_or_ext, cv2.COLOR_RGB2HSV)

        # Creación de la mascara de color
        mascara = cv2.inRange(imagen_HSV, rango_min, rango_max)
        segmentada_color = cv2.bitwise_and(img, img, mask=mascara)

        #Umbralización el area dañanda
        img_gris_dañada = cv2.cvtColor(segmentada_color, cv2.COLOR_BGR2GRAY)
        ret, areAfectada = cv2.threshold(img_gris_dañada, 1, 255, cv2.THRESH_BINARY)

        print(areAfectada.shape)
        #Calculó del porcentaje del área dañada

        #Contar pixeles blancos del área dañada
        pixeles_blancos_dañados = np.where(areAfectada == 255, 1, 0)
        numero_pixeles_blancos_dañados = pixeles_blancos_dañados.sum()

        areaDañada = ((numero_pixeles_blancos_dañados*100)/numero_pixeles_blancos)
        print("Total de pixeles dañados = ", numero_pixeles_blancos_dañados)


        areAfectada_3canales = cv2.cvtColor(areAfectada, cv2.COLOR_GRAY2RGB)
        #Operación OR
        op_or = cv2.bitwise_or(img, areAfectada_3canales)

        #Colocar en el cuadro
        standard_width = 310
        standard_height = 390

        # Convertir la imagen RGB a un objeto Image de PIL
        op_or_pil_image = Image.fromarray(op_or)

        # Redimensionar la imagen PIL al tamaño estándar manteniendo la relación de aspecto
        op_or_pil_image.thumbnail((standard_width, standard_height))

        # Convertir la imagen PIL a un objeto PhotoImage de Tkinter
        op_or_photo = ImageTk.PhotoImage(op_or_pil_image)

        # Crear una etiqueta para mostrar la imagen
        image_label = tk.Label(self.image_label2, image=op_or_photo)
        image_label.photo = op_or_photo  # Mantener una referencia para evitar la recolección de basura
        image_label.place(x=0, y=0)

        text_label = tk.Label(self.text_label, text=f"Porcentaje área dañada = {areaDañada:.2f} %", font=("Arial", 14, "bold"))
        text_label.place(x=0, y=0)


    def reset_images(self):
        # Clear image_label content
        self.image_label.config(image="")  # Empty string for image

        # Clear image_label2 content (assuming it holds an image)
        self.image_label2.config(image="")

        # Clear text_label content
        self.text_label.config(text="")  # Empty string for text


aplication = Aplication()