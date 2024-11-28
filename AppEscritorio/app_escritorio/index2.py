import tkinter as tk
from tkinter import ttk
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

#Modelo CNN
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



#url_imagen = ""
imagenes_seleccionadas = []
class Aplication:
    
    def __init__(self):
        self.raiz = tk.Tk() 
        self.raiz.title("Prototipo 1") #Cambiar el nombre de la ventana 
        self.raiz.geometry("1024x768") #Configurar tamaño
        self.raiz.resizable(width=0, height=0)
        #self.raiz.iconbitmap("./Imagenes/ant.ico") #Cambiar el icono
        self.imagen= tk.PhotoImage(file="./AppEscritorio/app_escritorio/fondo1.png")
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

        # Canvas para mostrar las imágenes
        #self.canvas = tk.Canvas(self.raiz, width=270, height=390)
        #self.canvas.place(x=60, y=145, width=280, height=390)
        #self.canvas.pack(pady=10)


        tk.Label(self.raiz, text="Generar histograma: ", bg="#F1F0F0").place(relx=0.704, rely=0.095,)
        self.Acciones = ttk.Combobox(self.raiz, state="disabled")
        self.Acciones['values']=['RGB','HSV','LAB']
        self.Acciones.place(relx=0.706, rely=0.120, relwidth=0.16)
        self.Acciones.bind("<<ComboboxSelected>>",  self.AccionElegida)
        self.Acciones.current(0)


        #Botones
        self.boton = tk.Button(text="Elegir imagen", bg="#73B731", fg="#ffffff",font=("Verdana", 9, "bold"), borderwidth = 0, command=self.seleccionar)
        self.boton.place(x=340, y=70, width=120)
        self.boton_area = tk.Button(self.raiz, text="ÁREA DAÑADA", width=2, height=2, bg="#73B731", fg="#ffffff",font=("Verdana", 6, "bold"), borderwidth = 0, command=self.area_dañada, state="disabled")
        self.boton_area.place(x= 704, y= 166, width=100, height=47)
        self.boton_clasific= tk.Button(self.raiz, text="CLASIFICACIÓN ENFERMEDAD", width=2, height=2, bg="#73B731", fg="#ffffff",font=("Verdana", 9, "bold"), borderwidth = 0, command=self.clasificacion_imagen, state="disabled")
        self.boton_clasific.place(x= 704, y= 242, width=260, height=47)
        self.boton_maduracion = tk.Button(self.raiz, text="NIVEL DE MADURACIÓN", width=2, height=2, bg="#73B731", fg="#ffffff",font=("Verdana", 9, "bold"), borderwidth = 0, state="disabled")
        self.boton_maduracion.place(x= 704, y= 318, width=260, height=47)
        self.boton_reset = tk.Button(self.raiz, text="REINICIAR", width=2, height=2, bg="#73B731", fg="white",font=("Verdana", 9, "bold"), borderwidth = 0, command=self.reset_images, state="disabled")
        self.boton_reset.place(x= 704, y= 393, width=260, height=47)

        # Botón "Generar Reporte" (después de "Elegir Imagen")
        self.boton_reporte = tk.Button(text="Generar Reporte",
            bg="#4A90E2",  # Color de fondo del botón
            fg="#ffffff",  # Color del texto
            font=("Verdana", 9, "bold"),
            borderwidth=0,
            command=self.generar_reporte
        )
        self.boton_reporte.place(x=480, y=70, width=120) 

        self.boton_anterior = tk.Button(
            self.raiz, text="Anterior", 
            bg="#73B731", fg="#ffffff", font=("Verdana", 9, "bold"), 
            borderwidth=0, 
            command=self.mostrar_anterior, 
            state="disabled"
        )
        self.boton_anterior.place(x=60, y=550, width=120)

        self.boton_siguiente = tk.Button(
            self.raiz, text="Siguiente", 
            bg="#73B731", fg="#ffffff", font=("Verdana", 9, "bold"), 
            borderwidth=0, 
            command=self.mostrar_siguiente, 
            state="disabled"
        )
        self.boton_siguiente.place(x=200, y=550, width=120)

        self.boton_anterior_2 = tk.Button(
            self.raiz, text="Anterior", 
            bg="#73B731", fg="#ffffff", font=("Verdana", 9, "bold"), 
            borderwidth=0, 
            command=self.mostrar_anterior_2, 
            state="disabled"
        )
        self.boton_anterior_2.place(x=370, y=550, width=120)

        self.boton_siguiente_2 = tk.Button(
            self.raiz, text="Siguiente", 
            bg="#73B731", fg="#ffffff", font=("Verdana", 9, "bold"), 
            borderwidth=0, 
            command=self.mostrar_siguiente_2, 
            state="disabled"
        )
        self.boton_siguiente_2.place(x=540, y=550, width=120)

        self.raiz.mainloop() 

        # Atributos para la galería
        self.imagenes = []  # Lista de imágenes cargadas
        self.imagenes_area_dañada = []
        self.indice_actual = 0  # Índice de la imagen actual
        self.indice_actual_2 = 0  # Índice de la imagen actual
        self.porcentajes_area_dañada = []

    def seleccionar(self):
            # Abrir cuadro de diálogo para seleccionar múltiples imágenes
        archivos = fd.askopenfilenames(
            initialdir="C:/Users/USER/OneDrive/Escritorio/AImages",
            title="Seleccionar Archivos",
            filetypes=(("Image files", "*.png;*.jpg;*.gif"), ("todos los archivos", "*.*"))
        )
        
        if archivos:  # Verificar si se seleccionaron archivos
            self.imagenes = []  # Lista para almacenar imágenes cargadas
            self.indice_actual = 0  # Reiniciar índice actual
            #Guardar rutas de imagenes
            imagenes_seleccionadas.extend(archivos)

            for archivo in archivos:
                try:
                    # Abrir y procesar la imagen
                    imagen = Image.open(archivo)
                    rotated_image = imagen.rotate(90, expand=True)
                    rotated_image.thumbnail((270, 390))  # Redimensionar al tamaño estándar
                    photo = ImageTk.PhotoImage(rotated_image)
                    self.imagenes.append(photo)  # Agregar imagen procesada a la lista
                except Exception as e:
                    print(f"Error al procesar la imagen {archivo}: {e}")

            if self.imagenes:
                # Mostrar la primera imagen de la lista
                self.image_label.configure(image=self.imagenes[self.indice_actual])

                # Habilitar botones de navegación y funcionalidades
                self.boton_area.configure(state="normal")
                self.boton_clasific.configure(state="normal")
                self.boton_maduracion.configure(state="normal")
                self.boton_reset.configure(state="normal")
                self.Acciones.configure(state="readonly")
                if len(self.imagenes) > 1:
                    self.boton_anterior.configure(state="normal")
                    self.boton_siguiente.configure(state="normal")
    
                    
    def AccionElegida(self, eventObject):
            print(imagenes_seleccionadas) 
            for url_img in imagenes_seleccionadas:
                img = cv2.imread(url_img)
                if eventObject.widget.get()=="RGB":
                    channels = ('Blue', 'Green', 'Red')
                    for i, color in enumerate(channels):
                        plt.hist(np.array(img)[:, :, i].ravel(), bins=256, color=color.lower(), alpha=0.5, label=f'{color}')

                if eventObject.widget.get()=="HSV":
                    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    channels = ('Hue', 'Saturation', 'Value')
                    colors = ['#5356FF', '#378CE7', '#67C6E3']
                    for i, color in enumerate(channels):
                        plt.hist(np.array(hsv_img)[:, :, i].ravel(), bins=256, color=colors[i], alpha=0.5, label=f'{color}')
                if eventObject.widget.get()=="LAB":
                    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                    channels = ('L', 'A', 'B')
                    colors = ['#6439FF', '#4F75FF', '#00CCDD']
                    for i, color in enumerate(channels):
                        plt.hist(np.array(lab_img)[:, :, i].ravel(), bins=256, color=colors[i], alpha=0.5, label=f'{color}')

                # Configuración del gráfico
                plt.title(f'-Histograma- \n{url_img.split('/')[-1]}', fontsize=12)
                plt.xlabel('Valor de píxel')
                plt.ylabel('Frecuencia')
                plt.legend(loc='upper right')
                plt.show()
    

    def area_dañada(self):
        self.imagenes_area_dañada = []  # Lista para almacenar imágenes cargadas
        self.indice_actual_2 = 0  # Reiniciar índice actual
        self.porcentajes_area_dañada = []
        for url_img in imagenes_seleccionadas:
            img = cv2.imread(url_img)
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

            # Convertir la imagen RGB a un objeto Image de PIL
            op_or_pil_image = Image.fromarray(op_or)

            # Detectar si la imagen es horizontal
            if op_or_pil_image.width > op_or_pil_image.height:
                # Rotar la imagen para que quede vertical
                op_or_pil_image = op_or_pil_image.rotate(90, expand=True)

            # Redimensionar la imagen para que ocupe todo el espacio disponible
            op_or_pil_image = op_or_pil_image.resize((270, 390), Image.Resampling.LANCZOS)

            # Convertir la imagen PIL a un objeto PhotoImage de Tkinter
            op_or_photo = ImageTk.PhotoImage(op_or_pil_image)

            #Guardar la imagen en la lista
            self.imagenes_area_dañada.append(op_or_photo)
            # Crear una etiqueta para mostrar la imagen
            image_label = tk.Label(self.image_label2, image=op_or_photo)
            image_label.photo = op_or_photo  # Mantener una referencia para evitar la recolección de basura
            image_label.place(x=0, y=0)

            self.text_label.config(text=" ")
            self.text_label.update()

            #text_label = tk.Label(self.text_label, text=f"Porcentaje área dañada = {areaDañada:.2f} %", font=("Arial", 14, "bold"))
            self.text_label.config(text=f"Porcentaje área dañada = {areaDañada:.2f} %", font=("Arial", 14, "bold"))
            self.porcentajes_area_dañada.append(f"Porcentaje área dañada = {areaDañada:.2f} %")
            self.text_label.place(x=30, y=600)
        
        # Mostrar la primera imagen de la lista
        self.image_label2.configure(image=self.imagenes_area_dañada[self.indice_actual_2])
        if len(self.imagenes_area_dañada) > 1:
                    self.boton_anterior_2.configure(state="normal")
                    self.boton_siguiente_2.configure(state="normal")


    def clasificacion_imagen(self):
        if imagenes_seleccionadas:  # Verificar si se seleccionaron archivos
            # Cargar el modelo previamente entrenado
            modelo = tf.keras.models.load_model('D:/Documentos/Protocolo/app_escritorio/modelo_entrenado_VGG16.keras')

            clases = ['Sano', 'Enfermo_Body_Rot', 'Enfermo_Stem_end_Rot']  # Define tus clases
            resultados = []  # Lista para guardar los resultados

            for url_imagen in imagenes_seleccionadas:
                # Preprocesar la imagen
                img = image.load_img(url_imagen, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                # Hacer la predicción
                prediccion = modelo.predict(img_array)
                indice_prediccion = np.argmax(prediccion[0])
                clase_predicha = clases[indice_prediccion]

                # Guardar resultado en la lista
                nombre_imagen = url_imagen.split('/')[-1]  # Obtener solo el nombre del archivo
                resultados.append(f"{nombre_imagen}: {clase_predicha}")

            # Crear una nueva ventana para mostrar todos los resultados
            ventana_resultados = tk.Toplevel(self.raiz)
            ventana_resultados.title("Resultados de Clasificación")
            ventana_resultados.geometry("500x400")

            # Mostrar el título
            tk.Label(ventana_resultados, text="Resultados de Clasificación", font=("Arial", 16, "bold")).pack(pady=10)

            # Mostrar los resultados en un cuadro de texto (permite scroll si hay muchos)
            cuadro_resultados = tk.Text(ventana_resultados, wrap=tk.WORD, font=("Arial", 12))
            cuadro_resultados.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Insertar los resultados en el cuadro de texto
            for resultado in resultados:
                cuadro_resultados.insert(tk.END, resultado + "\n")

            # Deshabilitar edición del cuadro de texto
            cuadro_resultados.config(state=tk.DISABLED)

            # Botón para cerrar la ventana
            tk




    def reset_images(self):
        # Clear image_label content
        self.image_label.config(image="")  # Empty string for image

        # Clear image_label2 content (assuming it holds an image)
        self.image_label2.config(image="")

        # Clear text_label content
        self.text_label.config(text="")  # Empty string for text

        self.imagenes_seleccionadas = []

    def mostrar_imagen(self):
        if self.imagenes:
            # Mostrar la imagen actual
            self.image_label.configure(image=self.imagenes[self.indice_actual])

    def mostrar_anterior(self):
        if self.imagenes:
            # Mover al índice anterior
            self.indice_actual = (self.indice_actual - 1) % len(self.imagenes)
            self.mostrar_imagen()

    def mostrar_siguiente(self):
        if self.imagenes:
            # Mover al índice siguiente
            self.indice_actual = (self.indice_actual + 1) % len(self.imagenes)
            self.mostrar_imagen()
    
    def mostrar_imagen_2(self):
        if self.imagenes_area_dañada:
            # Mostrar la imagen actual
            self.image_label2.configure(image=self.imagenes_area_dañada[self.indice_actual_2])
            self.text_label.config(text=self.porcentajes_area_dañada[self.indice_actual_2], font=("Arial", 14, "bold"))
            self.text_label.place(x=30, y=600)

    def mostrar_anterior_2(self):
        if self.imagenes_area_dañada:
            # Mover al índice anterior
            self.indice_actual_2 = (self.indice_actual_2 - 1) % len(self.imagenes_area_dañada)
            self.mostrar_imagen_2()

    def mostrar_siguiente_2(self):
        if self.imagenes_area_dañada:
            # Mover al índice siguiente
            self.indice_actual_2 = (self.indice_actual_2 + 1) % len(self.imagenes_area_dañada)
            self.mostrar_imagen_2()

    def generar_reporte(self):
        # Aquí puedes implementar la lógica para generar un reporte
        if hasattr(self, 'imagenes') and self.imagenes:
            with open("reporte_imagenes.txt", "w") as f:
                f.write(f"Se cargaron {len(self.imagenes)} imágenes.\n")
                for idx, img in enumerate(self.imagenes):
                    f.write(f"Imagen {idx + 1}: procesada correctamente.\n")
            print("Reporte generado: reporte_imagenes.txt")
        else:
            print("No hay imágenes cargadas para generar un reporte.")



aplication = Aplication()

