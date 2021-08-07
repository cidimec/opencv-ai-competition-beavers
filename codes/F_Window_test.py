import tkinter
from tkinter import ttk
import sys

Color_Background='dark slate gray'
Font_Color='snow'
ventana = tkinter.Tk()
ventana.configure(background=Color_Background)
ventana.title("Configuración Inicial")

def windows_events(event):
    print(Combo_modo.get(),
          Combo_ecuacion.get(),
          Entry_DimX.get(),
          Entry_DimZ.get(),
          Entry_PC.get(),
          event)
    if Combo_ecuacion.get()=="Wells-Riley":
        Label_wells_q.grid(row=6, column=1, columnspan=2, padx=4, pady=4)
        Label_wells_p.grid(row=7, column=1, columnspan=2, padx=4, pady=4)
        Label_wells_Q.grid(row=8, column=1, columnspan=2, padx=4, pady=4)
    else:
        Label_wells_q.grid_forget()
        Label_wells_p.grid_forget()
        Label_wells_Q.grid_forget()
    if event=='Cerrar':
        sys.exit()

##----------Labels
Label_Modo = tkinter.Label(ventana, text = "Modo", width = 10, height = 2, fg= Font_Color, bg= Color_Background)
Label_Dimension = tkinter.Label(ventana, text = "Dimension de la habitación (X,Z)", width = 30, height = 2, fg= Font_Color, bg= Color_Background)
Label_Infectados = tkinter.Label(ventana, text = "Infectados (%)", width = 10, height = 2, fg= Font_Color, bg= Color_Background)
Label_Ecuacion = tkinter.Label(ventana, text = "Ecuación", width = 10, height = 2, fg= Font_Color, bg= Color_Background)
Label_Parámetros = tkinter.Label(ventana, text = "Parámetros", width = 40, height = 2, fg= 'snow', bg= 'chocolate2')
Label_wells_q = tkinter.Label(ventana, text = "q", width = 10, height = 2, fg= Font_Color, bg= Color_Background)
Label_wells_p = tkinter.Label(ventana, text = "p", width = 10, height = 2, fg= Font_Color, bg= Color_Background)
Label_wells_Q = tkinter.Label(ventana, text = "Q", width = 10, height = 2, fg= Font_Color, bg= Color_Background)
Label_BLANK = tkinter.Label(ventana, text = "  ", width = 10, height = 2, fg= Font_Color, bg= Color_Background)
##----------Options
Options1 = [
    "Prueba",
    "Aplicativo"
]
Combo_modo = ttk.Combobox(ventana, value=Options1)
Combo_modo.current(0)
Combo_modo.bind("<<ComboboxSelected>>",windows_events)

Options2 = [
    "Wells-Riley",
    "Condicionantes"
]
Combo_ecuacion = ttk.Combobox(ventana, value=Options2)
Combo_ecuacion.current(0)
Combo_ecuacion.bind("<<ComboboxSelected>>", windows_events)

##----------Entrys
Entry_DimX = tkinter.Entry(ventana)
Entry_DimZ = tkinter.Entry(ventana)
Entry_PC = tkinter.Entry(ventana)
##----------Buttons
Button_iniciar = tkinter.Button(ventana, text= "Iniciar", command=lambda: windows_events("Iniciar"))
Button_cerrar = tkinter.Button(ventana, text= "Cerrar", command=lambda: windows_events("Cerrar"))
#Orden en matriz
Label_Modo.grid(row=0,column=1, columnspan=2, padx=4, pady=4)
Label_Dimension.grid(row=1,column=1, columnspan=2, padx=4, pady=4)
Label_Infectados.grid(row=2,column=1, columnspan=2, padx=4, pady=4)
Label_Ecuacion.grid(row=4,column=1, columnspan=2, padx=4, pady=4)
Label_Parámetros.grid(row=5,column=1, columnspan=4, padx=4, pady=4)

Entry_DimX.grid(row=1,column=3, columnspan=1, padx=4, pady=4)
Entry_DimZ.grid(row=1,column=4, columnspan=1, padx=4, pady=4)
Entry_PC.grid(row=2,column=3, columnspan=1, padx=4, pady=4)

Combo_modo.grid(row=0,column=3, columnspan=1, padx=4, pady=4)
Combo_ecuacion.grid(row=4,column=3, columnspan=1, padx=4, pady=4)

Button_iniciar.grid(row=20,column=5, columnspan=1, padx=4, pady=4)
Button_cerrar.grid(row=20,column=0, columnspan=1, padx=4, pady=4)

#loop
ventana.mainloop()