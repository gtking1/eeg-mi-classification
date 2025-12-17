from tkinter import *
from tkinter.ttk import *
import time

# Create the main window
master = Tk()
# master.geometry("300x200")
master.title("Main Window")
master.configure(bg='red')
master.attributes('-fullscreen',True)
color = 'red'
count = 0
def background_task(color):
    newColor = ''
    if color == 'red':
        master.configure(bg='green')
        newColor = 'green'
    elif color == 'green':
        master.configure(bg='red')
        newColor = 'red'
    master.after(3000, background_task(newColor))
    count += 1
    if count > 10:
        master.quit()
master.after(3000, background_task('green'))
master.mainloop()