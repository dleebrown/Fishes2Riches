from Tkinter import *
from PIL import ImageTk, Image
import os
import tkMessageBox

class GetInput(Tk):
    
    def __init__(self):
        
        Tk.__init__(self)
        self.workingDir = ""
        
        Label(self, text=displayText).pack()
        
        e = Entry(self)
        e.pack()
        
        e.focus_set()
        
        def callback():
            self.workingDir = e.get()
            self.destroy()
        
        
        b = Button(self, text = "Ok", width = 10, command=callback)
        b.pack()
                                        
            
class RectangleDrawer(Tk):
     
    def __init__(self):
        Tk.__init__(self)     
        self.x = self.y = 0 #intialize the click position
        self.boxDim = 100 #intialize the default size of the box
        self.currentFileIndex = 0
        self.path = fileList[self.currentFileIndex]
        
        self.variable = StringVar()
        self.variable.set("100") #the variables for the option menu, with default value
        
        self.boxDimSelector = OptionMenu(self, self.variable, "100", "150", "200", "250","300","350","400")
        self.boxDimSelector.pack() #creating the drop down menu
        
        
        self.canvas = Canvas(self, width=1280, height=720, cursor="gumby") #creating gumby
        
        self.canvas.bind("<ButtonPress-1>", self.on_lmouse_press) #drawing box on-click
        self.bind("<ButtonPress-1>", lambda event: self.canvas.focus_set()) #setting focus to canvas so key press is registered
        self.canvas.bind("s", self.on_s_press) #saving subimage on pressing s
        self.canvas.bind("1", self.previous_image) #move to previous image on pressing 1
        self.canvas.bind("2", self.next_image) #move to next image on pressing 2
        #self.canvas.bind("g", self.goto_image) #move to a chosen image on pressing g
        
       
        self.background= ImageTk.PhotoImage(file=self.path)
        self.canvas.create_image(640,360,image=self.background) #loading the image
        
        self.canvas.pack(side="top", fill="both", expand=True) #drawing the canvas
          
        

    def on_lmouse_press(self, event): #on left click, draw a box of the specified size centered at the click
        self.x = event.x
        self.y = event.y
        self.boxDim = int(self.variable.get()) #grab the drop down option, and cast to int (better way?)
        
        x0=self.x-self.boxDim/2
        x1=self.x+self.boxDim/2
        
        y0=self.y-self.boxDim/2
        y1=self.y+self.boxDim/2
        
        self.canvas.delete("box") #clean up the last box
        box=self.canvas.create_rectangle(x0,y0,x1,y1, outline='red',dash=(3,4)) #draw the box
        self.canvas.itemconfig(box,tags=("box")) #tag this box so it can be deleted
        
    def on_s_press(self, event):
        
        dst = self.subimage(self.path, self.x-self.boxDim/2, self.y-self.boxDim/2, self.x+self.boxDim/2, self.y+self.boxDim/2)
        imageName = savingDir+ "\\" + self.path[0:-4]+ "cutout.jpg"
        dst.save(imageName)
        print "Image saved."
        
    def previous_image(self,event):
        if self.currentFileIndex>0:
            self.currentFileIndex-=1
            self.canvas.delete("ALL")
            self.path = fileList[self.currentFileIndex]
            self.background= ImageTk.PhotoImage(file=self.path)
            self.canvas.create_image(640,360,image=self.background) #reloading the image
            self.canvas.pack(side="top", fill="both", expand=True)            
        else:
            tkMessageBox.showerror("Error","Already at the first image.")
            
    def next_image(self,event):
        if self.currentFileIndex<numFiles:
            self.currentFileIndex+=1
            self.canvas.delete("ALL")
            self.path = fileList[self.currentFileIndex]
            self.background= ImageTk.PhotoImage(file=self.path)
            self.canvas.create_image(640,360,image=self.background) #reloading the image
            self.canvas.pack(side="top", fill="both", expand=True) 
        else:
            tkMessageBox.showerror("Error","Already at the last image.")       
    
    #def goto_image(self,event):
        
        #imageNum = GetInput()
        #imageNum.mainloop()
        
        #self.currentFileIndex = int(imageNum.workingDir)
        #print fileList[self.currentFileIndex]
        #self.canvas.delete("ALL")
        #self.path = fileList[self.currentFileIndex]
        #self.background= ImageTk.PhotoImage(file=self.path)
        #self.canvas.create_image(640,360,image=self.background) #reloading the image
        #self.canvas.pack(side="top", fill="both", expand=True)        
        
        
    def subimage(self,path, l, t, r, b):
        dst = Image.open(path)
        dst = dst.crop((l, t, r, b))
        return dst    
                    
if __name__ == "__main__":
    
    displayText = "Enter the directory containing the pictures you want to crop."
    getWorkingDir = GetInput()
    getWorkingDir.mainloop()
    workingDir = getWorkingDir.workingDir
    
    displayText = "Enter the directory where you want the cutouts to be saved (it must exist)."
    getSavingDir = GetInput()
    getSavingDir.task = 1
    getSavingDir.mainloop()
    savingDir = getSavingDir.workingDir     
    
    displayText = "Jump to image (enter its numerical order in the directory, starting from 1):"
    
    os.chdir(workingDir)
    fileList = os.listdir(workingDir)
    numFiles = len(fileList)-1
    
    app = RectangleDrawer() 
    app.mainloop()