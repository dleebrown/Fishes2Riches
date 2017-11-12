from tkinter import *
from PIL import ImageTk, Image
import os
from tkinter import messagebox

def return_images(directory):
    """Returns all .jpg images in a specified directory
    you know, in UNIX you can just use glob and it's way easier
    Returns the images with their full path names
    """
    allfiles = os.listdir(directory)
    image_list = [im for im in allfiles if '.jpg' in str(im)]
    image_list = [directory + '/'+im for im in image_list]
    return image_list

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
            self.quit()
        
        
        b = Button(self, text = "Ok", width = 10, command=callback)
        b.pack()
                                        
            
class RectangleDrawer(Tk):
     
    def __init__(self):
        Tk.__init__(self)     
        self.x = self.y = 0 #intialize the click position
        self.boxDim = 100 #intialize the default size of the box
        self.fileNameIndex = 0
        self.currentFileIndex = 0
        self.path = fileList[self.currentFileIndex]
        self.cutoutData=[]
        
        self.variable = StringVar()
        self.variable.set("100") #the variables for the option menu, with default value
        
        self.boxDimSelector = OptionMenu(self, self.variable, "100", "150", "200", "250","300","350","400")
        self.boxDimSelector.pack() #creating the drop down menu
        
        self.background= ImageTk.PhotoImage(file=self.path)
        self.width = self.background.width()
        self.height = self.background.height()              
        self.canvas = Canvas(self, width=self.width, height=self.height, cursor="gumby") #creating gumby
        
        self.canvas.bind("<ButtonPress-1>", self.on_lmouse_press) #drawing box on-click
        self.bind("<ButtonPress-1>", lambda event: self.canvas.focus_set()) #setting focus to canvas so key press is registered
        self.canvas.bind("s", self.on_s_press) #saving subimage on pressing s
        self.canvas.bind("BackSpace", self.previous_image) #move to previous image on pressing 1
        self.canvas.bind("<ButtonPress-3>", self.next_image) #move to next image on pressing right click
        self.canvas.bind("g", self.goto_image) #move to a chosen image on pressing g
        self.canvas.bind("1", lambda event, newDims = 100: self.change_boxDim(newDims)) #could do this with anonymous functions, but we've come too far
        self.canvas.bind("q", lambda event, newDims = 150: self.change_boxDim(newDims))
        self.canvas.bind("2", lambda event, newDims = 200: self.change_boxDim(newDims))
        self.canvas.bind("w", lambda event, newDims = 250: self.change_boxDim(newDims))
        self.canvas.bind("3", lambda event, newDims = 300: self.change_boxDim(newDims))
        self.canvas.bind("e", lambda event, newDims = 350: self.change_boxDim(newDims))
        self.canvas.bind("4", lambda event, newDims = 400: self.change_boxDim(newDims))
       
        
        self.canvas.create_image(self.width/2,self.height/2,image=self.background) #loading the image
        
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
        
    ##switch s functions to change between saving cutouts and making the text file    
    #def on_s_press(self, event):
        
        #self.fileNameIndex += 1
        #dst = self.subimage(self.path, self.x-self.boxDim/2, self.y-self.boxDim/2, self.x+self.boxDim/2, self.y+self.boxDim/2)
        #imageName = savingDir+ "\\" + self.path[0:-4]+ "cutout" +str(self.fileNameIndex)+".jpg"
        #dst.save(imageName)
        #print("Image saved.")
        
    def on_s_press(self, event):
        #save the current filename, the center of the cutout (x,y), the size of the cutout, and the class
        info = [fileList[self.currentFileIndex][-9:-4], [self.x,self.y], self.boxDim,workingDir.split('/')[-1]]
        self.cutoutData.append(info)
       
        
        
    def previous_image(self,event):
        if self.currentFileIndex>0:
            self.fileNameIndex = 0
            self.currentFileIndex-=1
            self.canvas.delete("ALL")
            self.path = fileList[self.currentFileIndex]
            self.background= ImageTk.PhotoImage(file=self.path)
            self.width = self.background.width()
            self.height = self.background.height()         
            self.canvas.create_image(self.width/2,self.height/2,image=self.background) #loading the image
            self.canvas.pack(side="top", fill="both", expand=True) #drawing the canvas            
        else:
            messagebox.showerror("Error","Already at the first image.")
            
    def next_image(self,event):
        if self.currentFileIndex<numFiles:
            self.fileNameIndex = 0
            self.currentFileIndex+=1
            self.canvas.delete("ALL")
            self.path = fileList[self.currentFileIndex]
            self.background= ImageTk.PhotoImage(file=self.path)
            self.width = self.background.width()
            self.height = self.background.height()         
            self.canvas.create_image(self.width/2,self.height/2,image=self.background) #loading the image
            self.canvas.pack(side="top", fill="both", expand=True) #drawing the canvas
        else:
            messagebox.showerror("Error","Already at the last image.")       
    
    def goto_image(self,event):
        
        imageNum = GetInput()
        imageNum.mainloop()
        imageNum.destroy()
        
        self.currentFileIndex = int(imageNum.workingDir)-1
        self.canvas.delete("ALL")
        self.path = fileList[self.currentFileIndex]
        self.background= ImageTk.PhotoImage(file=self.path)
        self.width = self.background.width()
        self.height = self.background.height()         
        self.canvas.create_image(self.width/2,self.height/2,image=self.background) #loading the image
        self.canvas.pack(side="top", fill="both", expand=True) #drawing the canvas
        
    def change_boxDim(self,newDim):
        #hotkeyzzzz
        self.variable.set(str(newDim))
        
    def subimage(self,path, l, t, r, b):
        dst = Image.open(path)
        dst = dst.crop((l, t, r, b))
        return dst   
    

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit and write data to file?"):
        with open('Cutout Data.txt','a') as file:
            for line in app.cutoutData:
                file.write(str(line)+"\n")
    app.destroy() 

                    
if __name__ == "__main__":
    
    
    displayText = "Enter the directory containing the pictures you want to crop."
    getWorkingDir = GetInput()
    getWorkingDir.mainloop()
    workingDir = getWorkingDir.workingDir
    savingDir = workingDir
    getWorkingDir.destroy()
    
    #displayText = "Enter the directory where you want the text file to be saved."
    #getSavingDir = GetInput()
    #getSavingDir.task = 1
    #getSavingDir.mainloop()
    #savingDir = getSavingDir.workingDir  
    #getSavingDir.destroy()
    
    displayText = "Jump to image (enter its numerical order in the directory, starting from 1):"
    
    os.chdir(workingDir)
    fileList = return_images(workingDir)
    numFiles = len(fileList)-1
    
    os.chdir(savingDir)
    
    
    app = RectangleDrawer() 
    app.protocol("WM_DELETE_WINDOW", on_closing)
    app.mainloop()