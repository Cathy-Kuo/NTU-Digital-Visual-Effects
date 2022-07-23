import cv2
import numpy as np
from argparse import ArgumentParser, Namespace
from tkinter import *
from PIL import Image, ImageTk, ImageDraw
from MVCClone import MVC_Clone
import warnings
warnings.filterwarnings('ignore')

def MVC_Clone_Dummy(src, tar, mask, pos_x, pos_y):
    result = tar.copy()
    h, w, _ = src.shape
    result[pos_y:pos_y+h,pos_x:pos_x+w] = src
    return result

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--source_img",
        type=str,
        default="data/source.jpg",
        help="Path to the source image.",
    )
    parser.add_argument(
        "--target_img",
        type=str,
        default="data/target.jpg",
        help="Path to the source image.",
    )
    args = parser.parse_args()
    return args

def createFirstWindow(root):
    h, w, _ = src.shape
    canvas = Canvas(root,width=w,height=h)
    canvas.config(highlightthickness=0)
    canvas.pack()

    src_img = ImageTk.PhotoImage(image=Image.fromarray(src[:,:,::-1]))
    canvas.create_image(0, 0, anchor='nw', image=src_img)

    # variables
    vertices = []
    lines = []
    tmp_line = []
    vars = {
        'flag': False
    }

    # event listeners
    def mouseLeftCallback(event):
        if not vars['flag']:
            if len(vertices) > 0:
                last_x, last_y = vertices[-1]
                first_x, first_y = vertices[0]
                if len(vertices) > 2 and (event.x - first_x) ** 2 + (event.y - first_y) ** 2 < 10 ** 2:
                    vars['flag'] = True
                    lines.append(canvas.create_line(last_x, last_y, first_x, first_y, width=2))
                    btn3.config(state=NORMAL)
                else:
                    lines.append(canvas.create_line(last_x, last_y, event.x, event.y, width=2))
            vertices.append((event.x, event.y))
            btn2.config(state=NORMAL)

    def mouseMoveCallback(event):
        if len(tmp_line) > 0:
            canvas.delete(tmp_line[-1])
            tmp_line.pop()
        if len(vertices) > 0 and not vars['flag']:
            last_x, last_y = vertices[-1]
            first_x, first_y = vertices[0]
            if len(vertices) > 2 and (event.x - first_x) ** 2 + (event.y - first_y) ** 2 < 10 ** 2:
                tmp_line.append(canvas.create_line(last_x, last_y, first_x, first_y, width=2, fill='green'))
            else:
                tmp_line.append(canvas.create_line(last_x, last_y, event.x, event.y, width=2, fill='red'))

    canvas.bind('<Button-1>', mouseLeftCallback)
    canvas.bind('<Motion>', mouseMoveCallback)

    def Btn1Callback():
        for widget in root.winfo_children():
            widget.destroy()
        createFirstWindow(root)

    def Btn2Callback():
        if len(vertices) > 0:
            vertices.pop()
            if len(vertices) == 0:
                btn2.config(state=DISABLED)
        if len(lines) > 0:
            canvas.delete(lines[-1])
            lines.pop()
        if len(tmp_line) > 0:
            canvas.delete(tmp_line[-1])
            tmp_line.pop()
        vars['flag'] = False
        btn3.config(state=DISABLED)

    def Btn3Callback():
        if vars['flag']:
            img = Image.new('L', (w, h), 0)
            ImageDraw.Draw(img).polygon(vertices, outline=1, fill=1)
            mask = np.array(img) * 255
            # cv2.imwrite('mask.png', mask)
            for widget in root.winfo_children():
                widget.destroy()
            createSecondWindow(root, mask)

    button_frame = Frame(root)
    button_frame.pack(fill=X, side=BOTTOM)

    btn1 = Button(button_frame, text='Reset', command=Btn1Callback)
    btn2 = Button(button_frame, text='Undo', command=Btn2Callback, state=DISABLED)
    btn3 = Button(button_frame, text='Next', command=Btn3Callback, state=DISABLED)

    button_frame.columnconfigure(0, weight=1)
    button_frame.columnconfigure(1, weight=1)
    button_frame.columnconfigure(2, weight=1)

    btn1.grid(row=0, column=0, sticky=W+E)
    btn2.grid(row=0, column=1, sticky=W+E)
    btn3.grid(row=0, column=2, sticky=W+E)

    root.mainloop()

def createSecondWindow(root, mask):
    masked = np.dstack((src, mask))
    # cv2.imwrite('masked.png', masked)

    h, w, _ = tar.shape
    canvas = Canvas(root,width=w,height=h)
    canvas.config(highlightthickness=0)
    canvas.pack()

    tar_img = ImageTk.PhotoImage(image=Image.fromarray(tar[:,:,::-1]))
    masked_img = ImageTk.PhotoImage(image=Image.fromarray(masked[:,:,[2,1,0,3]]))

    imgs = []
    imgs.append(canvas.create_image(0, 0, anchor='nw', image=tar_img))
    imgs.append(canvas.create_image(0, 0, anchor='nw', image=masked_img))

    # variables
    vars = {
        'pos_x': 0,
        'pos_y': 0,
        'start_x': 0,
        'start_y': 0,
    }

    # event listeners
    def mouseLeftCallback(event):
        vars['start_x'], vars['start_y'] = event.x, event.y

    def mouseLeftReleaseCallback(event):
        vars['pos_x'] = vars['pos_x'] + (event.x - vars['start_x'])
        vars['pos_y'] = vars['pos_y'] + (event.y - vars['start_y'])

    def mouseLeftMoveCallback(event):
        canvas.delete(imgs[-1])
        imgs.pop()
        imgs.append(
            canvas.create_image(
                vars['pos_x'] + (event.x - vars['start_x']), 
                vars['pos_y'] + (event.y - vars['start_y']), 
                anchor='nw', image=masked_img
            )
        )

    canvas.bind('<Button-1>', mouseLeftCallback)
    canvas.bind('<ButtonRelease-1>', mouseLeftReleaseCallback)
    canvas.bind('<B1-Motion>', mouseLeftMoveCallback)
    
    def Btn1Callback():
        for widget in root.winfo_children():
            widget.destroy()
        createFirstWindow(root)

    def Btn2Callback():
        result = MVC_Clone(src, tar, mask, vars['pos_x'], vars['pos_y'])
        #result = MVC_Clone_Dummy(src, tar, mask, vars['pos_x'], vars['pos_y'])
        cv2.imwrite('result.jpg', result)
        for widget in root.winfo_children():
            widget.destroy()
        createThirdWindow(root, result)

    button_frame = Frame(root)
    button_frame.pack(fill=X, side=BOTTOM)

    btn1 = Button(button_frame, text='Back', command=Btn1Callback)
    btn2 = Button(button_frame, text='Next', command=Btn2Callback)

    button_frame.columnconfigure(0, weight=1)
    button_frame.columnconfigure(1, weight=1)

    btn1.grid(row=0, column=0, sticky=W+E)
    btn2.grid(row=0, column=1, sticky=W+E)

    root.mainloop()

def createThirdWindow(root, result):
    h, w, _ = result.shape
    canvas = Canvas(root,width=w,height=h)
    canvas.config(highlightthickness=0)
    canvas.pack()

    result_img = ImageTk.PhotoImage(image=Image.fromarray(result[:,:,::-1]))
    canvas.create_image(0, 0, anchor='nw', image=result_img)

    # event listeners
    def Btn1Callback():
        for widget in root.winfo_children():
            widget.destroy()
        createFirstWindow(root)

    button_frame = Frame(root)
    button_frame.pack(fill=X, side=BOTTOM)

    btn1 = Button(button_frame, text='Restart', command=Btn1Callback)

    button_frame.columnconfigure(0, weight=1)

    btn1.grid(row=0, column=0, sticky=W+E)

    root.mainloop()

if __name__ == '__main__':
    args = parse_args()
    
    src = cv2.imread(args.source_img)
    tar = cv2.imread(args.target_img)
    
    root = Tk()
    root.title('MVCClone')
    root.resizable(False, False)
    createFirstWindow(root)