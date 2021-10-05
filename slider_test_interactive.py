""" 一つのウィンドウ内にスライダーと画像を表示し，
スライダーの操作によって画像をリアルタイム更新させる検証。
想定はスライダー2本と画像，ラベル二つ """
import tkinter as tk

root = tk.Tk()
root.geometry('500x80')
label1 = tk.Label(root)
label1.pack()


def on_slider_changed(value):
    num = int(value)
    label1['text'] = "{} * 3 = {}".format(num, num*3)


slider1 = tk.Scale(root,
                   command=on_slider_changed,
                   orient='horizontal',
                   label="Average size",
                   length=500,
                   from_=0,
                   to=1000,
                   sliderlength=20)
slider1.pack()

root.mainloop()
