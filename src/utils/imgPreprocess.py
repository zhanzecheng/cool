# -*- coding: utf-8 -*-
# @Time    : 2018/3/20 下午2:15
# @File    : imgPreprocess.py

import Image
import sys


root_dir = '/home/zzc/cool/'

def processImage(infile):
    try:
        im = Image.open(infile)
    except IOError:
        print "Cant load", infile
        sys.exit(1)
    i = 0
    mypalette = im.getpalette()

    try:
        while 1:
            im.putpalette(mypalette)
            new_im = Image.new("RGBA", im.size)
            new_im.paste(im)
            new_im.save('foo'+str(i)+'.png')

            i += 1
            im.seek(im.tell() + 1)

    except EOFError:
        pass # end of sequence

processImage(root_dir + 'data')