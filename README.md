﻿This is a Readme file. #
 You need to install certain python modules by going to your terminal and type:
 for onnxruntime module: pip install onnxruntime
 for PIL module: pip install pillow
 for rembg module: pip install rembg
 (Note: you will use some data connection during installation and during your first run, a file will be downloaded from github.)
 If you get this error, "OSError: cannot write mode RGBA as JPEG", this error is due to saving the removed background image as JPEG instead of png
 
 The file contains some commented codes that didn't work as desired.
 The code to convert an image to jpeg works fine.
 This code can remove background for jpeg, jpeg, these were the two I tested it with. You can also test it with other image types.

I saw this code on a facebook image belonging to clcoding.com
