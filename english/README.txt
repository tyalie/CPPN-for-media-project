start with $ bpython -i train_on_image.py

Then use the cmd:

>>> startReading("final.ckpt")
to load the final network configuration.

>>> openLoop( file_of_img, file_of_txt )
to start the process of rendering. 

This can now be used inside the server. 

Open the server with:
$ python -m SimpleHTTPServer 8000
in the server folder. Access the website and link the "openLoop" to out.png and c.txt. 
