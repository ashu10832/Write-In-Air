# Write-In-The-Air

This project is used to detect your finger and track it as you move it around and draw it on the screen and then recognise the character drawn using Deep Learning. (Just like MAGIC..!!)

Made with :-
 ~ OpenCV
 ~ Python
 ~ Pytorch

 ![](output_gif.gif)

Skin tone of hand is extracted so that it works in different lighting conditions using Histogram and hand is pulled out from the image using Back-Projection.

Try this out yourself.
 - Clone the repo and run the finger.py script.
 - Keep your Palm inside the 5 red boxes and press 'a'
 - Point your index finger up like this - ☝🏻 and press 'd' to start drawing.
 - To save your drawing - Press 'f'.
 - To Clear the output - Press 'c'.
 - To quit - Press 'q'
 
 
 Convolutional Neural Network trained on EMNIST-Letter Dataset to recognise the characters drawn with the finger.
