# guided eran code

This repository contains commented files from both ERAN and ELINA.
You can replace the original files with these ones in their respective folders :

In ~/SapoForNN/ERAN/tf_verify :
* [\_\_main\_\_.py](__main__.py)
* [eran.py](eran.py)
* [analyser.py](analyzer.py)
* [deeppoly_nodes.py](deeppoly_nodes.py)

In ~/ERAN/ELINA/python_interface :
* [fppoly.py](fppoly.py)

In ~/ERAN/ELINA/fppoly :
* [fppoly.c](fppoly.c)

## ERAN steps
You can find a graph representing these steps in [visual_call_stack.pdf](visual_call_stack.pdf)
* Run the analyze_box() function on the ERAN object.
    * Initialize the analyzer according to the chosen domain.
    * Run the analyzer with class function analyse().
        * Process the input layer with the transformer() function -> ELINA.
        * Process the hidden layers (FC then ReLU) with the transformer() function -> ELINA.
    * Find the dominant class of the output layer.