MiTgaze
======

.. image:: https://img.shields.io/pypi/v/MiTgaze.svg
    :target: https://pypi.python.org/pypi/MiTgaze
    :alt: Latest PyPI version


A python-based tool to analyse gaze behaviour recorded by an eye-tracking system.

Introduction
------------
This is the code I used to analyse the gaze data I collected while working at Max Planck Institute for Biological Cybernetics, Department of Prof Nikos Logothetis. The experiment that I designed, and implemented concerns the relationship between gaze behaviour and the low-level saliency feature of a complex image; features such as contrast, luminance, depth of field, etc. As the stimuli, I used photographs taken by Mr Bruce Barnbaum (barnbaum.com) who kindly shared the photographs he has used in his book, The Art of Photography, with me.  

This library was specifically designed to address that type of analysis. Since I used a Tobii eye-tracker and also used the Tobii Pro Lab software to record the gaze data, some of the functions you find in this library are specific to the format of Tobii Pro Lab exports. But I have tried to make the core functions as hardware-independent as possible. 

The experiments included EEG recordings too, but this library does not involve any EEG analysis. Such analyses were implemented in the MiTARES library, which you can find in my GitHub account. 


Documentation
-------------
The library, as it stands, is criminally under-documented. I hope I can find time to prepare a decent documentation. But looking at examples in the example folder can give an idea of what this library is capable of. 

License
-------
MIT license.

Dependencies
------------

.. code-block:: python
    
    matplotlib
    numpy
    pathlib
    pandas
    pillow
    joblib
    scikit-learn
    opencv-python
    
