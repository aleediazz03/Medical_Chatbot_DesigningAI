from setuptools import find_packages, setup

'''
This code creates a setup.py file, which is used to package your Python project. 
Packaging means preparing your code so it can be installed easily on your computer or shared with others. 
The goal is to describe your project’s name, version, author, what code files should be included, and what libraries it needs to work.

You would create a setup.py like this when you want to organize your project professionally or make it easy to install with commands like pip install .. 
It’s especially important if you are working in a team, want to reuse your code later, or eventually publish your project (for example, on GitHub or PyPI). 
Without a setup.py, Python wouldn't know how to find all your files or what extra tools (like external libraries) your code depends on.

In short, this code turns your project into a package — like how apps are packaged in an app store — so that it’s clean, shareable, and installable.
'''

setup(
    name = 'Designing AI Medical Chatbot',
    version = '0.0.0',
    author = 'Alejandro Maria Juanba Julia Apsara',
    packages = find_packages(),
    install_requires = []
)