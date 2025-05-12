# Chatting
# Importing libraries
# Jupyter-specific imports
from IPython.display import display, Markdown

def chat_interface(chain, question):
    """
    TO CHAT WITH THE PDF
    """
    return display(Markdown(chain.invoke(question)))