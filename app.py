import sys
from nltk_model import vader_predict
from hf_model import hf_predict

text = " ".join(sys.argv[1:]) or input("Enter text: ")
print("VADER      :", vader_predict(text))
print("Transformer:", hf_predict(text))