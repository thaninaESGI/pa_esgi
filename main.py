'''from dotenv import load_dotenv
import os
#from instantiation import HelpDesk

load_dotenv()
'''
'''if __name__ == '__main__':
    model = HelpDesk()

    prompt = 'Comment est-ce que la formation permet l’obtention de la Certification Professionnelle ?'
    #prompt = 'Quelles sont les modalités d''évaluation?'
    result, sources = model.retrieval_qa_inference(prompt)
    print(result, sources)'''

    # main.py
# Demo

from dotenv import load_dotenv
import os
os.environ['SSL_CERT_FILE'] = 'C:/Users/Sandro Lena/cacert.pem'
#from instantiation import HelpDesk

load_dotenv()

if __name__ == '__main__':
    from help_desk import HelpDesk

    model = HelpDesk(new_db=True)

    print(model.db._collection.count())

    prompt = 'Quelles sont les modalités d''évaluation pour la filière intelligence artificielle?'
    result, sources = model.retrieval_qa_inference(prompt, verbose=False)  # Désactivez l'impression des sources ici
    print(result)
