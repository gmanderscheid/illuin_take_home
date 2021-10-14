import pandas as pd
import string
import unidecode
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from numpy import dot
from numpy.linalg import norm

TRANSLATION_TABLE = str.maketrans(string.punctuation+string.ascii_uppercase," "*len(string.punctuation)+string.ascii_lowercase)
MY_STOPWORDS = stopwords.words('english')
MY_STOPWORDS.extend(['', ' '])
ps = PorterStemmer()

def get_cosin_sim(question, contexts):
    """
    Vectorises question and contexts to then mesure cosin between the question and all the contexts   
    Returns a series of all the cosins for one question and all it's contexts 
    """
    cos_sim_for_question = []
    for context in contexts :
        cv = CountVectorizer(stop_words=MY_STOPWORDS, lowercase=False)
        matrix = cv.fit_transform(pd.DataFrame([question, context])[0]).toarray()
        cos_sim = dot(matrix[0], matrix[1])/(norm(matrix[0])*norm(matrix[1]))
        cos_sim_for_question.append(cos_sim)
    return pd.Series(cos_sim_for_question)

def stem_text(text):
    """
    Stems, decodes and processes text 
    Returns a string 
    """
    return ps.stem(unidecode.unidecode(text.translate(TRANSLATION_TABLE)))

def main():
    '''
    Reads the validation file, picks a question at random and predicts which context corresponds the best 
    Returns the randomly sampled question and the corresponding "best" context
    '''

    validation = pd.read_csv('data/validation.csv')

    validation['processed_questions'] = validation.question.apply(stem_text)
    validation['processed_context'] = validation.correct_context.apply(stem_text)
    
    question = validation.processed_questions.sample().item()
    contexts = validation.processed_context.drop_duplicates()

    validate_cosins = get_cosin_sim(question, contexts)
    strongest_cosin_id = validate_cosins[validate_cosins == validate_cosins.max()].index
    best_prediction = validation.processed_context.drop_duplicates().iloc[strongest_cosin_id]

    print('Question : ', question)
    print('Best context : ', best_prediction)
    pd.DataFrame([question, best_prediction]).to_csv('./data/latest_prediction.csv')

    return question, validation.processed_context.drop_duplicates().iloc[strongest_cosin_id]


if __name__ == '__main__':
    main()