from numpy import argmax, array
from sklearn.metrics.pairwise import cosine_similarity
from fastembed import TextEmbedding

class model:

    def __init__(self, base, response):

        self.knowledge_base = base

        self.response_base = response

        self.model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

        self.knowledge_embedding = array(list(self.model.embed(base)))

    def answer_question(self, user_question):
        try:
            question_embedding = list(self.model.embed(user_question))

            similarities = cosine_similarity(question_embedding, self.knowledge_embedding)[0]

            max_index = argmax(similarities)

            if similarities[max_index] < 0.6:
                raise Exception("No similarity")
            most_similar_text = self.response_base[max_index]

            return most_similar_text
        except Exception as e:
            print(e)
            return None
