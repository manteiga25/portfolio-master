from numpy import argmax
from sentence_transformers import SentenceTransformer, util

class model:

    def __init__(self, base, response):

        self.knowledge_base = base

        self.response_base = response

        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        self.knowledge_embedding = self.model.encode(base, convert_to_numpy=True, normalize_embeddings=True)

    def answer_question(self, user_question):
        try:
            question_embedding = self.model.encode(user_question, convert_to_numpy=True, normalize_embeddings=True)

            similarities = util.cos_sim(question_embedding, self.knowledge_embedding)[0]

            max_index = argmax(similarities)

            if similarities[max_index] < 0.5:
                raise Exception("No similarity")
            most_similar_text = self.response_base[max_index]

            return most_similar_text
        except Exception as e:
            return None