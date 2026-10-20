from RAG_Engine.contracts.interfaces import ResponseGeneratorInterface


class ResponseGenerator(ResponseGeneratorInterface):

    def __init__(self, rag_chain):
        self.rag_chain = rag_chain

    def generate(self, question: str, context: str) -> str:
  
        response = self.rag_chain.invoke({
            "context": context,
            "question": question
        })

        return response


