from RAG_Engine.contracts.interfaces import QueryGeneratorInterface

class QueryGenerator(QueryGeneratorInterface):

    def __init__(
        self,
        query_chain,
        n_queries: int = 4
    ):
        self.query_chain = query_chain
        self.n_queries = n_queries

    def generate(self, question: str) -> list[str]:

        raw_results = self.query_chain.invoke({
            "question": question,
            "n_queries": self.n_queries
        })

        final = [q.strip() for q in raw_results if q.strip()]
        return final[:self.n_queries]




