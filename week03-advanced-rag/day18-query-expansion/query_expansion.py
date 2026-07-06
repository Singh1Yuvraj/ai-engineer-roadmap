class QueryExpander:

    def __init__(self):

        self.synonyms = {

            "employee": [
                "worker",
                "staff"
            ],

            "employees": [
                "workers",
                "staff"
            ],

            "company": [
                "organization",
                "employer"
            ],

            "confidential": [
                "private",
                "proprietary"
            ],

            "secrets": [
                "trade secrets",
                "confidential information"
            ],

            "termination": [
                "ending",
                "dismissal"
            ],

            "contract": [
                "agreement"
            ]
        }

    def expand(self, question):

        expanded_queries = [question]

        lower_question = question.lower()

        for word, synonyms in self.synonyms.items():

            if word in lower_question:

                for synonym in synonyms:

                    expanded_queries.append(
                        lower_question.replace(
                            word,
                            synonym
                        )
                    )

        return list(set(expanded_queries))


if __name__ == "__main__":

    expander = QueryExpander()

    question = "Can employees reveal company secrets?"

    queries = expander.expand(question)

    print()

    for q in queries:

        print(q)