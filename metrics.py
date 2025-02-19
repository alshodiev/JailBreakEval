def measure_harmfulness(response):
    """
    Placeholder function to measure the harmfulness of a response.
    Replace this with a real harmfulness classifier (e.g., OpenAI Moderation API, Perspective API).
    """
    harmful_keywords = ["bomb", "hate", "kill", "harm", "illegal"]
    harmfulness_score = sum(keyword in response.lower() for keyword in harmful_keywords)
    return harmfulness_score
