def is_contaminated(p: str) -> bool:
    """
    Judge if a problem is contaminated.

    Parameters
    ----------
    p : str
        Problem name.

    Returns
    -------
    bool
        True if the problem contains shuffled modalities, False otherwise.
    """
    if "shuffle" not in p and "from" not in p:
        return False
    if p.startswith("<no_question>shuffle_image"):
        return False
    if p.startswith("<no_image>shuffle_question"):
        return False
    if p.startswith("<no_image>shuffled_image"):
        return False
    if p.startswith("<no_question>shuffled_question"):
        return False
    if "image_from" in p and "<no_image>" in p:
        return False
    if "question_from" in p and "<no_question>" in p:
        return False
    if "<no_info>" in p:
        return False
    return True
