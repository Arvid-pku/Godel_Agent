

def action_evaluate_on_task(task, solver):
    """
    Evaluate the current solver on the goal task samples and return the evaluation feedback.

    Returns:
        feedback (str): Evaluation feedback including valid set accuracy, test set accuray, test sample inputs, model outputs and valid sample answer.
    """
    feedback, acc = task.evaluate(solver)
    if acc > mmlu.last_test_acc:
        logic.store_all_logic(f"../{mmlu.__name__}_{round(acc, 4)}")
        mmlu.last_test_acc = acc
    return feedback
