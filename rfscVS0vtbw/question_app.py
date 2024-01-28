from Questions import Questions

question_prompts = [
    "First letter of the alphabet is :\na) a\nb) b\nc) c\n",
    "Second q\n",
]

questions = [
    Questions(question_prompts[0], "a"),
    Questions(question_prompts[1], "a"),
]


def run_questions():
    score = 0
    for question in questions:
        if input(question.question_prompt) == question.answer:
            score += 1
    print(f"Your score is: {score} / {len(question_prompts)}")


run_questions()
