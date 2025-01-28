import pytest

from goose import Result, flow, task


class CourseObjective(Result):
    objective: str


class NumberOfLearningOutcomes(Result):
    number: int


class LearningOutcome(Result):
    outcome: str


class QuizQuestion(Result):
    question: str
    answer: str


@task
async def course_objective() -> CourseObjective:
    return CourseObjective(objective="Learn how to code")


@task
async def number_of_learning_outcomes() -> NumberOfLearningOutcomes:
    return NumberOfLearningOutcomes(number=3)


@task
async def learning_outcome(
    *, objective: CourseObjective, previous_outcomes: list[LearningOutcome]
) -> LearningOutcome:
    return LearningOutcome(outcome="Learn Python")


@task
async def quiz_question(*, outcome: str) -> QuizQuestion:
    return QuizQuestion(question=f"What is the meaning of {outcome}?", answer=outcome)


@flow
async def course_content() -> None:
    objective = await course_objective()
    num_outcomes = await number_of_learning_outcomes()
    outcomes: list[LearningOutcome] = []
    for _ in range(num_outcomes.number):
        outcomes.append(
            await learning_outcome(objective=objective, previous_outcomes=outcomes)
        )

    for outcome in outcomes:
        await quiz_question(outcome=outcome.outcome)


@pytest.mark.asyncio
async def test_generate_course_content() -> None:
    with course_content.run() as state:
        await course_content.generate()

    quiz_questions = state.get_all(task=quiz_question)
    assert quiz_questions[0].result.question == "What is the meaning of Learn Python?"
    assert quiz_questions[0].result.answer == "Learn Python"
    assert len(quiz_questions) == 3
