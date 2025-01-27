import pytest

from goose.core import Result, RunContext, flow, task
from goose.types.agent import TextMessagePart, UserMessage
from goose.types.conversation import ConversationState


class CourseObjective(Result):
    objective: str


class LearningOutcomes(Result):
    content: list[str]


class Quiz(Result):
    name: str
    questions: list[str]


class Test(Result):
    name: str
    questions: list[str]


class Course(Result):
    objective: CourseObjective
    learning_outcomes: LearningOutcomes
    quizzes: list[Quiz]
    test: Test


@task
async def generate_course_objective() -> CourseObjective:
    return CourseObjective(objective="Learn about the course objective")


@generate_course_objective.adapter
async def adapt_course_objective(
    *, conversation_state: ConversationState[CourseObjective]
) -> CourseObjective:
    return CourseObjective(objective="Learn about the new course objective")


@task
async def generate_learning_outcomes(
    *,
    course_objective: CourseObjective,
) -> LearningOutcomes:
    return LearningOutcomes(
        content=[
            f"Learn about {course_objective.objective}",
            f"Learn about {course_objective.objective}",
        ]
    )


@generate_learning_outcomes.adapter
async def adapt_learning_outcomes(
    *, conversation_state: ConversationState[LearningOutcomes]
) -> LearningOutcomes:
    return LearningOutcomes(content=["Learn about new objective"])


@task
async def generate_quiz(*, learning_outcome: str) -> Quiz:
    return Quiz(name="Quiz 1", questions=[f"Question about {learning_outcome}"])


@generate_quiz.adapter
async def adapt_quiz(*, conversation_state: ConversationState[Quiz]) -> Quiz:
    return Quiz(name="Quiz 1", questions=["Question 1", "Question 2", "Question 3"])


@task
async def generate_test(*, quizzes: list[Quiz]) -> Test:
    return Test(name="Test 1", questions=["Question 1", "Question 2", "Question 3"])


@generate_test.adapter
async def adapt_test(*, conversation_state: ConversationState[Test]) -> Test:
    return Test(name="Test 1", questions=["Question 1", "Question 2", "Question 3"])


@flow
async def adapt_dynamic() -> Course:
    course_objective = await generate_course_objective()
    learning_outcomes = await generate_learning_outcomes(
        course_objective=course_objective
    )
    quizzes = [
        await generate_quiz(learning_outcome=outcome)
        for outcome in learning_outcomes.content
    ]
    test = await generate_test(quizzes=quizzes)

    return Course(
        objective=course_objective,
        learning_outcomes=learning_outcomes,
        quizzes=quizzes,
        test=test,
    )


@pytest.mark.asyncio
async def test_adapt_dynamic() -> None:
    run_context = RunContext()
    with adapt_dynamic.run(ctx=run_context):
        course = await adapt_dynamic.generate()
        new_learning_outcomes = await adapt_dynamic.adapt(
            message=UserMessage(parts=[TextMessagePart(text="Try again")]),
            target=course.learning_outcomes,
        )
        new_course = await adapt_dynamic.generate()

    assert (
        new_learning_outcome.result.content
        != course.learning_outcomes[0].result.content
    )
    assert new_course != course
