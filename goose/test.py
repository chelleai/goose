import asyncio
import random
import string

from goose.core import Flow, Node, task


@task
async def generate_random_sentence(*, num_words: int) -> str:
    return " ".join(
        ["".join(random.sample(string.ascii_lowercase, 10)) for _ in range(num_words)]
    )


@task
async def generate_random_goal() -> str:
    return "goal"


@task
async def copy_sentence(*, sentence: Node[str], times: int) -> str:
    print(f"Copying sentence {sentence.out} {times} times")
    return " ".join([sentence.out] * times)


@task
async def summarize(*, paragraph: Node[str], goal: Node[str]) -> str:
    return f"This paragraph has {len(paragraph.out)} characters"


@task
async def concatenate(*, sentences: list[Node[str]]) -> str:
    return " ".join([sentence.out for sentence in sentences])


with Flow(name="random_multiply") as flow:
    sentence = generate_random_sentence(num_words=10)
    paragraph = copy_sentence(sentence=sentence, times=10)
    summaries: list[Node[str]] = []

    for _ in range(10):
        goal = generate_random_goal()
        summary = summarize(paragraph=paragraph, goal=goal)
        summaries.append(summary)

    full_summary = concatenate(sentences=summaries)

asyncio.run(flow.run())
print(full_summary.out)
