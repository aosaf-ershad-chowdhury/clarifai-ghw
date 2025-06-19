import os
import json
from crewai import Agent, Task, Crew, Process, LLM

# CLARIFAI_PAT = os.getenv("CLARIFAI_PAT")

clarifai_llm = LLM(
    model="openai/deepseek-ai/deepseek-chat/models/DeepSeek-R1-Distill-Qwen-7B",
    base_url="https://api.clarifai.com/v2/ext/openai/v1",
    # Ensure this environment variable is set
    api_key=os.environ["CLARIFAI_PAT"]
)

gym_trainer = Agent(
    role="Gym Trainer",
    goal="Train a gym member to achieve their fitness goals",
    backstory="You are a gym trainer with expertise in fitness and nutrition. You will guide the gym member through their training regimen, providing advice on exercises, nutrition, and motivation.",
    llm=clarifai_llm,
    verbose=False,
)


def create_gym_advice_task(member_name: str, fitness_goal: str, current_fitness_level: str) -> Task:
    return Task(
        name="Gym Advice",
        description=f"Provide personalized fitness and nutrition advice to {member_name}. Their fitness goal is {fitness_goal} and current fitness level is {current_fitness_level}.",
        agent=gym_trainer,
        expected_output="Detailed fitness/nutrition plan and motivation"
    )


def run_gym(member_name: str, fitness_goal: str, current_fitness_level: str):
    task = create_gym_advice_task(
        member_name, fitness_goal, current_fitness_level)

    crew = Crew(
        name="Gym Training Crew",
        agents=[gym_trainer],
        tasks=[task],
        process=Process.sequential,
    )

    result = crew.kickoff()
    print(task.output)
    print(result)
    return task.output


if __name__ == "__main__":
    member_name = input("Enter the gym member's name: ")
    fitness_goal = input(
        "Enter the fitness goal (e.g., weight loss, muscle gain, endurance): ")
    current_fitness_level = input(
        "Enter the current fitness level (e.g., beginner, intermediate, advanced): ")

    result = run_gym(member_name, fitness_goal, current_fitness_level)
    print(result)
    # print(f"Advice for {member_name}: {result['advice']}")
    # print(f"Motivation for {member_name}: {result['motivation']}")
